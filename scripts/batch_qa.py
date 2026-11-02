import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import DOC_QA_SYSTEM_PROMPT, MAX_NEW_TOKENS
from src.core.question_bank import extract_question_texts, parse_question_file
from src.core.rag_engine import RAICOMSearcher


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def _load_questions(questions_path: Optional[str], data_dir: str) -> List[Tuple[int, str]]:
    if questions_path and os.path.exists(questions_path):
        parsed = parse_question_file(questions_path)
        if parsed:
            return [(q.qid, q.raw_text) for q in parsed]
        return [(i + 1, q) for i, q in enumerate(extract_question_texts(_read_text(questions_path)))]

    txt_files: List[str] = []
    for base, _, names in os.walk(data_dir):
        for name in names:
            if name.lower().endswith(".txt"):
                txt_files.append(os.path.join(base, name))
    txt_files.sort()
    out: List[Tuple[int, str]] = []
    seen = set()
    for fp in txt_files:
        for q in parse_question_file(fp):
            key = re.sub(r"\s+", " ", q.raw_text).strip()
            if key and key not in seen:
                seen.add(key)
                out.append((q.qid, q.raw_text))
    return out


def _build_user_prompt(question: str, chunks: List[Dict]) -> str:
    parts = [f"问题：{question}", "", "资料片段："]
    for c in chunks:
        cid = c.get("chunk_id")
        ls = c.get("line_start")
        le = c.get("line_end")
        header = f"[chunk_id={cid} lines={ls}-{le}]"
        parts.append(header)
        parts.append(c.get("text", ""))
        parts.append("")
    return "\n".join(parts).strip() + "\n"


def _generate(model, tokenizer, system_prompt: str, user_prompt: str, max_new_tokens: int = MAX_NEW_TOKENS) -> str:
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    gen = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


_CITE_RE = re.compile(r'^\-\s+(.+?)\s+\[(\d+)\-(\d+)\]\s+"(.+)"\s*$', re.M)


def _validate(output: str, chunks: List[Dict]) -> Tuple[bool, List[Dict]]:
    chunk_map: Dict[str, Dict] = {str(c.get("chunk_id")): c for c in chunks if c.get("chunk_id")}
    cites: List[Dict] = []
    for m in _CITE_RE.finditer(output):
        cid, ls, le, quote = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4)
        c = chunk_map.get(cid)
        if not c:
            continue
        if c.get("line_start") is not None and int(c["line_start"]) != ls:
            continue
        if c.get("line_end") is not None and int(c["line_end"]) != le:
            continue
        text = c.get("text", "")
        if quote not in text:
            continue
        cites.append({"chunk_id": cid, "line_start": ls, "line_end": le, "quote": quote})
    if "无法从资料中确定" in output:
        return True, cites
    if not output.startswith("答案："):
        return False, cites
    if "引用：" not in output:
        return False, cites
    if not cites:
        return False, cites
    return True, cites[:3]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--questions", default=None)
    ap.add_argument("--out-md", default="answers.md")
    ap.add_argument("--out-jsonl", default="answers.jsonl")
    ap.add_argument("--persist-dir", default="storage")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--max-questions", type=int, default=0)
    args = ap.parse_args()

    try:
        from src.model_loader import load_qwen

        model, tokenizer = load_qwen()
    except Exception as e:
        raise RuntimeError("无法加载本地大模型（依赖 torch/transformers/bitsandbytes 等）。") from e

    searcher = RAICOMSearcher(
        data_path=args.data_dir,
        persist_dir=args.persist_dir,
        similarity_top_k=args.top_k,
        force_rebuild=args.rebuild_index,
    )
    questions = _load_questions(args.questions, args.data_dir)
    if args.max_questions and args.max_questions > 0:
        questions = questions[: args.max_questions]

    md_lines: List[str] = []
    with open(args.out_jsonl, "w", encoding="utf-8") as fj:
        for qid, q in questions:
            chunks = searcher.retrieve_chunks(q, top_k=args.top_k)
            if not chunks:
                ans = "无法从资料中确定"
                cites: List[Dict] = []
            else:
                user_prompt = _build_user_prompt(q, chunks)
                ans = _generate(model, tokenizer, DOC_QA_SYSTEM_PROMPT, user_prompt, args.max_new_tokens)
                ok, cites = _validate(ans, chunks)
                if not ok:
                    retry_user = (
                        user_prompt
                        + "\n你的输出不符合格式或引用不来自资料片段。请严格按系统格式重写，只能依据资料片段。"
                    )
                    ans = _generate(model, tokenizer, DOC_QA_SYSTEM_PROMPT, retry_user, args.max_new_tokens)
                    ok, cites = _validate(ans, chunks)
                    if not ok:
                        ans = "无法从资料中确定"
                        cites = []

            md_lines.append(f"## {qid}. {q}\n\n{ans}\n")
            fj.write(
                json.dumps(
                    {
                        "qid": qid,
                        "question": q,
                        "answer": ans,
                        "citations": cites,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    with open(args.out_md, "w", encoding="utf-8") as fm:
        fm.write("\n".join(md_lines).strip() + "\n")


if __name__ == "__main__":
    main()
