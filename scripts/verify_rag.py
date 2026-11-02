from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.config import RAG_TOP_K
from src.core.question_bank import Question, parse_question_file
from src.core.rag_engine import RAICOMSearcher


def _norm(text: str) -> str:
    return re.sub(r"[^\w\u4e00-\u9fff]+", "", text or "").lower()


def _check_question_in_chunks(q: Question, chunks: List[Dict]) -> Tuple[bool, List[str]]:
    context = "\n".join(str(c.get("text", "")) for c in chunks)
    ctx_norm = _norm(context)
    missing: List[str] = []
    if _norm(q.stem) not in ctx_norm:
        missing.append("题干")
    for opt in q.options:
        text = str(opt.get("text") or "")
        if _norm(text) and _norm(text) not in ctx_norm:
            missing.append(f"选项{opt.get('label')}")
    return not missing, missing


def _check_answer_reading(q: Question, answer: str) -> Tuple[bool, List[str]]:
    ans_norm = _norm(answer)
    missing: List[str] = []
    if _norm(q.stem) not in ans_norm:
        missing.append("题干")
    for opt in q.options:
        text = str(opt.get("text") or "")
        if _norm(text) and _norm(text) not in ans_norm:
            missing.append(f"选项{opt.get('label')}")
    return not missing, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/样题.txt")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--persist-dir", default="storage")
    ap.add_argument("--top-k", type=int, default=RAG_TOP_K)
    ap.add_argument("--rebuild-index", action="store_true")
    ap.add_argument("--model", action="store_true", help="同时运行本地大模型读取测试")
    ap.add_argument("--limit", type=int, default=3, help="模型读取测试的题目数量")
    ap.add_argument("--qids", default=None, help="指定题号，如 1,5,71")
    ap.add_argument("--max-new-tokens", type=int, default=768)
    ap.add_argument("--out-md", default="reports/verify_report.md")
    ap.add_argument("--out-json", default="reports/verify_report.json")
    args = ap.parse_args()

    questions = parse_question_file(args.file)
    if not questions:
        raise SystemExit(f"未能从 {args.file} 解析出题目，请检查文件格式。")
    print(f"解析题目数量: {len(questions)}")

    searcher = RAICOMSearcher(
        data_path=args.data_dir,
        persist_dir=args.persist_dir,
        similarity_top_k=args.top_k,
        force_rebuild=args.rebuild_index,
    )
    if searcher.embedding_dim == 0:
        print("提示: 向量模型不可用，当前使用 BM25 检索。")
    else:
        print(f"向量检索已启用，向量维度: {searcher.embedding_dim}")

    retrieval_rows: List[Dict] = []
    retrieval_pass = 0
    exact_qid_pass = 0
    for q in questions:
        chunks = searcher.retrieve_chunks(q.raw_text, top_k=args.top_k)
        ok, missing = _check_question_in_chunks(q, chunks)
        exact_qid = any(int(c.get("qid") or -1) == q.qid for c in chunks)
        if ok:
            retrieval_pass += 1
        if exact_qid:
            exact_qid_pass += 1
        retrieval_rows.append(
            {
                "qid": q.qid,
                "type": q.qtype,
                "stem": q.stem,
                "retrieval_ok": ok,
                "exact_qid": exact_qid,
                "missing": missing,
                "top_chunks": [
                    {
                        "chunk_id": c.get("chunk_id"),
                        "qid": c.get("qid"),
                        "line_start": c.get("line_start"),
                        "line_end": c.get("line_end"),
                        "score": c.get("score"),
                    }
                    for c in chunks
                ],
            }
        )

    print(f"检索完整命中: {retrieval_pass}/{len(questions)}")
    print(f"目标题目进入 Top-{args.top_k}: {exact_qid_pass}/{len(questions)}")
    bad = [r for r in retrieval_rows if not r["retrieval_ok"]]
    for r in bad[:10]:
        print(f"  未完整命中 第{r['qid']}题，缺少: {', '.join(r['missing'])}")

    model_rows: List[Dict] = []
    model_pass = 0
    if args.model:
        from src.core.brain import RAICOMAgent
        from src.model_loader import load_qwen

        selected: List[Question]
        if args.qids:
            wanted = {int(x.strip()) for x in args.qids.split(",") if x.strip()}
            selected = [q for q in questions if q.qid in wanted]
        else:
            selected = questions[: max(args.limit, 0)]
        model, tokenizer = load_qwen()
        agent = RAICOMAgent(model, tokenizer, searcher=searcher, top_k=args.top_k)
        for q in selected:
            prompt = (
                f"请阅读本地资料，列出第{q.qid}题的完整题干和全部选项。"
                "不要作答，不要添加资料中没有的内容。"
            )
            answer = agent.answer(prompt, top_k=args.top_k, max_new_tokens=args.max_new_tokens)
            ok, missing = _check_answer_reading(q, answer)
            if ok:
                model_pass += 1
            model_rows.append(
                {
                    "qid": q.qid,
                    "ok": ok,
                    "missing": missing,
                    "answer": answer,
                }
            )
            state = "通过" if ok else f"缺少: {', '.join(missing)}"
            print(f"模型读取 第{q.qid}题: {state}")
        print(f"模型读取完整命中: {model_pass}/{len(selected)}")

    report = {
        "total_questions": len(questions),
        "top_k": args.top_k,
        "retrieval_pass": retrieval_pass,
        "exact_qid_pass": exact_qid_pass,
        "retrieval_rows": retrieval_rows,
        "model_tested": bool(args.model),
        "model_pass": model_pass,
        "model_rows": model_rows,
    }
    for path in (args.out_md, args.out_json):
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    lines = [
        "# 本地 RAG 验证报告",
        "",
        f"- 题目总数: {len(questions)}",
        f"- 检索完整命中: {retrieval_pass}/{len(questions)}",
        f"- 目标题目进入 Top-{args.top_k}: {exact_qid_pass}/{len(questions)}",
    ]
    if args.model:
        lines.append(f"- 模型读取完整命中: {model_pass}/{len(model_rows)}")
    lines.append("")
    if bad:
        lines.append("## 检索未完整命中")
        for r in bad:
            lines.append(f"- 第{r['qid']}题: 缺少 {', '.join(r['missing'])}")
        lines.append("")
    if model_rows:
        lines.append("## 模型读取结果")
        for r in model_rows:
            lines.append(f"### 第{r['qid']}题: {'通过' if r['ok'] else '未通过'}")
            if r["missing"]:
                lines.append(f"缺少: {', '.join(r['missing'])}")
            lines.append("")
            lines.append("```")
            lines.append(r["answer"][:3000])
            lines.append("```")
            lines.append("")
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"报告已写入: {args.out_md} / {args.out_json}")


if __name__ == "__main__":
    main()
