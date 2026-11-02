from __future__ import annotations

import re
import threading
from typing import Dict, List, Optional

from src.config import (
    AGENT_SYSTEM_PROMPT,
    ALLOW_MODEL_INFERENCE,
    INFERENCE_SYSTEM_PROMPT,
    INFERENCE_WARNING,
    MAX_NEW_TOKENS,
    RAG_TOP_K,
)
from src.core.agent_logic import get_searcher
from src.core.question_bank import question_number_hint
from src.core.rag_engine import RAICOMSearcher

_ANSWER_INTENT = re.compile(r"答案|正确选项|参考答案|选哪|选什么|选哪个|应该选|该选|正确.{0,4}选项")
_ANSWER_AVAILABILITY = re.compile(r"(有没有|是否|有无).{0,8}答案|答案.{0,8}(有没有|是否|有无)")
_COUNT_INTENT = re.compile(r"(多少|几)\s*道?.{0,4}题|题目数量|题目总数|总共有.{0,8}题|一共.{0,8}题|共.{0,8}道?题")
_LIST_INTENT = re.compile(
    r"(列出|列一下|列举|展示|看看|读出|读取|输出).{0,20}(题目|题号|题)|有哪些题|题目列表|题.{0,20}(读出|读取|输出)"
)
_HAS_ANSWER = re.compile(r"(答案|正确答案|参考答案|解析)\s*[：:]")


def _source_line(chunks: List[Dict]) -> str:
    parts: List[str] = []
    for idx, c in enumerate(chunks, start=1):
        parts.append(
            f"[来源{idx}] {c.get('file_name')} 行{c.get('line_start')}-{c.get('line_end')}"
        )
    return "引用来源：" + "；".join(parts) if parts else ""


def _chunk_has_answer(chunk: Dict) -> bool:
    return bool(_HAS_ANSWER.search(str(chunk.get("text") or "")))


def _generate(
    model,
    tokenizer,
    messages: List[Dict[str, str]],
    max_new_tokens: int = MAX_NEW_TOKENS,
    stream: bool = False,
) -> str:
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    kwargs = dict(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    if stream:
        from transformers import TextIteratorStreamer

        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        thread = threading.Thread(target=model.generate, kwargs={**kwargs, "streamer": streamer})
        thread.start()
        parts: List[str] = []
        for new_text in streamer:
            print(new_text, end="", flush=True)
            parts.append(new_text)
        thread.join()
        return "".join(parts).strip()

    out = model.generate(**kwargs)
    gen = out[0][inputs["input_ids"].shape[-1] :]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()


class RAICOMAgent:
    def __init__(
        self,
        model,
        tokenizer,
        searcher: Optional[RAICOMSearcher] = None,
        system_prompt: str = AGENT_SYSTEM_PROMPT,
        top_k: int = RAG_TOP_K,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.searcher = searcher or get_searcher()
        self.system_prompt = system_prompt
        self.top_k = top_k
        self.history: List[Dict[str, str]] = []
        self.last_chunks: List[Dict] = []
        self.last_answer: str = ""
        self.last_streamed: bool = False

    def reset(self) -> None:
        self.history = []
        self.last_chunks = []
        self.last_answer = ""
        self.last_streamed = False

    def _catalog_answer(self, query: str) -> Optional[str]:
        catalog = self.searcher.question_catalog
        if not catalog:
            return None
        files = sorted({str(c.get("file_name")) for c in catalog if c.get("file_name")})
        groups: Dict[str, List[int]] = {}
        for c in catalog:
            qtype = str(c.get("qtype") or "未知")
            groups.setdefault(qtype, []).append(int(c.get("qid")))
        lines = [f"本地资料（{'、'.join(files)}）共解析出 {len(catalog)} 道题："]
        order = ["单选题", "多选题", "判断题", "案例题"]
        for qtype in order + sorted(set(groups) - set(order)):
            ids = sorted(groups.get(qtype, []))
            if not ids:
                continue
            if len(ids) == 1:
                lines.append(f"- {qtype}：1 道（题号 {ids[0]}）")
            else:
                lines.append(f"- {qtype}：{len(ids)} 道（题号 {ids[0]}-{ids[-1]}）")
        if _LIST_INTENT.search(query):
            ordered = sorted(catalog, key=lambda c: int(c.get("qid")))
            if "题号" in query and "题目" not in query:
                lines.append("全部题号：" + "、".join(str(int(c.get("qid"))) for c in ordered))
            else:
                lines.append("题目清单：")
                for c in ordered:
                    lines.append(f"{int(c.get('qid'))}. {c.get('stem')}")
        lines.append(_source_line(catalog[:1]))
        return "\n".join(line for line in lines if line)

    def _availability_answer(self) -> Optional[str]:
        catalog = self.searcher.question_catalog
        if not catalog:
            return None
        files = sorted({str(c.get("file_name")) for c in catalog if c.get("file_name")})
        answer_chunks = [c for c in catalog if _chunk_has_answer(c)]
        if answer_chunks:
            return (
                f"本地资料（{'、'.join(files)}）中包含答案字段，"
                f"共 {len(answer_chunks)} 个片段带有答案或解析。\n{_source_line(answer_chunks[:3])}"
            )
        return (
            f"本地资料（{'、'.join(files)}）只包含题干和选项，"
            "未提供“答案 / 正确答案 / 解析”字段，因此无法从本地资料中确定各题答案。\n"
            f"{_source_line(catalog[:1])}"
        )

    def _inference_answer(self, query: str, chunks: List[Dict], max_new_tokens: int) -> str:
        context = self.searcher.format_context(chunks)
        user_prompt = (
            "【本地资料】\n"
            f"{context}\n\n"
            "【用户问题】\n"
            f"{query}\n\n"
            f"请先输出“{INFERENCE_WARNING}”，再给出你的推断。"
        )
        messages = [{"role": "system", "content": INFERENCE_SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": user_prompt})
        answer = _generate(
            self.model,
            self.tokenizer,
            messages,
            max_new_tokens=max_new_tokens,
            stream=False,
        )
        if not answer.startswith(INFERENCE_WARNING):
            answer = f"{INFERENCE_WARNING}\n{answer}"
        if "以上推断不代表本地资料内容" not in answer:
            answer = f"{answer}\n\n以上推断不代表本地资料内容。"
        if "引用来源" not in answer:
            answer = f"{answer}\n\n{_source_line(chunks)}"
        return answer

    def answer(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_new_tokens: int = MAX_NEW_TOKENS,
        stream: bool = False,
    ) -> str:
        self.last_streamed = False
        if _ANSWER_AVAILABILITY.search(query):
            answer = self._availability_answer()
            if answer:
                self.last_chunks = self.searcher.question_catalog[:1]
                self.last_answer = answer
                return answer

        qid_hint = question_number_hint(query)
        if qid_hint is None and (_COUNT_INTENT.search(query) or _LIST_INTENT.search(query)):
            answer = self._catalog_answer(query)
            if answer:
                self.last_chunks = self.searcher.question_catalog[:1]
                self.last_answer = answer
                return answer

        k = top_k or self.top_k
        self.last_chunks = self.searcher.retrieve_chunks(query, top_k=k)
        if not self.last_chunks:
            answer = "无法从本地资料中确定：当前本地文档中没有检索到与该问题相关的内容。"
            self.last_answer = answer
            return answer

        if _ANSWER_INTENT.search(query):
            target_qid = question_number_hint(query)
            if target_qid is None:
                for c in self.last_chunks:
                    if c.get("qid") is not None:
                        target_qid = int(c["qid"])
                        break
            targets = [c for c in self.last_chunks if target_qid is not None and c.get("qid") == target_qid]
            if targets and not any(_chunk_has_answer(c) for c in targets):
                if ALLOW_MODEL_INFERENCE:
                    answer = self._inference_answer(query, targets, max_new_tokens)
                else:
                    answer = (
                        f"本地资料中未提供第{target_qid}题的答案。"
                        "本地资料仅包含题干和选项，无法据此确定答案。\n"
                        f"{_source_line(targets)}"
                    )
                self.last_answer = answer
                return answer

        context = self.searcher.format_context(self.last_chunks)
        user_prompt = (
            "【本地资料】\n"
            f"{context}\n\n"
            "【用户问题】\n"
            f"{query}\n\n"
            "请严格依据【本地资料】回答。资料中没有的信息必须明确说明，不得编造。"
            "回答末尾列出引用来源。"
        )
        messages = [{"role": "system", "content": self.system_prompt}]
        messages.extend(self.history[-4:])
        messages.append({"role": "user", "content": user_prompt})

        answer = _generate(
            self.model,
            self.tokenizer,
            messages,
            max_new_tokens=max_new_tokens,
            stream=stream,
        )
        self.last_streamed = bool(stream)
        self.history.append({"role": "user", "content": query})
        self.history.append({"role": "assistant", "content": answer})
        self.last_answer = answer
        return answer

    def think_and_act(self, query: str, stream: bool = False) -> str:
        return self.answer(query, stream=stream)
