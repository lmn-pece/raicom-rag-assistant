from __future__ import annotations

import os
import re
from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional


QUESTION_RE = re.compile(
    r"^\s*(?:第\s*)?(\d+)\s*[\.、．\)）]\s*[（(]\s*(\d+)\s*分\s*[）)]\s*(.*)$"
)
QUESTION_NO_SCORE_RE = re.compile(r"^\s*(?:第\s*)?(\d+)\s*[\.、．\)）]\s+(\S.*)$")
OPTION_RE = re.compile(r"^\s*[-*•]?\s*([A-Za-z])\s*[\.、．\)）]\s*(\S.*)$")
SECTION_RE = re.compile(r"(单选题|多选题|不定项|判断题|案例题|简答题|问答题|填空题)")


@dataclass
class Question:
    qid: int
    score: Optional[int]
    stem: str
    qtype: str
    start_line: int
    end_line: int
    options: List[Dict[str, object]] = field(default_factory=list)
    raw_text: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    def label_set(self) -> List[str]:
        return [str(o.get("label")) for o in self.options]


def detect_section(line: str) -> Optional[str]:
    text = line.strip()
    if not text or text.startswith("-"):
        return None
    if not (
        text.startswith("#")
        or re.match(r"^[一二三四五六七八九十]+、", text)
        or re.match(r"^(单选题|多选题|判断题|案例题|简答题|问答题|填空题)", text)
    ):
        return None
    m = SECTION_RE.search(text)
    if not m:
        return None
    kind = m.group(1)
    if kind == "不定项":
        return "多选题"
    return kind


def _finish(cur: Optional[Dict[str, object]], lines: List[str], out: List[Question]) -> None:
    if not cur:
        return
    start = int(cur["start_line"])
    end = int(cur.get("end_line") or start)
    raw = "\n".join(ln.rstrip("\n") for ln in lines[start - 1 : end]).strip()
    out.append(
        Question(
            qid=int(cur["qid"]),
            score=cur.get("score"),
            stem=str(cur.get("stem", "")).strip(),
            qtype=str(cur.get("qtype") or "未知"),
            start_line=start,
            end_line=end,
            options=list(cur.get("options") or []),
            raw_text=raw,
        )
    )


def parse_question_bank(lines: Iterable[str]) -> List[Question]:
    raw_lines = [ln.rstrip("\n") for ln in lines]
    out: List[Question] = []
    cur: Optional[Dict[str, object]] = None
    current_type = "未知"

    for idx, line in enumerate(raw_lines, start=1):
        m = QUESTION_RE.match(line)
        no_score = False
        if not m:
            m = QUESTION_NO_SCORE_RE.match(line)
            no_score = m is not None
        if m:
            _finish(cur, raw_lines, out)
            if no_score:
                qid = int(m.group(1))
                score = None
                stem = m.group(2).strip()
            else:
                qid = int(m.group(1))
                score = int(m.group(2))
                stem = m.group(3).strip()
            cur = {
                "qid": qid,
                "score": score,
                "stem": stem,
                "qtype": current_type,
                "start_line": idx,
                "end_line": idx,
                "options": [],
            }
            continue

        section = detect_section(line)
        if section:
            _finish(cur, raw_lines, out)
            cur = None
            current_type = section
            continue

        if cur is None:
            continue

        om = OPTION_RE.match(line)
        if om:
            label = om.group(1).upper()
            cur["options"].append({"label": label, "text": om.group(2).strip()})
            cur["end_line"] = idx
            continue

        if not line.strip():
            continue

        if cur["options"]:
            last = cur["options"][-1]
            last["text"] = f"{last['text']}\n{line.strip()}".strip()
        else:
            cur["stem"] = f"{cur['stem']}\n{line.strip()}".strip()
        cur["end_line"] = idx

    _finish(cur, raw_lines, out)
    return [q for q in out if q.stem]


def parse_question_file(path: str) -> List[Question]:
    with open(path, "r", encoding="utf-8") as f:
        return parse_question_bank(f.readlines())


def extract_question_texts(text: str) -> List[str]:
    parsed = parse_question_bank(text.splitlines())
    if parsed:
        return [q.raw_text for q in parsed]

    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    blocks: List[str] = []
    cur: List[str] = []
    q_start = re.compile(r"^\s*(?:Q\d+|题目\d+|\d+)[\.\、\)\:]?\s+.+")

    def flush() -> None:
        nonlocal cur
        q = "\n".join([x.strip() for x in cur if x.strip()]).strip()
        if q:
            blocks.append(q)
        cur = []

    for ln in lines:
        if q_start.match(ln):
            if cur:
                flush()
            cur = [ln]
            continue
        if cur:
            if ln.strip() == "":
                flush()
            else:
                cur.append(ln)
    if cur:
        flush()
    if blocks:
        return blocks
    return [ln.strip() for ln in lines if ln.strip()]


def question_number_hint(query: str) -> Optional[int]:
    patterns = [
        r"第\s*(\d+)\s*题",
        r"题号\s*[:：]?\s*(\d+)",
        r"^\s*(\d+)\s*[\.、．\)）]",
    ]
    for p in patterns:
        m = re.search(p, query)
        if m:
            return int(m.group(1))
    return None


def find_question_file(root: str, preferred: Optional[str] = None) -> Optional[str]:
    if preferred and os.path.isfile(preferred):
        return preferred
    if os.path.isdir(root):
        for base, _, names in os.walk(root):
            for name in sorted(names):
                if name.lower().endswith(".txt"):
                    return os.path.join(base, name)
    return None
