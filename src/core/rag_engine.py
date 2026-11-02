from __future__ import annotations

import json
import os
import re
from collections import Counter
from math import isfinite, log
from typing import Dict, List, Optional, Tuple

os.environ.setdefault("MODELSCOPE_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    import torch
except Exception:
    torch = None

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:
    AutoModel = None
    AutoTokenizer = None

from src.core.question_bank import parse_question_bank, question_number_hint
from src.utils.path import DATA_DIR, EMBED_MODEL_DIR, STORAGE_DIR


INDEX_VERSION = 3
DEFAULT_MAX_CHARS = 2000
DEFAULT_OVERLAP = 1


def _iter_txt_files(root_dir: str) -> List[str]:
    files: List[str] = []
    if not os.path.exists(root_dir):
        return files
    for base, _, names in os.walk(root_dir):
        for name in names:
            if name.lower().endswith(".txt"):
                files.append(os.path.join(base, name))
    files.sort()
    return files


def _paragraphs_with_line_ranges(lines: List[str]) -> List[Tuple[int, int, str]]:
    paras: List[Tuple[int, int, str]] = []
    buf: List[str] = []
    start: Optional[int] = None
    split_start = re.compile(r"^\s*(?:Q\d+|题目\d+|\d+[\.\、\)]|[一二三四五六七八九十]+、)\s*.+")

    def flush(end_idx: int) -> None:
        nonlocal buf, start
        if buf:
            paras.append((start or end_idx, end_idx, "\n".join(buf).strip()))
        buf = []
        start = None

    for idx, raw in enumerate(lines, start=1):
        line = raw.rstrip("\n")
        if line.strip() == "":
            if buf:
                flush(idx - 1)
            continue
        if split_start.match(line) and buf:
            flush(idx - 1)
        if start is None:
            start = idx
        buf.append(line)
    if buf:
        flush(len(lines))
    return [p for p in paras if p[2]]


def _chunk_paragraphs(
    paras: List[Tuple[int, int, str]],
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_paras: int = DEFAULT_OVERLAP,
) -> List[Tuple[int, int, str]]:
    if not paras:
        return []
    chunks: List[Tuple[int, int, str]] = []
    i = 0
    while i < len(paras):
        cur_texts: List[str] = []
        line_start = paras[i][0]
        line_end = paras[i][1]
        size = 0
        j = i
        while j < len(paras):
            p_start, p_end, p_text = paras[j]
            add_len = len(p_text) + (2 if cur_texts else 0)
            if cur_texts and size + add_len > max_chars:
                break
            cur_texts.append(p_text)
            size += add_len
            line_end = p_end
            j += 1
        chunks.append((line_start, line_end, "\n\n".join(cur_texts).strip()))
        if j >= len(paras):
            break
        i = max(i + 1, j - overlap_paras)
    return [c for c in chunks if c[2]]


def _split_question_text(
    text: str,
    start_line: int,
    max_chars: int,
) -> List[Tuple[int, int, str]]:
    if len(text) <= max_chars:
        return [(start_line, start_line + text.count("\n"), text)]
    parts: List[Tuple[int, int, str]] = []
    lines = text.splitlines()
    buf: List[str] = []
    size = 0
    line_no = start_line
    chunk_start = start_line
    for line in lines:
        add = len(line) + 1
        if buf and size + add > max_chars:
            parts.append((chunk_start, line_no - 1, "\n".join(buf).strip()))
            buf = [buf[-1]] if buf else []
            size = len(buf[-1]) + 1 if buf else 0
            chunk_start = line_no - 1
        buf.append(line)
        size += add
        line_no += 1
    if buf:
        parts.append((chunk_start, line_no - 1, "\n".join(buf).strip()))
    return [p for p in parts if p[2]]


def _question_chunks(fp: str, lines: List[str], max_chars: int) -> List[Dict]:
    questions = parse_question_bank(lines)
    if len(questions) < 2:
        return []
    base = os.path.basename(fp)
    out: List[Dict] = []
    for q in questions:
        pieces = _split_question_text(q.raw_text, q.start_line, max_chars)
        for idx, (ls, le, text) in enumerate(pieces, start=1):
            suffix = "" if len(pieces) == 1 else f":p{idx}"
            out.append(
                {
                    "chunk_id": f"{base}:q{q.qid:03d}{suffix}",
                    "file_name": base,
                    "file_path": fp,
                    "line_start": ls,
                    "line_end": le,
                    "text": text,
                    "qid": q.qid,
                    "qtype": q.qtype,
                    "score": q.score,
                    "stem": q.stem,
                    "options": q.options,
                }
            )
    return out


def _build_chunks(
    data_path: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    overlap_paras: int = DEFAULT_OVERLAP,
) -> List[Dict]:
    chunks_out: List[Dict] = []
    for fp in _iter_txt_files(data_path):
        with open(fp, "r", encoding="utf-8") as f:
            lines = f.readlines()
        q_chunks = _question_chunks(fp, lines, max_chars)
        if q_chunks:
            chunks_out.extend(q_chunks)
            continue
        paras = _paragraphs_with_line_ranges(lines)
        chunks = _chunk_paragraphs(paras, max_chars=max_chars, overlap_paras=overlap_paras)
        base = os.path.basename(fp)
        for idx, (ls, le, text) in enumerate(chunks, start=1):
            chunks_out.append(
                {
                    "chunk_id": f"{base}:c{idx:04d}",
                    "file_name": base,
                    "file_path": fp,
                    "line_start": ls,
                    "line_end": le,
                    "text": text,
                    "qid": None,
                    "qtype": None,
                    "score": None,
                    "stem": None,
                    "options": [],
                }
            )
    return chunks_out


class _SentenceTransformerEmbedder:
    def __init__(self, model_dir: str):
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_dir, device=device, local_files_only=True)
        self.model.max_seq_length = 512
        get_dim = getattr(self.model, "get_embedding_dimension", None)
        if get_dim is None:
            get_dim = self.model.get_sentence_embedding_dimension
        self.dim = int(get_dim())

    def encode(self, texts: List[str], batch_size: int = 32, is_query: bool = False) -> "torch.Tensor":
        if torch is None:
            raise RuntimeError("缺少 torch，无法计算向量。")
        arr = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return torch.as_tensor(arr, dtype=torch.float32)


class _TransformersEmbedder:
    def __init__(self, model_dir: str):
        if torch is None or AutoModel is None or AutoTokenizer is None:
            raise RuntimeError("缺少 torch/transformers，无法启用向量检索。")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
        self.model = AutoModel.from_pretrained(model_dir, local_files_only=True)
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.dim = int(self.model.config.hidden_size)

    def encode(self, texts: List[str], batch_size: int = 32, is_query: bool = False) -> "torch.Tensor":
        if torch is None:
            raise RuntimeError("缺少 torch，无法计算向量。")
        outs: List[torch.Tensor] = []
        with torch.inference_mode():
            for i in range(0, len(texts), batch_size):
                batch = texts[i : i + batch_size]
                encoded = self.tokenizer(
                    batch,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                hidden = self.model(**encoded).last_hidden_state
                pooled = hidden[:, 0]
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
                outs.append(pooled.cpu())
        return torch.cat(outs, dim=0) if outs else torch.empty((0, self.dim))


def _make_embedder(model_dir: str):
    if not os.path.isdir(model_dir):
        return None
    try:
        return _SentenceTransformerEmbedder(model_dir)
    except Exception:
        pass
    try:
        return _TransformersEmbedder(model_dir)
    except Exception:
        return None


def _tokenize(text: str) -> List[str]:
    tokens: List[str] = []
    for m in re.finditer(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]+", text):
        s = m.group(0)
        if s[0].isascii():
            tokens.append(s.lower())
        else:
            tokens.extend(s)
            tokens.extend(s[i : i + 2] for i in range(len(s) - 1))
    return tokens


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", "", text or "").strip().lower()


class _BM25Index:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.tfs: List[Counter] = []
        self.doc_len: List[int] = []
        df: Counter = Counter()
        for c in chunks:
            toks = _tokenize(c.get("text", ""))
            tf = Counter(toks)
            self.tfs.append(tf)
            self.doc_len.append(sum(tf.values()))
            for t in tf.keys():
                df[t] += 1
        self.avgdl = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        n = len(chunks)
        self.idf = {t: log((n - f + 0.5) / (f + 0.5) + 1.0) for t, f in df.items()}

    def scores(self, query: str) -> List[float]:
        if not self.chunks:
            return []
        q_toks = _tokenize(query)
        if not q_toks:
            return [0.0] * len(self.chunks)
        k1 = 1.5
        b = 0.75
        out: List[float] = []
        for i, tf in enumerate(self.tfs):
            dl = self.doc_len[i] or 1
            s = 0.0
            for t in q_toks:
                f = tf.get(t, 0)
                if f <= 0:
                    continue
                idf = self.idf.get(t, 0.0)
                denom = f + k1 * (1 - b + b * (dl / (self.avgdl or 1.0)))
                s += idf * (f * (k1 + 1) / denom)
            out.append(float(s) if isfinite(s) else 0.0)
        return out


def _data_signature(data_path: str, max_chars: int, overlap_paras: int, embed_dir: str) -> str:
    items = []
    for fp in _iter_txt_files(data_path):
        st = os.stat(fp)
        items.append(
            {
                "path": os.path.relpath(fp, data_path),
                "size": st.st_size,
                "mtime_ns": st.st_mtime_ns,
            }
        )
    payload = {
        "version": INDEX_VERSION,
        "max_chars": max_chars,
        "overlap": overlap_paras,
        "embed_dir": os.path.basename(embed_dir),
        "files": items,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _normalize_scores(scores: List[float]) -> List[float]:
    if not scores:
        return []
    mx = max(scores)
    if mx <= 0:
        return [0.0] * len(scores)
    return [max(0.0, s) / mx for s in scores]


class RAICOMSearcher:
    def __init__(
        self,
        data_path: str = DATA_DIR,
        persist_dir: str = STORAGE_DIR,
        similarity_top_k: int = 4,
        force_rebuild: bool = False,
        max_chars_per_chunk: int = DEFAULT_MAX_CHARS,
        overlap_paragraphs: int = DEFAULT_OVERLAP,
        embed_model_dir: str = EMBED_MODEL_DIR,
    ):
        self.data_path = os.path.abspath(data_path)
        self.persist_dir = os.path.abspath(persist_dir)
        self.similarity_top_k = similarity_top_k
        self.max_chars_per_chunk = max_chars_per_chunk
        self.overlap_paragraphs = overlap_paragraphs
        self.embed_model_dir = embed_model_dir
        os.makedirs(self.persist_dir, exist_ok=True)

        self._chunks: List[Dict] = []
        self._embeddings: Optional["torch.Tensor"] = None
        self._embedder = None
        self._bm25: Optional[_BM25Index] = None
        self._load_or_build(force_rebuild)

    def _paths(self) -> Tuple[str, str, str]:
        return (
            os.path.join(self.persist_dir, "rag_chunks.json"),
            os.path.join(self.persist_dir, "rag_embeddings.pt"),
            os.path.join(self.persist_dir, "rag_meta.json"),
        )

    def _load_or_build(self, force_rebuild: bool) -> None:
        chunks_path, emb_path, meta_path = self._paths()
        signature = _data_signature(
            self.data_path,
            self.max_chars_per_chunk,
            self.overlap_paragraphs,
            self.embed_model_dir,
        )
        cached = False
        if not force_rebuild and os.path.exists(chunks_path) and os.path.exists(meta_path):
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                if meta.get("signature") == signature:
                    with open(chunks_path, "r", encoding="utf-8") as f:
                        self._chunks = json.load(f)
                    cached = bool(self._chunks)
            except Exception:
                cached = False

        if not cached:
            self._chunks = _build_chunks(
                self.data_path,
                max_chars=self.max_chars_per_chunk,
                overlap_paras=self.overlap_paragraphs,
            )
            with open(chunks_path, "w", encoding="utf-8") as f:
                json.dump(self._chunks, f, ensure_ascii=False)
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump({"signature": signature, "count": len(self._chunks)}, f, ensure_ascii=False)

        self._bm25 = _BM25Index(self._chunks)
        self._embedder = _make_embedder(self.embed_model_dir)
        if self._embedder is None:
            self._embeddings = None
            return

        texts = [c.get("text", "") for c in self._chunks]
        if not texts:
            self._embeddings = None
            return

        loaded = False
        if not force_rebuild and os.path.exists(emb_path):
            try:
                emb = torch.load(emb_path, map_location="cpu", weights_only=True)
                if hasattr(emb, "shape") and emb.shape[0] == len(texts) and emb.shape[1] == self._embedder.dim:
                    self._embeddings = emb.to(torch.float32)
                    loaded = True
            except Exception:
                loaded = False
        if not loaded:
            self._embeddings = self._embedder.encode(texts, batch_size=32)
            try:
                torch.save(self._embeddings, emb_path)
            except Exception:
                pass

    @property
    def question_catalog(self) -> List[Dict]:
        return [c for c in self._chunks if c.get("qid") is not None]

    @property
    def embedding_dim(self) -> int:
        if self._embeddings is None or not hasattr(self._embeddings, "shape"):
            return 0
        if len(self._embeddings.shape) < 2:
            return 0
        return int(self._embeddings.shape[1])

    def _query_embedding(self, query: str) -> Optional["torch.Tensor"]:
        if self._embedder is None:
            return None
        try:
            emb = self._embedder.encode([query], batch_size=1, is_query=True)
        except Exception:
            return None
        if emb.numel() == 0:
            return None
        return emb[0].to(torch.float32)

    def _exact_bonus(self, query: str, chunk: Dict, qid_hint: Optional[int]) -> float:
        bonus = 0.0
        if qid_hint is not None and chunk.get("qid") == qid_hint:
            bonus += 2.0
        q_norm = _norm_text(query)
        stem_norm = _norm_text(str(chunk.get("stem") or ""))
        if stem_norm and (stem_norm in q_norm or q_norm in stem_norm):
            bonus += 0.6
        for opt in chunk.get("options") or []:
            opt_norm = _norm_text(str(opt.get("text") or ""))
            if opt_norm and len(opt_norm) >= 4 and opt_norm in q_norm:
                bonus += 0.05
        return bonus

    def retrieve_chunks(self, query: str, top_k: int = 6) -> List[Dict]:
        if not self._chunks:
            return []
        qid_hint = question_number_hint(query)
        bm25_raw = self._bm25.scores(query) if self._bm25 else [0.0] * len(self._chunks)
        bm25_norm = _normalize_scores(bm25_raw)

        vec_norm: List[float] = [0.0] * len(self._chunks)
        q_emb = self._query_embedding(query) if self._embeddings is not None else None
        if q_emb is not None and self._embeddings is not None:
            vec = torch.matmul(self._embeddings, q_emb)
            vec_norm = _normalize_scores(vec.tolist())

        scored: List[Tuple[float, int]] = []
        for i, chunk in enumerate(self._chunks):
            exact = self._exact_bonus(query, chunk, qid_hint)
            score = 0.55 * vec_norm[i] + 0.30 * bm25_norm[i] + exact
            if score > 0:
                scored.append((score, i))
        scored.sort(reverse=True, key=lambda x: x[0])

        out: List[Dict] = []
        for score, i in scored[: max(top_k, 0)]:
            c = dict(self._chunks[i])
            c["score"] = float(score)
            c["vector_score"] = float(vec_norm[i])
            c["bm25_score"] = float(bm25_norm[i])
            out.append(c)
        return out

    def format_context(self, chunks: List[Dict]) -> str:
        parts: List[str] = []
        for idx, c in enumerate(chunks, start=1):
            header = (
                f"[来源{idx}] 文件={c.get('file_name')} 行={c.get('line_start')}-{c.get('line_end')} "
                f"chunk_id={c.get('chunk_id')}"
            )
            if c.get("qid") is not None:
                header += f" 题号={c.get('qid')}"
            parts.append(header)
            parts.append(str(c.get("text", "")))
        return "\n\n".join(parts).strip()

    def search(self, query: str, top_k: Optional[int] = None) -> str:
        k = top_k or self.similarity_top_k
        chunks = self.retrieve_chunks(query, top_k=k)
        if not chunks:
            return "未在本地文档中找到相关内容。"
        return self.format_context(chunks)
