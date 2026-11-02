import datetime
from typing import Optional

from src.config import RAG_TOP_K
from src.core.rag_engine import RAICOMSearcher
from src.utils.path import DATA_DIR, STORAGE_DIR

_searcher: Optional[RAICOMSearcher] = None


def get_searcher() -> RAICOMSearcher:
    global _searcher
    if _searcher is None:
        _searcher = RAICOMSearcher(
            data_path=DATA_DIR,
            persist_dir=STORAGE_DIR,
            similarity_top_k=RAG_TOP_K,
        )
    return _searcher


def get_current_time(*args, **kwargs) -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def read_competition_rules(query: str) -> str:
    return query_competition_rules(query)


def query_competition_rules(keyword: str) -> str:
    return get_searcher().search(keyword)


TOOLS = {
    "get_current_time": get_current_time,
    "read_competition_rules": read_competition_rules,
    "query_competition_rules": query_competition_rules,
}
