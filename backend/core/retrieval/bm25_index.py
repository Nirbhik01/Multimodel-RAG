import json
import re
import logging
from pathlib import Path

from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)

_bm25_index: BM25Okapi | None = None
_bm25_corpus_ids: list | None = None


def _tokenize(text: str) -> list[str]:
    if not text:
        return []
    return re.findall(r"\b[a-z0-9]+\b", text.lower())


def get_bm25_index() -> tuple[BM25Okapi, list]:
    global _bm25_index, _bm25_corpus_ids
    if _bm25_index is not None:
        return _bm25_index, _bm25_corpus_ids

    root_path = Path(__file__).parent.parent.parent.parent
    json_path = root_path / "data" / "processed" / "embedding_input.json"

    with open(json_path, "r") as f:
        data = json.load(f)

    _bm25_corpus_ids = []
    tokenized_corpus = []

    for item in data:
        text = item.get("cleaned_text") or item.get("text", "")
        _bm25_corpus_ids.append(item["id"])
        tokenized_corpus.append(_tokenize(text))

    _bm25_index = BM25Okapi(tokenized_corpus)
    logger.info(f"[BM25] Index built with {len(_bm25_corpus_ids)} documents.")
    return _bm25_index, _bm25_corpus_ids


def query_bm25(query_text: str, top_k: int = 10) -> list[tuple]:
    """Returns list of (original_id, bm25_score) sorted by descending score."""
    bm25, corpus_ids = get_bm25_index()
    tokens = _tokenize(query_text)
    scores = bm25.get_scores(tokens)
    ranked = sorted(zip(corpus_ids, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_k]
