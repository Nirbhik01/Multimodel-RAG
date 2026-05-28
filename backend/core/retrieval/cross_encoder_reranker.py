import logging
import os

logger = logging.getLogger(__name__)

_cross_encoder = None


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is not None:
        return _cross_encoder

    from sentence_transformers import CrossEncoder

    model_name = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    logger.info(f"[CrossEncoder] Loading model: {model_name}")
    _cross_encoder = CrossEncoder(model_name)
    logger.info("[CrossEncoder] Model loaded.")
    return _cross_encoder


def rerank(query_text: str, candidates: list, json_data: list, top_k: int = 3) -> list:
    """
    Second-stage reranker: scores (query, document) pairs with a cross-encoder.
    Returns:
        List of (original_id, cross_encoder_score) sorted descending.
    """
    if not query_text or not candidates:
        return candidates[:top_k]

    id_to_text = {
        item["id"]: (item.get("cleaned_text") or item.get("text", ""))
        for item in json_data
    }

    cross_encoder = get_cross_encoder()

    pairs = []
    valid_ids = []
    for oid, _ in candidates:
        pairs.append([query_text, id_to_text.get(oid, "")])
        valid_ids.append(oid)

    scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(valid_ids, scores.tolist()), key=lambda x: x[1], reverse=True)
    logger.info(f"[CrossEncoder] Reranked {len(ranked)} candidates, returning top {top_k}.")
    return ranked[:top_k]
