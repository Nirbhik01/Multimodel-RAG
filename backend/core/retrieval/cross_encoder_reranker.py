import logging
import os

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

logger = logging.getLogger(__name__)

_tokenizer = None
_model = None
_device = None


def _get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cross_encoder():
    global _tokenizer, _model, _device
    if _model is not None:
        return _tokenizer, _model, _device

    model_name = os.getenv("CROSS_ENCODER_MODEL", "ncbi/MedCPT-Cross-Encoder")
    _device = _get_device()

    logger.info("[CrossEncoder] Loading model: %s on %s", model_name, _device)
    _tokenizer = AutoTokenizer.from_pretrained(model_name)
    _model = AutoModelForSequenceClassification.from_pretrained(model_name)
    _model.to(_device)
    _model.eval()
    logger.info("[CrossEncoder] Model loaded.")

    return _tokenizer, _model, _device


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

    tokenizer, model, device = get_cross_encoder()

    pairs = []
    valid_ids = []
    for oid, _ in candidates:
        pairs.append([query_text, id_to_text.get(oid, "")])
        valid_ids.append(oid)

    with torch.no_grad():
        encoded = tokenizer(
            [p[0] for p in pairs],
            [p[1] for p in pairs],
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        logits = model(**encoded).logits.squeeze(-1)

    scores = logits.cpu().tolist()
    if isinstance(scores, float):
        scores = [scores]

    ranked = sorted(zip(valid_ids, scores), key=lambda x: x[1], reverse=True)
    logger.info("[CrossEncoder] Reranked %d candidates, returning top %d.", len(ranked), top_k)
    return ranked[:top_k]
