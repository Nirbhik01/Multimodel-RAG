import ollama
import logging

logger = logging.getLogger(__name__)

_HYDE_PROMPT = (
    "You are an expert radiologist. Your task is to output a brief, structured hypothetical radiology report "
    "based on the chest X-ray image (if provided) and the clinical note/query below.\n\n"
    "Clinical note/query: {query}\n\n"
    "Quality check instructions:\n"
    "1. First, check the quality of the clinical note. If it is already a detailed, highly technical, and high-quality "
    "radiological description or report, do NOT fabricate or add extra findings. Make NO modifications"
    "2. If the clinical note is brief, vague, or non-technical (e.g., 'cough', 'shortness of breath'), use the chest X-ray "
    "image and your medical knowledge to expand it into a professional, structured radiological report.\n\n"
    "Write ONLY the report with these two sections — no preamble, no explanation, no commentary of your quality check:\n"
    "FINDINGS: <describe visible abnormalities, lung fields, cardiac silhouette, bony structures>\n"
    "IMPRESSION: <concise diagnosis or summary>\n\n"
    "Keep it under 120 words. Use standard radiological terminology."
)


def generate_hypothetical_report(
    query_text: str,
    image_bytes: bytes | None,
    model_name: str,
) -> str:
    """
    HyDE: generate a hypothetical radiology report from the query + optional image.
    The returned text replaces the raw query for PubMedBERT embedding and BM25
    retrieval — bringing the query embedding closer to the stored report embeddings.
    Falls back to raw query_text if the Ollama call fails.
    """
    prompt = _HYDE_PROMPT.format(query=query_text or "No clinical note provided.")
    message: dict = {"role": "user", "content": prompt}
    if image_bytes:
        message["images"] = [image_bytes]

    try:
        response = ollama.chat(model=model_name, messages=[message], options={"temperature": 0.0})
        report = response["message"]["content"].strip()
        logger.info("[HyDE] Generated hypothetical report (%d chars)", len(report))
        return report
    except Exception as e:
        logger.error("[HyDE] Generation failed, falling back to raw query: %s", e)
        return query_text or ""
