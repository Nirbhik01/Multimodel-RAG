from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response
from django.http import StreamingHttpResponse

from upstash_vector import Index

from core.embedding.generate_embeddings import get_text_embedding, get_image_embedding
from core.embedding.medical_models import get_medical_models

from core.retrieval.query_vector_db import query_vector_db
from core.retrieval.bm25_index import query_bm25
from core.retrieval.cross_encoder_reranker import rerank as ce_rerank

from core.generation.generate_answer import generate_answer

from datetime import datetime
import os
import re
import json
import base64
import io
import logging
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image
from bson import ObjectId

from .mongodb_utils import db_client

logger = logging.getLogger(__name__)
load_dotenv()

ROOT_PATH = Path(__file__).parent.parent.parent
JSON_DATA = []

try:
    with open(ROOT_PATH / "data" / "processed" / "embedding_input.json") as f:
        JSON_DATA = json.load(f)
except Exception as e:
    logger.error(f"Failed to load embedding_input.json: {e}")


SYSTEM_PROMPT = """
You are an expert Radiologist AI Assistant.

You are given:
1. A patient's X-ray and clinical text.
2. Multiple retrieved reference cases (anonymized as "Reference Case 1", "Reference Case 2", etc.).

Each reference case includes retrieval metadata:
- retrieval_sources: which search methods surfaced it (text_vector, image_vector, bm25) and their rank within that method.
- cross_encoder_score: semantic similarity between the patient query and the case report (higher = more relevant).
- rrf_rank: rank after fusing all retrieval signals via Reciprocal Rank Fusion.

Use retrieval metadata as a confidence signal:
- Cases found by multiple methods and with a high cross_encoder_score are strong candidates.
- Cases found only by bm25 may share keywords but have lower semantic similarity.
- cross_encoder_score is the most reliable single signal for text-based relevance.

Your tasks:
- Compare all retrieved cases against the patient presentation.
- Identify the best matching reference case. Refer to it only by its anonymized label (e.g., "Reference Case 1") — never by any ID, name, or sensitive identifier.
- Explain why it matches and note any conflicting evidence.
- Produce a final Findings section and a final Impression section.
- Assign a confidence level.
- Provide clinical suggestions based on the retrieved cases.
- Place slightly higher emphasis on the nature, location, and severity of airspace opacities or consolidations when comparing cases.

Output constraints:
- No meta-commentary, thought process, or internal guideline references in the response.
- No real IDs, names, or patient data — only "Reference Case X" labels.

Return output in this exact format with no extra text:

### Retrieval Analysis
- Best Matching Case: [anonymized label, e.g., Reference Case 1]
- Why It Matches (Similarity evidence):
- Conflicting Evidence:
- Confidence Level:

### Final Findings
...

### Final Impression
...

### Suggestions
...
"""


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def compress_image_bytes(image_bytes, max_size=(512, 512)):
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Error compressing image: {e}")
        return image_bytes


def get_image_b64(image_name, root_path):
    if not image_name:
        return None
    path = root_path / "data" / "images" / "images_normalized" / image_name
    if not path.exists():
        logger.error(f"Image not found: {path}")
        return None
    with open(path, "rb") as f:
        return f"data:image/png;base64,{base64.b64encode(f.read()).decode('utf-8')}"


def get_case_image_and_text(oid, json_data):
    for item in json_data:
        if item["id"] == oid:
            return item["image"], item["text"]
    return None, None


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

def _run_retrieval(query_text, query_image, index, text_model, image_model, image_processor):
    text_vectors, image_vectors, bm25_results = [], [], []

    if query_text and query_text.strip():
        text_embedding = get_text_embedding(text=query_text, model=text_model)
        text_vectors = query_vector_db(query_vector=text_embedding, index=index, vector_type="text")
        bm25_results = query_bm25(query_text, top_k=10)
        logger.info(f"[Retrieval] text={len(text_vectors)}, bm25={len(bm25_results)}")

    if query_image:
        image_embedding = get_image_embedding(image=query_image, processor=image_processor, model=image_model)
        image_vectors = query_vector_db(query_vector=image_embedding, index=index, vector_type="image")
        logger.info(f"[Retrieval] image={len(image_vectors)}")

    return text_vectors, image_vectors, bm25_results


def _rank_results(text_vectors, image_vectors, bm25_results, k=60):
    def build_ranks(vectors):
        sorted_v = sorted(vectors, key=lambda r: getattr(r, "score", 0.0), reverse=True)
        ranks = {}
        for idx, r in enumerate(sorted_v):
            if r.metadata and "original_id" in r.metadata:
                ranks.setdefault(r.metadata["original_id"], idx + 1)
        return ranks

    text_ranks = build_ranks(text_vectors)
    image_ranks = build_ranks(image_vectors)
    bm25_ranks = {oid: idx + 1 for idx, (oid, _) in enumerate(bm25_results or [])}

    all_ids = set(text_ranks) | set(image_ranks) | set(bm25_ranks)
    combined = {
        oid: sum(
            1.0 / (k + ranks[oid])
            for ranks in (text_ranks, image_ranks, bm25_ranks)
            if oid in ranks
        )
        for oid in all_ids
    }
    return combined, text_ranks, image_ranks, bm25_ranks


def _select_top_cases(query_text, text_vectors, image_vectors, bm25_results, json_data):
    combined, text_ranks, image_ranks, bm25_ranks = _rank_results(
        text_vectors, image_vectors, bm25_results
    )
    joint = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    rrf_pool = joint[:10]

    reranked = (
        ce_rerank(query_text, rrf_pool, json_data, top_k=3)
        if (query_text and query_text.strip())
        else rrf_pool[:3]
    )

    rrf_rank_lookup = {oid: i + 1 for i, (oid, _) in enumerate(joint)}

    annotated = []
    for oid, ce_score in reranked:
        sources = []
        if oid in text_ranks:
            sources.append(f"text_vector (rank {text_ranks[oid]})")
        if oid in image_ranks:
            sources.append(f"image_vector (rank {image_ranks[oid]})")
        if oid in bm25_ranks:
            sources.append(f"bm25 (rank {bm25_ranks[oid]})")

        annotated.append((oid, {
            "ce_score": round(ce_score, 4) if ce_score is not None else None,
            "rrf_rank": rrf_rank_lookup.get(oid),
            "sources": sources,
        }))

    return annotated


# ---------------------------------------------------------------------------
# Case loading & prompt building
# ---------------------------------------------------------------------------

def _load_reference_cases(top_cases, json_data, root_path):
    cases = []
    for oid, meta in top_cases:
        image_name, text = get_case_image_and_text(oid, json_data)
        img_bytes = None
        if image_name:
            path = root_path / "data" / "images" / "images_normalized" / image_name
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        img_bytes = compress_image_bytes(f.read())
                except Exception as e:
                    logger.error(f"Error reading reference image {image_name}: {e}")

        cases.append({
            "id": oid,
            "image_name": image_name,
            "text": text,
            "img_bytes": img_bytes,
            "has_image": img_bytes is not None,
            "sources": meta.get("sources", []),
            "ce_score": meta.get("ce_score"),
            "rrf_rank": meta.get("rrf_rank"),
        })
    return cases


def _load_patient_image(query_image):
    if not query_image:
        return None, False
    try:
        query_image.seek(0)
        img_bytes = compress_image_bytes(query_image.read())
        query_image.seek(0)
        return img_bytes, True
    except Exception as e:
        logger.error(f"Error loading patient image: {e}")
        return None, False


def _build_candidate_cases(reference_cases):
    candidates = []
    for idx, case in enumerate(reference_cases):
        entry = {
            "case_label": f"Reference Case {idx + 1}",
            "report": case["text"],
            "retrieval_sources": case["sources"],
            "rrf_rank": case["rrf_rank"],
        }
        if case["ce_score"] is not None:
            entry["cross_encoder_score"] = case["ce_score"]
        candidates.append(entry)
    return candidates


def _build_prompt_text(query_text, candidate_cases):
    return f"""
<current_patient>
<query>
{query_text}
</query>
</current_patient>

<retrieved_reference_cases>
{json.dumps(candidate_cases, indent=2)}
</retrieved_reference_cases>
"""


def _extract_matched_image_b64(response_text, reference_cases, json_data, root_path):
    try:
        for line in response_text.splitlines():
            if "Best Matching Case" in line:
                m = re.search(r"(\d+)", line)
                if m:
                    n = int(m.group(1))
                    if 1 <= n <= len(reference_cases):
                        return get_image_b64(reference_cases[n - 1]["image_name"], root_path)
    except Exception as e:
        logger.error(f"Error parsing matched case: {e}")

    if reference_cases:
        return get_image_b64(reference_cases[0]["image_name"], root_path)
    return None


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------

def _make_stream_fn(conversation_id, query_text, reference_cases, all_images, json_data, root_path):
    def stream_responses():
        yield json.dumps({"status": "init", "conversation_id": conversation_id}) + "\n"

        candidate_cases = _build_candidate_cases(reference_cases)
        prompt_text = _build_prompt_text(query_text, candidate_cases)
        ollama_model = os.getenv("OLLAMA_MODEL", "qwen2")

        try:
            response_text = generate_answer(
                model_name=ollama_model,
                system_prompt=SYSTEM_PROMPT,
                prompt_text=prompt_text,
                images=all_images,
            )
        except Exception as e:
            logger.error(f"LLM inference error: {e}")
            response_text = f"An error occurred: {str(e)}"

        matched_image_b64 = _extract_matched_image_b64(response_text, reference_cases, json_data, root_path)

        db_client.add_message(conversation_id, {
            "role": "assistant",
            "content": response_text,
            "similarityImage": matched_image_b64,
            "timestamp": datetime.utcnow().isoformat(),
            "id": str(ObjectId()),
        })

        yield json.dumps({
            "status": "message",
            "response": {"text": response_text, "image": matched_image_b64},
        }) + "\n"

    return stream_responses


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

@api_view(["POST"])
@permission_classes([AllowAny])
def GetResponse(request):
    text_model, image_model, image_processor = get_medical_models()
    index = Index(url=os.getenv("UPSTASH_DB_URL"), token=os.getenv("UPSTASH_READ_ONLY_TOKEN"))

    query_text = request.data.get("query")
    query_image = request.FILES.get("image")
    conversation_id = request.data.get("conversation_id")

    if not conversation_id or conversation_id in ("null", "undefined"):
        conversation_id = db_client.create_conversation(
            title=query_text[:50] if query_text else "New Image Chat"
        )

    _save_user_message(conversation_id, query_text, query_image)

    text_vectors, image_vectors, bm25_results = _run_retrieval(
        query_text, query_image, index, text_model, image_model, image_processor
    )
    top_cases = _select_top_cases(query_text, text_vectors, image_vectors, bm25_results, JSON_DATA)

    patient_img_bytes, has_patient_image = _load_patient_image(query_image)
    reference_cases = _load_reference_cases(top_cases, JSON_DATA, ROOT_PATH)

    all_images = []
    if has_patient_image:
        all_images.append(patient_img_bytes)
    all_images.extend(c["img_bytes"] for c in reference_cases if c["img_bytes"] is not None)

    stream_fn = _make_stream_fn(
        conversation_id, query_text, reference_cases, all_images, JSON_DATA, ROOT_PATH
    )
    return StreamingHttpResponse(stream_fn(), content_type="application/json")


def _save_user_message(conversation_id, query_text, query_image):
    message = {
        "role": "user",
        "content": query_text,
        "timestamp": datetime.utcnow().isoformat(),
        "id": str(ObjectId()),
    }
    if query_image:
        try:
            query_image.seek(0)
            content = query_image.read()
            content_type = getattr(query_image, "content_type", "image/png")
            message["image"] = f"data:{content_type};base64,{base64.b64encode(content).decode('utf-8')}"
            message["has_image"] = True
            query_image.seek(0)
        except Exception as e:
            logger.error(f"Error encoding user image: {e}")
            message["has_image"] = False
    db_client.add_message(conversation_id, message)


@api_view(["GET"])
@permission_classes([AllowAny])
def ListConversations(request):
    return Response(db_client.get_conversations())


@api_view(["GET"])
@permission_classes([AllowAny])
def GetConversation(request, conversation_id):
    conversation = db_client.get_conversation(conversation_id)
    if conversation:
        return Response(conversation)
    return Response({"error": "Conversation not found"}, status=404)


@api_view(["DELETE"])
@permission_classes([AllowAny])
def DeleteConversation(request, conversation_id):
    success = db_client.delete_conversation(conversation_id)
    if success:
        return Response({"status": "ok"})
    return Response({"error": "Failed to delete"}, status=400)


@api_view(["POST"])
@permission_classes([AllowAny])
def CreateConversation(request):
    title = request.data.get("title", "New Chat")
    conversation_id = db_client.create_conversation(title=title)
    return Response({"conversation_id": conversation_id})
