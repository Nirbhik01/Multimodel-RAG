from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from upstash_vector import Index

from core.embedding.generate_embeddings import get_text_embedding, get_image_embedding
from core.embedding.medical_models import get_medical_models

from core.retrieval.query_vector_db import query_vector_db

from core.generation.generate_answer import generate_answer

from datetime import datetime
import os
from dotenv import load_dotenv

from pathlib import Path

import json
import base64
import io
from PIL import Image

import logging
from .mongodb_utils import db_client
from bson import ObjectId

logger = logging.getLogger(__name__)
load_dotenv()

# Create your views here.
JSON_DATA = None
ROOT_PATH = Path(__file__).parent.parent.parent
try:
    json_data_path = ROOT_PATH / "data" / "processed" / "embedding_input.json"
    with open(json_data_path, "r") as file:
        JSON_DATA = json.load(file)
except Exception as e:
    logger.error(f"Failed to load json data: {e}")
    JSON_DATA = []


def compress_image_bytes(image_bytes, max_size=(512, 512)):
    """Compresses image to a max resolution to drastically reduce LLM inference time."""
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


@api_view(["POST"])
@permission_classes([AllowAny])
def GetResponse(request):

    text_model, image_model, image_processor = get_medical_models()

    UPSTASH_VECTOR_REST_URL = os.getenv("UPSTASH_DB_URL")
    UPSTASH_VECTOR_READ_ONLY_REST_TOKEN = os.getenv("UPSTASH_READ_ONLY_TOKEN")

    index = Index(
        url=UPSTASH_VECTOR_REST_URL, token=UPSTASH_VECTOR_READ_ONLY_REST_TOKEN
    )

    root_path = ROOT_PATH
    json_data = JSON_DATA

    query_text = request.data.get("query")
    query_image = request.FILES.get("image")
    conversation_id = request.data.get("conversation_id")

    if (
        not conversation_id
        or conversation_id == "null"
        or conversation_id == "undefined"
    ):
        conversation_id = db_client.create_conversation(
            title=query_text[:50] if query_text else "New Image Chat"
        )

    user_message = {
        "role": "user",
        "content": query_text,
        "timestamp": datetime.utcnow().isoformat(),
        "id": str(ObjectId()),
    }

    if query_image:
        try:
            # Read image content and convert to base64 for storage
            query_image.seek(0)
            image_content = query_image.read()
            encoded_string = base64.b64encode(image_content).decode("utf-8")

            content_type = getattr(query_image, "content_type", "image/png")
            user_message["image"] = f"data:{content_type};base64,{encoded_string}"
            user_message["has_image"] = True

            # Reset file pointer for subsequent use in get_image_embedding
            query_image.seek(0)
        except Exception as e:
            logger.error(f"Error processing user image: {e}")
            user_message["has_image"] = False

    db_client.add_message(conversation_id, user_message)

    text_vectors = []
    image_vectors = []

    logger.info(
        f"[GetResponse] Entered. Query text length: {len(query_text) if query_text else 0}. Query image provided: {query_image is not None}"
    )

    if query_text and query_text.strip():
        logger.info("[GetResponse] Generating text embedding...")
        text_embedding = get_text_embedding(text=query_text, model=text_model)
        logger.info("[GetResponse] Querying vector DB for text modality...")
        text_vectors = query_vector_db(
            query_vector=text_embedding, index=index, vector_type="text"
        )
        logger.info(
            f"[GetResponse] Text vectors retrieved: {len(text_vectors)} results."
        )

    if query_image:
        logger.info("[GetResponse] Generating image embedding...")
        image_embedding = get_image_embedding(
            image=query_image, processor=image_processor, model=image_model
        )
        logger.info("[GetResponse] Querying vector DB for image modality...")
        image_vectors = query_vector_db(
            query_vector=image_embedding, index=index, vector_type="image"
        )
        logger.info(
            f"[GetResponse] Image vectors retrieved: {len(image_vectors)} results."
        )

    # Find overlapping and distinct IDs using the rank_results function
    logger.info("[GetResponse] Ranking and combining multimodal search results...")
    combined_scores = rank_results(text_vectors, image_vectors)
    logger.info(
        f"[GetResponse] Combined score matches (intersection if both are active): {list(combined_scores.keys())}"
    )

    joint_results = sorted(
        combined_scores.items(),
        key=lambda x: x[1],
        reverse=True,
    )

    top_cases = joint_results[:5]

    selected_cases = [
        ("retrieved_case", oid)
        for oid, _ in top_cases
    ]

    image_bytes_list = []
    image = None

    has_patient_image = False
    if query_image:
        try:
            logger.info("[GetResponse] Compressing patient query image...")
            query_image.seek(0)
            patient_img_bytes = compress_image_bytes(query_image.read())
            image_bytes_list.append(patient_img_bytes)
            query_image.seek(0)
            has_patient_image = True
            logger.info("[GetResponse] Patient query image successfully loaded.")
        except Exception as e:
            logger.error(
                f"[GetResponse] Error reading patient's query image bytes: {e}"
            )

    reference_metadata = []
    for case_type, oid in selected_cases:
        image_name, text = get_image_name_and_text(id=oid, json_data=json_data)
        logger.info(
            f"[GetResponse] Reference case details: ID={oid}, Image={image_name}"
        )

        if image is None:
            image = get_image(image_name=image_name, root_path=root_path)

        has_img_in_case = False
        if image_name:
            data_path = root_path / "data" / "images" / "images_normalized" / image_name
            if data_path.exists():
                try:
                    logger.info(
                        f"[GetResponse] Reading and compressing reference image {image_name}..."
                    )
                    with open(data_path, "rb") as img_f:
                        compressed_bytes = compress_image_bytes(img_f.read())
                        image_bytes_list.append(compressed_bytes)
                        has_img_in_case = True
                        logger.info(
                            f"[GetResponse] Reference image {image_name} successfully compressed."
                        )
                except Exception as e:
                    logger.error(
                        f"[GetResponse] Error reading reference image {image_name}: {e}"
                    )
            else:
                logger.warn(
                    f"[GetResponse] Reference image path does not exist: {data_path}"
                )

        reference_metadata.append(
            {"type": case_type, "id": oid, "text": text, "has_image": has_img_in_case}
        )

    from django.http import StreamingHttpResponse

    def stream_responses():
        logger.info("[stream_responses] Initiating SINGLE multimodal response stream...")

        yield json.dumps(
            {
                "status": "init",
                "conversation_id": conversation_id,
            }
        ) + "\n"

        candidate_cases = []

        for idx, case in enumerate(reference_metadata):

            case_id = case["id"]
            case_type = case["type"]
            case_text = case["text"]

            ref_image_name, _ = get_image_name_and_text(
                id=case_id,
                json_data=json_data,
            )

            ref_image_b64 = get_image(
                image_name=ref_image_name,
                root_path=root_path,
            )

            candidate_cases.append(
                {
                    "id": case_id,
                    "type": case_type,
                    "report": case_text,
                }
            )

        prompt_text = f"""
        <current_patient>
        <query>
        {query_text}
        </query>
        </current_patient>

        <retrieved_reference_cases>

        {json.dumps(candidate_cases, indent=2)}

        </retrieved_reference_cases>
        """

        system_prompt = """
        You are an expert Radiologist AI Assistant.

        You are given:
        1. A patient's X-ray and clinical text
        2. Multiple retrieved reference cases

        Your tasks:

        - Compare all retrieved cases
        - Determine which reference case best matches the patient
        - Explain why
        - Mention conflicting evidence if present
        - Produce a final radiology Findings section
        - Produce a final Impression section
        - Assign a confidence level
        - Focus just a tad bit more on the conditions that the patient's facing in reference as well as user's case while comparing rather than other things i.e. say there are 5 aspects to compare and condition is one of them, ideally you'd focus 20% on each aspect but on condition i want you to focus about 21-22%.
        - Dont mention any sensitive relation about the cases like id's, names, patient info etc.

        Return output in this exact format:

        ---
        ### Retrieval Analysis
        - Best Matching Case:
        - Why It Matches:
        - Conflicting Evidence:
        - Confidence Level:

        ### Final Findings
        ...

        ### Final Impression
        ...
        ---
        """

        all_images = []

        if has_patient_image:
            all_images.append(image_bytes_list[0])

        for idx, case in enumerate(reference_metadata):

            if case["has_image"]:

                img_index = (
                    idx + 1
                    if has_patient_image
                    else idx
                )

                if img_index < len(image_bytes_list):
                    all_images.append(
                        image_bytes_list[img_index]
                    )

        ollama_model = os.getenv("OLLAMA_MODEL", "qwen2")

        try:

            logger.info(
                f"[stream_responses] Running SINGLE LLM inference with {len(candidate_cases)} retrieved cases."
            )

            response_text = generate_answer(
                model_name=ollama_model,
                system_prompt=system_prompt,
                prompt_text=prompt_text,
                images=all_images,
            )

        except Exception as e:

            logger.error(
                f"[stream_responses] Error during single LLM inference: {e}"
            )

            response_text = f"An error occurred: {str(e)}"

        assistant_message = {
            "role": "assistant",
            "content": response_text,
            "timestamp": datetime.utcnow().isoformat(),
            "id": str(ObjectId()),
        }

        db_client.add_message(
            conversation_id,
            assistant_message,
        )

        yield json.dumps(
            {
                "status": "message",
                "response": {
                    "text": response_text,
                },
            }
        ) + "\n"

        logger.info("[stream_responses] Single response completed.")
    return StreamingHttpResponse(stream_responses(), content_type="application/json")


@api_view(["GET"])
@permission_classes([AllowAny])
def ListConversations(request):
    conversations = db_client.get_conversations()
    return Response(conversations)


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


def get_image(image_name, root_path):
    if not image_name:
        logger.error("No image name provided to get_image")
        return None

    data_path = root_path / "data" / "images" / "images_normalized" / image_name

    if not data_path.exists():
        logger.error(f"Image not found at path: {data_path}")
        return None

    with open(data_path, "rb") as img:
        encoded_string = base64.b64encode(img.read()).decode("utf-8")
        return f"data:image/png;base64,{encoded_string}"


def get_image_name_and_text(id, json_data):
    for item in json_data:
        if item["id"] == id:
            return item["image"], item["text"]
    return None, None


def rank_results(text_vectors, image_vectors, k=60):

    # Sort text vectors in descending order of score to assign ranks
    sorted_text_vectors = sorted(
        text_vectors,
        key=lambda r: r.score if hasattr(r, "score") else 0.0,
        reverse=True,
    )
    text_ranks = {}
    for idx, r in enumerate(sorted_text_vectors):
        if r.metadata and "original_id" in r.metadata:
            oid = r.metadata["original_id"]
            if oid not in text_ranks:
                text_ranks[oid] = idx + 1  # 1-indexed rank

    # Sort image vectors in descending order of score to assign ranks
    sorted_image_vectors = sorted(
        image_vectors,
        key=lambda r: r.score if hasattr(r, "score") else 0.0,
        reverse=True,
    )
    image_ranks = {}
    for idx, r in enumerate(sorted_image_vectors):
        if r.metadata and "original_id" in r.metadata:
            oid = r.metadata["original_id"]
            if oid not in image_ranks:
                image_ranks[oid] = idx + 1  # 1-indexed rank

    combined_scores = {}

    if text_ranks and image_ranks:
        all_ids = set(text_ranks.keys()) | set(image_ranks.keys())

        for oid in all_ids:
            text_rrf = (
                1.0 / (k + text_ranks[oid])
                if oid in text_ranks
                else 0
            )

            image_rrf = (
                1.0 / (k + image_ranks[oid])
                if oid in image_ranks
                else 0
            )

            combined_scores[oid] = text_rrf + image_rrf

    elif text_ranks:
        for oid, rank in text_ranks.items():
            combined_scores[oid] = 1.0 / (k + rank)

    elif image_ranks:
        for oid, rank in image_ranks.items():
            combined_scores[oid] = 1.0 / (k + rank)

    return combined_scores
 