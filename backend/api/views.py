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

def compress_image_bytes(image_bytes, max_size=(512, 512)):
    """Compresses image to a max resolution to drastically reduce LLM inference time."""
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img.thumbnail(max_size, Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=85)
        return buf.getvalue()
    except Exception as e:
        logger.error(f"Error compressing image: {e}")
        return image_bytes

@api_view(['POST'])
@permission_classes([AllowAny])
def GetResponse(request):

    text_model, image_model, image_processor = get_medical_models()

    UPSTASH_VECTOR_REST_URL = os.getenv('UPSTASH_DB_URL')
    UPSTASH_VECTOR_READ_ONLY_REST_TOKEN = os.getenv('UPSTASH_READ_ONLY_TOKEN')

    index = Index(url = UPSTASH_VECTOR_REST_URL, token = UPSTASH_VECTOR_READ_ONLY_REST_TOKEN)

    root_path = Path(__file__).parent.parent.parent
    json_data_path = root_path / 'data' / 'processed' / 'embedding_input.json'
    
    try:
        with open(json_data_path, 'r') as file:
            json_data = json.load(file)
    except Exception as e:
        logger.error(f"Failed to load json data: {e}")
        json_data = []

    # Reading from multipart/form-data POST request
    query_text = request.data.get('query')
    query_image = request.FILES.get('image')
    conversation_id = request.data.get('conversation_id')
    
    # If no conversation_id, create one
    if not conversation_id or conversation_id == 'null' or conversation_id == 'undefined':
        conversation_id = db_client.create_conversation(title=query_text[:50] if query_text else "New Image Chat")
    
    # Save user message
    user_message = {
        "role": "user",
        "content": query_text,
        "timestamp": datetime.utcnow().isoformat(),
        "id": str(ObjectId())
    }
    
    # Handle user image if present
    if query_image:
        try:
            # Read image content and convert to base64 for storage
            query_image.seek(0)
            image_content = query_image.read()
            encoded_string = base64.b64encode(image_content).decode('utf-8')
            # Determine content type (defaulting to image/png if not available)
            content_type = getattr(query_image, 'content_type', 'image/png')
            user_message["image"] = f"data:{content_type};base64,{encoded_string}"
            user_message["has_image"] = True
            
            # Reset file pointer for subsequent use in get_image_embedding
            query_image.seek(0)
        except Exception as e:
            logger.error(f"Error processing user image: {e}")
            user_message["has_image"] = False

    db_client.add_message(conversation_id, user_message)

    # Initialize vectors
    text_vectors = []
    image_vectors = []

    logger.info(f"[GetResponse] Entered. Query text length: {len(query_text) if query_text else 0}. Query image provided: {query_image is not None}")

    # Get embeddings and query vector DB if inputs are present
    if query_text and query_text.strip():
        logger.info("[GetResponse] Generating text embedding...")
        text_embedding = get_text_embedding(text = query_text, model = text_model)
        logger.info("[GetResponse] Querying vector DB for text modality...")
        text_vectors = query_vector_db(query_vector = text_embedding, index = index, vector_type = 'text')
        logger.info(f"[GetResponse] Text vectors retrieved: {len(text_vectors)} results.")
    
    if query_image:
        logger.info("[GetResponse] Generating image embedding...")
        image_embedding = get_image_embedding(image = query_image, processor = image_processor, model = image_model)
        logger.info("[GetResponse] Querying vector DB for image modality...")
        image_vectors = query_vector_db(query_vector = image_embedding, index = index, vector_type = 'image')
        logger.info(f"[GetResponse] Image vectors retrieved: {len(image_vectors)} results.")

    # Find overlapping and distinct IDs using the rank_results function
    logger.info("[GetResponse] Ranking and combining multimodal search results...")
    combined_scores = rank_results(text_vectors, image_vectors)
    logger.info(f"[GetResponse] Combined score matches (intersection if both are active): {list(combined_scores.keys())}")

    # 1. Joint overlapping case (best score in combined_scores)
    joint_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
    top_joint_id = joint_results[0][0] if joint_results else None
    logger.info(f"[GetResponse] Selected top joint ID: {top_joint_id}")

    # 2. Best text-only match (highest score in text_vectors that is NOT present in combined_scores)
    top_text_only_id = None
    sorted_text_vectors = sorted(text_vectors, key=lambda r: r.score if hasattr(r, 'score') else 0.0, reverse=True)
    for r in sorted_text_vectors:
        if r.metadata and "original_id" in r.metadata:
            oid = r.metadata["original_id"]
            if oid not in combined_scores:
                top_text_only_id = oid
                break
    logger.info(f"[GetResponse] Selected top text-only ID: {top_text_only_id}")

    # 3. Best image-only match (highest score in image_vectors that is NOT present in combined_scores)
    top_image_only_id = None
    sorted_image_vectors = sorted(image_vectors, key=lambda r: r.score if hasattr(r, 'score') else 0.0, reverse=True)
    for r in sorted_image_vectors:
        if r.metadata and "original_id" in r.metadata:
            oid = r.metadata["original_id"]
            if oid not in combined_scores:
                top_image_only_id = oid
                break
    logger.info(f"[GetResponse] Selected top image-only ID: {top_image_only_id}")

    # Gather selected reference cases
    selected_cases = []
    if top_joint_id:
        selected_cases.append(("joint", top_joint_id))
    if top_text_only_id:
        selected_cases.append(("text_only", top_text_only_id))
    if top_image_only_id:
        selected_cases.append(("image_only", top_image_only_id))

    # Fallback if both set calculations left selected_cases empty (e.g. single-modal searches)
    if not selected_cases:
        logger.info("[GetResponse] selected_cases is empty. Attempting single-modality fallbacks...")
        if text_vectors:
            sorted_text = sorted(text_vectors, key=lambda r: r.score if hasattr(r, 'score') else 0.0, reverse=True)
            for r in sorted_text:
                if r.metadata and "original_id" in r.metadata:
                    selected_cases.append(("text_fallback", r.metadata["original_id"]))
                    break
        elif image_vectors:
            sorted_image = sorted(image_vectors, key=lambda r: r.score if hasattr(r, 'score') else 0.0, reverse=True)
            for r in sorted_image:
                if r.metadata and "original_id" in r.metadata:
                    selected_cases.append(("image_fallback", r.metadata["original_id"]))
                    break

    logger.info(f"[GetResponse] Selected cases to present to LLM: {selected_cases}")

    if not selected_cases:
        logger.warning("[GetResponse] No selected cases could be resolved.")
        return Response({
            "status": "error",
            "text": "No relevant results found for your query.",
            "conversation_id": conversation_id
        }, status=404)

    image_bytes_list = []
    image = None

    # 1. Load patient's image if present
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
            logger.error(f"[GetResponse] Error reading patient's query image bytes: {e}")

    # 2. Load selected reference cases
    reference_metadata = []
    for case_type, oid in selected_cases:
        image_name, text = get_image_name_and_text(id = oid, json_data = json_data)
        logger.info(f"[GetResponse] Reference case details: ID={oid}, Image={image_name}")
        
        if image is None:
            image = get_image(image_name = image_name, root_path = root_path)
            
        has_img_in_case = False
        if image_name:
            data_path = root_path / 'data' / 'images' / 'images_normalized' / image_name
            if data_path.exists():
                try:
                    logger.info(f"[GetResponse] Reading and compressing reference image {image_name}...")
                    with open(data_path, 'rb') as img_f:
                        compressed_bytes = compress_image_bytes(img_f.read())
                        image_bytes_list.append(compressed_bytes)
                        has_img_in_case = True
                        logger.info(f"[GetResponse] Reference image {image_name} successfully compressed.")
                except Exception as e:
                    logger.error(f"[GetResponse] Error reading reference image {image_name}: {e}")
            else:
                logger.warn(f"[GetResponse] Reference image path does not exist: {data_path}")
                    
        reference_metadata.append({
            "type": case_type,
            "id": oid,
            "text": text,
            "has_image": has_img_in_case
        })

    from django.http import StreamingHttpResponse

    # Construct the streaming generator
    def stream_responses():
        logger.info("[stream_responses] Initiating response stream back to client...")
        # Yield the initialization status and conversation ID
        yield json.dumps({
            "status": "init",
            "conversation_id": conversation_id
        }) + "\n"

        for idx, case in enumerate(reference_metadata, 1):
            case_type = case["type"]
            case_id = case["id"]
            case_text = case["text"]
            
            logger.info(f"[stream_responses] Processing Case {idx}/{len(reference_metadata)}: type={case_type}, id={case_id}")
            
            # Determine subset of images for this specific case (Patient image #1 + Case image #2 if present)
            case_images = []
            if has_patient_image:
                case_images.append(image_bytes_list[0])
            
            # Retrieve the case-specific image from the global list
            if case["has_image"]:
                meta_index = reference_metadata.index(case)
                img_index = (1 + meta_index) if has_patient_image else meta_index
                if img_index < len(image_bytes_list):
                    case_images.append(image_bytes_list[img_index])
                    logger.info(f"[stream_responses] Case has valid reference image. Attaching case image at index {img_index}.")
                    
            # Build comparative prompt for this single case
            single_case_prompt = f"""
                <current_patient>
                <query>{query_text}</query>
                </current_patient>

                <reference_case type="{case_type}" id="{case_id}">
                <report_text>{case_text}</report_text>
                </reference_case>
            """

            if case_type == "joint":
                title = f"🔬 COMPARISON 1: Joint Clinical & Visual Match "
                case_desc = "Evaluate the Patient's data against this highly correlated case that matches both visually and textually."
            elif case_type == "text_only":
                title = f"✍️ COMPARISON 2: Best Clinical Match "
                case_desc = "Evaluate the Patient's clinical symptoms against this case which has the closest textual diagnosis."
            elif case_type == "image_only":
                title = f"📸 COMPARISON 3: Best Visual Match "
                case_desc = "Evaluate the Patient's visual scans against this case which has the closest morphological visual pattern."
            else:
                title = f"🔍 COMPARISON: Reference Case Match "
                case_desc = "Evaluate the Patient's data against this similar clinical match."

            system_prompt = f"""
            You are an expert Radiologist AI Assistant. You specialize in analyzing chest X-rays and interpreting clinical findings.
            
            Your task is to generate:
            {title}
            
            Context detail: {case_desc}
            
            Generate a professional comparative report in the following exact format:
            ---
            ### {title}
            - **Visual Comparison**: Compare the patient's X-ray (Image #1) with the reference X-ray (Image #2 if provided). Highlight identical features or subtle differences.
            - **Clinical Interpretation**: Compare the patient's symptoms/observations with the reference report.
            - **Draft Findings & Impression**: Outline the Findings and Impression for the patient based on this comparison.
            - **Similarities & Differences**: List the similarities and differences between the patient's case and the reference case.
            ---

            (Note: dont assign number to images just say user uploaded and reference image in your response)
            """

            ollama_model = os.getenv('OLLAMA_MODEL', 'qwen2')
            try:
                logger.info(f"[stream_responses] Invoking local Ollama model '{ollama_model}' for Case ID {case_id}. Image count: {len(case_images)}.")
                response_text = generate_answer(
                    model_name = ollama_model,
                    system_prompt = system_prompt,
                    prompt_text = single_case_prompt,
                    images = case_images
                )
                logger.info(f"[stream_responses] Successfully generated clinical response from Ollama for Case ID {case_id}. Length: {len(response_text)} chars.")
            except Exception as e:
                logger.error(f"[stream_responses] Error calling generate_answer for case {case_id}: {e}")
                response_text = f"An error occurred while analyzing reference case {case_id}: {str(e)}"
            
            # Load the base64 reference image for frontend display mapping
            ref_image_name, _ = get_image_name_and_text(id = case_id, json_data = json_data)
            ref_image_b64 = get_image(image_name = ref_image_name, root_path = root_path)

            # Save assistant message individually in DB
            assistant_message = {
                "role": "assistant",
                "content": response_text,
                "similarityImage": ref_image_b64,
                "timestamp": datetime.utcnow().isoformat(),
                "id": str(ObjectId()) 
            }
            logger.info(f"[stream_responses] Saving assistant message for Case ID {case_id} to conversation {conversation_id} in MongoDB...")
            db_client.add_message(conversation_id, assistant_message)
            
            # Yield single response back to the client immediately
            logger.info(f"[stream_responses] Yielding parsed JSON chunk back to client stream for Case ID {case_id}...")
            yield json.dumps({
                "status": "message",
                "response": {
                    "text": response_text,
                    "image": ref_image_b64
                }
            }) + "\n"

        logger.info("[stream_responses] Response stream generation complete.")

    return StreamingHttpResponse(stream_responses(), content_type="application/json")

@api_view(['GET'])
@permission_classes([AllowAny])
def ListConversations(request):
    conversations = db_client.get_conversations()
    return Response(conversations)

@api_view(['GET'])
@permission_classes([AllowAny])
def GetConversation(request, conversation_id):
    conversation = db_client.get_conversation(conversation_id)
    if conversation:
        return Response(conversation)
    return Response({"error": "Conversation not found"}, status=404)

@api_view(['DELETE'])
@permission_classes([AllowAny])
def DeleteConversation(request, conversation_id):
    success = db_client.delete_conversation(conversation_id)
    if success:
        return Response({"status": "ok"})
    return Response({"error": "Failed to delete"}, status=400)

@api_view(['POST'])
@permission_classes([AllowAny])
def CreateConversation(request):
    title = request.data.get('title', 'New Chat')
    conversation_id = db_client.create_conversation(title=title)
    return Response({"conversation_id": conversation_id})

def get_image(image_name, root_path):
    if not image_name:
        logger.error("No image name provided to get_image")
        return None
    
    data_path = root_path / 'data' / 'images' / 'images_normalized' / image_name

    if not data_path.exists():
        logger.error(f"Image not found at path: {data_path}")
        return None

    with open (data_path, 'rb') as img:
        encoded_string = base64.b64encode(img.read()).decode('utf-8')
        return f"data:image/png;base64,{encoded_string}"

def get_image_name_and_text(id, json_data):
    for item in json_data:
        if item['id'] == id:
            return item['image'], item['text']
    return None, None

def rank_results(text_vectors, image_vectors):
    # Store text scores/metadata in a temporary dictionary
    text_map = {}
    for r in text_vectors:
        if r.metadata and "original_id" in r.metadata:
            oid = r.metadata["original_id"]
            text_map[oid] = r.score

    # Store image scores/metadata in a temporary dictionary
    image_map = {}
    for r in image_vectors:
        if r.metadata and "original_id" in r.metadata:
            oid = r.metadata["original_id"]
            image_map[oid] = r.score

    combined_scores = {}

    # If both modalities are active, strictly keep only the intersection (overlapping original_ids)
    if text_map and image_map:
        intersection_ids = set(text_map.keys()) & set(image_map.keys())
        for oid in intersection_ids:
            combined_scores[oid] = (0.6 * text_map[oid]) + (0.4 * image_map[oid])
            
    # Fallback if only text query was executed
    elif text_map:
        for oid, score in text_map.items():
            combined_scores[oid] = score
            
    # Fallback if only image query was executed
    elif image_map:
        for oid, score in image_map.items():
            combined_scores[oid] = score

    return combined_scores