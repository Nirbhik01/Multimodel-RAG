from os import truncate
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

import logging
from .mongodb_utils import db_client
from bson import ObjectId

logger = logging.getLogger(__name__)
load_dotenv()

# Create your views here.

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

    # Get embeddings and query vector DB if inputs are present
    if query_text and query_text.strip():
        text_embedding = get_text_embedding(text = query_text, model = text_model)
        text_vectors = query_vector_db(query_vector = text_embedding, index = index, vector_type = 'text')
    
    if query_image:
        image_embedding = get_image_embedding(image = query_image, processor = image_processor, model = image_model)
        image_vectors = query_vector_db(query_vector = image_embedding, index = index, vector_type = 'image')

    combined_scores = rank_results(text_vectors, image_vectors)

    logger.info(f'scores : {combined_scores}')

    final_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    if not final_results:
        return Response({
            "status": "error",
            "text": "No relevant results found for your query.",
            "conversation_id": conversation_id
        }, status=404)

    # Retrieve top 2 reference cases for two-case prompting
    top_results = final_results[:2]
    reference_cases = []
    image_bytes_list = []
    
    # Store the first reference case image for frontend display backwards compatibility
    image = None

    # Load patient's image if present
    if query_image:
        try:
            query_image.seek(0)
            patient_img_bytes = query_image.read()
            image_bytes_list.append(patient_img_bytes)
            query_image.seek(0)
        except Exception as e:
            logger.error(f"Error reading patient's query image bytes: {e}")

    for idx, (oid, score) in enumerate(top_results):
        image_name, text = get_image_name_and_text(id = oid, json_data = json_data)
        
        if idx == 0:
            image = get_image(image_name = image_name, root_path = root_path)
            
        if image_name:
            data_path = root_path / 'data' / 'images' / 'images_normalized' / image_name
            if data_path.exists():
                try:
                    with open(data_path, 'rb') as img_f:
                        image_bytes_list.append(img_f.read())
                except Exception as e:
                    logger.error(f"Error reading reference image {image_name}: {e}")
                    
        reference_cases.append({
            "id": oid,
            "text": text,
            "image_name": image_name
        })

    # Construct structured XML-like context prompt
    content_parts = []
    content_parts.append("<current_patient>")
    if query_text:
        content_parts.append(f"  <query>{query_text}</query>")
    content_parts.append("</current_patient>\n")
    
    content_parts.append("<similar_reference_cases>")
    for idx, case in enumerate(reference_cases, start=1):
        content_parts.append(f"  <case index='{idx}'>")
        content_parts.append(f"    <report_text>{case['text']}</report_text>")
        content_parts.append(f"  </case>")
    content_parts.append("</similar_reference_cases>")
    
    content = "\n".join(content_parts)

    system_prompt = """
    You are an expert Radiologist AI Assistant. You specialize in analyzing chest X-rays and interpreting clinical findings.
    
    You will be provided with:
    1. A Current Patient's X-ray image (if provided) and their preliminary query/observations.
    2. One or two Reference X-ray images and their corresponding Reports (Findings and Impression) from similar cases.
    
    Your Task:
    - Compare the current Patient's X-ray image with the Reference X-ray images.
    - Evaluate the patient's query against both the visual evidence and the reference cases.
    - Provide a professional, concise, and accurate Radiology Report for the Current Patient.
    - Structure your output clearly with 'Findings' and 'Impression' sections.
    - Highlight any significant similarities or differences between the current case and the reference cases that aided your analysis.
    """

    ollama_model = os.getenv('OLLAMA_MODEL', 'qwen2')
    final_response = generate_answer(
        model_name = ollama_model,
        system_prompt = system_prompt,
        prompt_text = content,
        images = image_bytes_list
    )

    # Save assistant message
    assistant_message = {
        "role": "assistant",
        "content": final_response,
        "similarityImage": image,
        "timestamp": datetime.utcnow().isoformat(),
        "id": str(ObjectId()) 
    }
    db_client.add_message(conversation_id, assistant_message)

    return Response({
        "status": "ok",
        "text": final_response,
        "image": image,
        "conversation_id": conversation_id
    })

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
    # Standard Reciprocal Rank Fusion (RRF)
    # Formula: Score(d) = sum_{m} ( 1 / (k + rank_m(d)) )
    # k = 60 as per standard RRF literature. Treats both modalities with equal importance.
    k = 60
    combined_scores = {}

    for rank, r in enumerate(text_vectors, start=1):
        if r.metadata and "original_id" in r.metadata:
            oid = r.metadata["original_id"]
            combined_scores[oid] = combined_scores.get(oid, 0.0) + 1.0 / (k + rank)

    for rank, r in enumerate(image_vectors, start=1):
        if r.metadata and "original_id" in r.metadata:
            oid = r.metadata["original_id"]
            combined_scores[oid] = combined_scores.get(oid, 0.0) + 1.0 / (k + rank)

    return combined_scores