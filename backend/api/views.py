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
        text_vectors = query_vector_db(query_vector = text_embedding, index = index)
    
    if query_image:
        image_embedding = get_image_embedding(image = query_image, processor = image_processor, model = image_model)
        image_vectors = query_vector_db(query_vector = image_embedding, index = index)

    combined_scores = {}

    for r in text_vectors:
        if r.metadata and "original_id" in r.metadata:
            oid = r.metadata["original_id"]
            combined_scores[oid] = combined_scores.get(oid, 0) + 0.6 * r.score

    for r in image_vectors:
        if r.metadata and "original_id" in r.metadata:
            oid = r.metadata["original_id"]
            combined_scores[oid] = combined_scores.get(oid, 0) + 0.4 * r.score

    logger.info(f'scores : {str(combined_scores)[:100]}')

    final_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    if not final_results:
        return Response({
            "status": "error",
            "text": "No relevant results found for your query.",
            "conversation_id": conversation_id
        }, status=404)

    image_name, text = get_image_name_and_text(id = final_results[0][0], json_data = json_data)
    image = get_image(image_name = image_name, root_path = root_path)

    content = f'''
        User input data:
            Query: {query_text}
        Similar data according to rag:
            Text: {text}
    '''

    final_response = generate_answer(gemini_model = 'gemini-2.5-flash' , GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') , contents = content)

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
