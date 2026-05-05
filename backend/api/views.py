from os import truncate
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from upstash_vector import Index

from core.embedding.generate_embeddings import get_text_embedding, get_image_embedding
from core.embedding.medical_models import get_medical_models

from core.retrieval.query_vector_db import query_vector_db

from core.generation.generate_answer import generate_answer

import os
from dotenv import load_dotenv

from pathlib import Path

import json
import base64

import logging

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
    with open(json_data_path, 'r') as file:
        json_data = json.load(file)

    # Reading from multipart/form-data POST request
    query_text = request.data.get('query')
    query_image = request.FILES.get('image')
    
    text_embedding = get_text_embedding(text = query_text, model = text_model)
    image_embedding = get_image_embedding(image = query_image, processor = image_processor, model = image_model)
    
    text_vectors = query_vector_db(query_vector = text_embedding, index = index)
    image_vectors = query_vector_db(query_vector = image_embedding, index = index)

    combined_scores = {}

    for r in text_vectors:
        oid = r.metadata["original_id"]
        combined_scores[oid] = combined_scores.get(oid, 0) + 0.6 * r.score

    for r in image_vectors:
        oid = r.metadata["original_id"]
        combined_scores[oid] = combined_scores.get(oid, 0) + 0.4 * r.score

    logger.info(f'scores : %.100s' % combined_scores)

    final_results = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)

    # logger.info(f'final result : {final_results}')

    image_name, text = get_image_name_and_text(id = final_results[0][0], json_data = json_data)
    image = get_image(image_name = image_name, root_path = root_path)

    content = f'''
        User input data:
            Query: {query_text}
        Similar data according to rag:
            Text: {text}
    '''

    final_response = generate_answer(gemini_model = 'gemini-2.5-flash' , GEMINI_API_KEY = os.getenv('GEMINI_API_KEY') , contents = content)

    return Response({
        "status": "ok",
        "text": final_response,
        "image": image
    })

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
    