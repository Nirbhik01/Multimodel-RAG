from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from upstash_vector import Index

from core.embedding.generate_embeddings import get_text_embedding, get_image_embedding
from core.embedding.clip_model import get_clip

from core.retrieval.query_vector_db import query_vector_db

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

    model, processor = get_clip()

    UPSTASH_VECTOR_REST_URL = os.getenv('UPSTASH_DB_URL')
    UPSTASH_VECTOR_READ_ONLY_REST_TOKEN = os.getenv('UPSTASH_READ_ONLY_TOKEN')

    index = Index(url = UPSTASH_VECTOR_REST_URL, token = UPSTASH_VECTOR_READ_ONLY_REST_TOKEN)

    root_path = Path(__file__).parent.parent.parent
    json_data_path = root_path / 'data' / 'processed' / 'impression_and_findings.json'
    with open(json_data_path, 'r') as file:
        json_data = json.load(file)

    # Reading from multipart/form-data POST request
    query = request.data.get('query')
    image = request.FILES.get('image')
    
    text_embedding = get_text_embedding(text = query , model = model , processor = processor)
    image_embedding = get_image_embedding(image = image , model = model , processor = processor)

    final_embedding = (0.6 * image_embedding) + (0.4 * text_embedding)
    
    result_id = int(query_vector_db(query_vector = final_embedding, index = index)[0].id)

    logger.info(f'retrieved result id is : {result_id}') 

    image_name, text = get_image_name_and_text(id = result_id, json_data = json_data)
    image = get_image(image_name = image_name, root_path = root_path)

    return Response({
        "status": "ok",
        "text": text,
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
        

