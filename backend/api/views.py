from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from upstash_vector import Index

from core.embedding.generate_embeddings import get_text_embedding, get_image_embedding
from core.embedding.clip_model import get_clip

from core.retrieval.query_vector_db import query_vector_db

import os
from dotenv import load_dotenv

import logging

logger = logging.getLogger(__name__)
load_dotenv()

# Create your views here.

@api_view(['POST'])
@permission_classes([AllowAny])
def GetResponse(request):

    model, processor = get_clip()

    UPSTASH_VECTOR_REST_URL = os.getenv('UPSTASH_DB_URL')
    UPSTASH_VECTOR_REST_TOKEN = os.getenv('UPSTASH_READ_ONLY_TOKEN')

    index = Index(url = UPSTASH_VECTOR_REST_URL, token = UPSTASH_VECTOR_REST_TOKEN)

    # Reading from multipart/form-data POST request
    query = request.data.get('query')
    image = request.FILES.get('image')
    
    text_embedding = get_text_embedding(text = query , model = model , processor = processor)
    image_embedding = get_image_embedding(image = image , model = model , processor = processor)

    final_embedding = (0.6 * image_embedding) + (0.4 * text_embedding)
    
    print(query_vector_db(query_vector = final_embedding, index = index))

    return Response({
        "status": "ok",
        "text": f"Processed query: {query}",
        "image": 'image1.png'
    })


