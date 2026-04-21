from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

from core.embedding.generate_embeddings import get_text_embedding, get_image_embedding

import logging
logger = logging.getLogger(__name__)

from core.embedding.clip_model import get_clip

# Create your views here.

@api_view(['POST'])
@permission_classes([AllowAny])
def GetResponse(request):
    model, processor = get_clip()

    # Reading from multipart/form-data POST request
    query = request.data.get('query')
    image = request.FILES.get('image')
    
    text_embedding = get_text_embedding(text = query , model = model , processor = processor)
    image_embedding = get_image_embedding(image = image , model = model , processor = processor)
    
    return Response({
        "status": "ok",
        "text": f"Processed query: {query}",
        "image": 'image1.png'
    })


