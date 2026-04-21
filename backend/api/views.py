from django.shortcuts import render
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import AllowAny
from rest_framework.response import Response

import logging
logger = logging.getLogger(__name__)

from api.generate_embedding.clip_model import get_clip

# Create your views here.

@api_view(['POST'])
@permission_classes([AllowAny])
def GetResponse(request):
    logger.info(f"log - Received request at GetResponse")
    print("print - Received request at GetResponse")
    
    # Reading from multipart/form-data POST request
    query = request.data.get('query')
    image = request.FILES.get('image')
    
    print(f"Query: {query}")
    print(f"Image: {image}")
    
    model, processor = get_clip()
    
    return Response({
        "status": "ok",
        "text": f"Processed query: {query}",
        "image": 'image1.png'
    })


