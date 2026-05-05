from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoImageProcessor
import torch

_text_model = None
_image_model = None
_image_processor = None

def get_medical_models():
    global _text_model, _image_model, _image_processor

    if _text_model is None:
        # PubMedBERT for text embeddings (768 dim)
        text_model_id = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
        _text_model = SentenceTransformer(text_model_id)

    if _image_model is None:
        # Rad-DINO for image embeddings (768 dim)
        image_model_id = "microsoft/rad-dino"
        _image_processor = AutoImageProcessor.from_pretrained(image_model_id)
        _image_model = AutoModel.from_pretrained(image_model_id)
        _image_model.eval()

    return _text_model, _image_model, _image_processor
