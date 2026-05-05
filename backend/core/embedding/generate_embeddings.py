import torch
from PIL import Image

def get_text_embedding(text, model):
    """
    Generates a 768-dim embedding using PubMedBERT (SentenceTransformer).
    """
    # SentenceTransformer handles truncation and normalization internally if configured, 
    # but we'll ensure it's a 768-dim vector.
    embedding = model.encode(text, convert_to_tensor=True)
    
    # Normalize if not already
    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    
    return embedding.cpu()

def get_image_embedding(image, processor, model):
    """
    Generates a 768-dim embedding using Rad-DINO.
    """
    image = Image.open(image).convert("RGB")
    
    # Preprocess image
    inputs = processor(images=image, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model(**inputs)
        # Rad-DINO usually uses the [CLS] token (first token of the last hidden state)
        # for global image representation.
        last_hidden_state = outputs.last_hidden_state
        # [CLS] token is at index 0
        emb = last_hidden_state[:, 0, :]
    
    # Normalize
    emb = emb / emb.norm(dim=-1, keepdim=True)
    
    return emb.squeeze().cpu()