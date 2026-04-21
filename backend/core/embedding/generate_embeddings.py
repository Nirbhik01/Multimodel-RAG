import torch
from PIL import Image

def get_text_embedding(text, processor, model):
    # Added truncation and max_length to handle long text inputs
    inputs = processor(text=[text], return_tensors="pt", padding=True, truncation=True, max_length=77)

    with torch.no_grad():
        outputs = model.get_text_features(**inputs)
        # Extract the tensor from the output object
        emb = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs

    # Normalize the embedding
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().cpu()

def get_image_embedding(image, processor, model):
    image = Image.open(image).convert("RGB")

    inputs = processor(images=image, return_tensors="pt")

    with torch.no_grad():
        outputs = model.get_image_features(**inputs)
        # Extract the tensor from the output object
        emb = outputs.pooler_output if hasattr(outputs, 'pooler_output') else outputs

    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.squeeze().cpu()