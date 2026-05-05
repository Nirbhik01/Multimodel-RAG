import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoImageProcessor
from PIL import Image
import numpy as np

def test_dimensions():
    print("Testing PubMedBERT...")
    text_model = SentenceTransformer("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext")
    text = "Normal chest X-ray."
    text_emb = text_model.encode(text)
    print(f"Text embedding shape: {text_emb.shape}")
    assert text_emb.shape[0] == 768

    print("Testing Rad-DINO...")
    image_model_id = "microsoft/rad-dino"

    processor = AutoImageProcessor.from_pretrained(image_model_id)
    model = AutoModel.from_pretrained(image_model_id)
    
    # Create a dummy image
    dummy_image = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    inputs = processor(images=dummy_image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        img_emb = outputs.last_hidden_state[:, 0, :]
    
    print(f"Image embedding shape: {img_emb.shape}")
    assert img_emb.shape[1] == 768
    print("Verification successful: Both models produce 768-dimensional embeddings.")

if __name__ == "__main__":
    try:
        test_dimensions()
    except Exception as e:
        print(f"Verification failed: {e}")
