import os
import json
import torch
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoImageProcessor

# Configuration
TEXT_MODEL_ID = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
IMAGE_MODEL_ID = "microsoft/rad-dino"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def generate_embeddings():
    root_path = Path(__file__).parent.parent
    data_path = root_path / 'data' / 'processed' / 'impression_and_findings.json'
    image_dir = root_path / 'data' / 'images' / 'images_normalized'
    output_path = root_path / 'data' / 'processed' / 'embedding_results.json'

    print(f"Loading models to {DEVICE}...")
    text_model = SentenceTransformer(TEXT_MODEL_ID).to(DEVICE)
    image_processor = AutoImageProcessor.from_pretrained(IMAGE_MODEL_ID)
    image_model = AutoModel.from_pretrained(IMAGE_MODEL_ID).to(DEVICE)
    image_model.eval()

    if not data_path.exists():
        print(f"Error: {data_path} not found.")
        return

    with open(data_path, 'r') as f:
        data = json.load(f)

    results = []
    print(f"Processing {len(data)} entries...")

    for entry in tqdm(data):
        entry_id = entry['id']
        text = entry['text']
        image_name = entry['image']
        image_path = image_dir / image_name

        try:
            # Text Embedding
            with torch.no_grad():
                text_emb = text_model.encode(text, convert_to_tensor=True)
                text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
            
            # Image Embedding
            if image_path.exists():
                img = Image.open(image_path).convert("RGB")
                inputs = image_processor(images=img, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    outputs = image_model(**inputs)
                    img_emb = outputs.last_hidden_state[:, 0, :] # [CLS] token
                    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                
                results.append({
                    "id": entry_id,
                    "text_embedding": text_emb.cpu().tolist(),
                    "image_embedding": img_emb.squeeze().cpu().tolist()
                })
            else:
                print(f"Warning: Image {image_name} not found for ID {entry_id}")

        except Exception as e:
            print(f"Error processing {entry_id}: {e}")

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Successfully saved {len(results)} embeddings to {output_path}")

if __name__ == "__main__":
    generate_embeddings()
