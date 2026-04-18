from dotenv import load_dotenv
import json
from pathlib import Path
import os
from upstash_vector import Index, Vector
# from transformers import CLIPProcessor, CLIPModel
import os

load_dotenv()

index = Index(
    url = os.getenv('UPSTASH_DB_URL'),
    token = os.getenv('UPSTASH_TOKEN'),
)

# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

root_path = Path(__file__).parent.parent
data_path = root_path / 'chest-x-ray-data' / 'impression_and_findings.json'

with open(data_path, 'r') as f:
    data = json.load(f)

for entry in data:
    index.upsert(
        vectors = [
            Vector(
                id = entry['id'],
                data = entry['text'],
                metadata = {"image": entry['image']},
            )
        ]
    )
    break