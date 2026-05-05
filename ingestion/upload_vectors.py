from dotenv import load_dotenv
import json
from pathlib import Path
import os
from upstash_vector import Index, Vector
import os

load_dotenv()

root_path = Path(__file__).parent.parent

data_path = root_path / 'data' / 'processed' / 'embedding_results.json'

index = Index(
    url = os.getenv('UPSTASH_DB_URL'),
    token = os.getenv('UPSTASH_TOKEN'),
)

def insert_vectors(data_path, index, batch_size=100):

    success_count = 0
    skip_count = 0
    error_count = 0

    batch = []

    with open(data_path, 'r') as f:
        data = json.load(f)

    fields = ["id","text_embedding","cleaned_text","image_embedding","image", "label","cluster","normal"]

    for entry in data:
        try:
            if any(x not in entry for x in fields):
                skip_count += 1
                continue

            # Add Image Vector
            batch.append(
                Vector(
                    id=f"img_{entry['id']}",
                    vector=entry['image_embedding'],
                    metadata={
                        "original_id": entry['id'],
                        "type": "image",
                        "label": entry['label'],
                        "cluster": entry['cluster'],
                        "normal": entry['normal'],
                        "image_name":entry['image']
                    }
                )
            )

            # Add Text Vector
            batch.append(
                Vector(
                    id=f"txt_{entry['id']}",
                    vector=entry['text_embedding'],
                    metadata={
                        "original_id": entry['id'],
                        "type": "text",
                        "label": entry['label'],
                        "cluster": entry['cluster'],
                        "normal": entry['normal'],
                        "cleaned_text":entry['cleaned_text']
                    }
                )
            )

            # Flush batch
            if len(batch) >= batch_size:
                index.upsert(vectors=batch)
                success_count += len(batch)
                batch = []

        except Exception as e:
            print(f"Error: {e}")
            error_count += 1

    # Final flush
    if batch:
        index.upsert(vectors=batch)
        success_count += len(batch)

    return {
        "success": success_count,
        "skipped": skip_count,
        "errors": error_count
    }

print(insert_vectors(data_path = data_path, index = index, batch_size = 100))