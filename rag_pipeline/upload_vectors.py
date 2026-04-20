from dotenv import load_dotenv
import json
from pathlib import Path
import os
from upstash_vector import Index, Vector
import os

load_dotenv()

root_path = Path(__file__).parent.parent

data_path = root_path / 'chest-x-ray-data' / 'embedding_results.json'

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

    for entry in data:
        try:
            if 'id' not in entry or 'final_embedding' not in entry:
                skip_count += 1
                continue

            batch.append(
                Vector(
                    id=entry['id'],
                    vector=entry['final_embedding'],
                    metadata={}
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