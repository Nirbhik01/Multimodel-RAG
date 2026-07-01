"""
Build a stratified eval dataset from embedding_input.json.

Samples SAMPLES_PER_CLUSTER cases per cluster (reproducible via SEED).
Output: evals/datasets/eval_cases.json

Usage (from repo root):
    python evals/datasets/build_eval_dataset.py
"""

import json
import random
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DATA_PATH = REPO_ROOT / "data" / "processed" / "embedding_input.json"
OUTPUT_PATH = Path(__file__).parent / "eval_cases.json"

SAMPLES_PER_CLUSTER = 20
SEED = 42


def build():
    with open(DATA_PATH) as f:
        data = json.load(f)

    by_cluster = defaultdict(list)
    for item in data:
        by_cluster[item.get("cluster", "unknown")].append(item)

    random.seed(SEED)
    eval_cases = []
    for cluster in sorted(by_cluster):
        items = by_cluster[cluster]
        sample = random.sample(items, min(SAMPLES_PER_CLUSTER, len(items)))
        for item in sample:
            eval_cases.append({
                "id": item["id"],
                "cluster": cluster,
                "normal": item.get("normal", False),
                "cleaned_text": item.get("cleaned_text") or item.get("text", ""),
                "image": item.get("image"),
            })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(eval_cases, f, indent=2)

    print(f"Saved {len(eval_cases)} eval cases → {OUTPUT_PATH}")
    for cluster in sorted(by_cluster):
        total = len(by_cluster[cluster])
        sampled = min(SAMPLES_PER_CLUSTER, total)
        print(f"  {cluster:<30} {sampled:>3}/{total} sampled")


if __name__ == "__main__":
    build()
