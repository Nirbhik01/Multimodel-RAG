"""
Precision@k and Recall@k eval for PubMedBERT text-vector search (Upstash).

Relevance definition: same cluster label as the query case.

Usage (from repo root):
    python evals/retrieval/eval_vector_search.py [--ks 1,3,5,10]

Requires:
    - backend/.env  with UPSTASH_DB_URL and UPSTASH_READ_ONLY_TOKEN
    - evals/datasets/eval_cases.json  (run build_eval_dataset.py first)
    - PubMedBERT + Rad-DINO downloaded (loaded on first run, ~minutes)
"""

import sys
import json
import argparse
import os
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))               # evals.retrieval.metrics
sys.path.insert(0, str(REPO_ROOT / "backend"))   # core.*

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / "backend" / ".env")

from upstash_vector import Index

from core.embedding.generate_embeddings import get_text_embedding
from core.embedding.medical_models import get_medical_models
from core.retrieval.query_vector_db import query_vector_db
from evals.retrieval.metrics import precision_at_k, recall_at_k, average_precision, summarize

EVAL_DATASET = REPO_ROOT / "evals" / "datasets" / "eval_cases.json"
FULL_DATA = REPO_ROOT / "data" / "processed" / "embedding_input.json"


def load_cluster_lookup():
    with open(FULL_DATA) as f:
        data = json.load(f)
    return {item["id"]: item.get("cluster", "unknown") for item in data}


def relevant_ids_for(query_id: str, query_cluster: str, cluster_lookup: dict) -> set:
    return {
        cid for cid, cluster in cluster_lookup.items()
        if cluster == query_cluster and cid != query_id
    }


def extract_ids(results) -> list:
    """Deduplicated original_ids in score-descending order from Upstash results."""
    seen = set()
    ids = []
    for r in results:
        if r.metadata and "original_id" in r.metadata:
            oid = r.metadata["original_id"]
            if oid not in seen:
                ids.append(oid)
                seen.add(oid)
    return ids


def print_table(by_cluster: dict, overall_summary: dict, ks: list):
    col = "  ".join(f"P@{k}  R@{k}" for k in ks)
    header = f"{'Cluster':<32}  {col}    MAP"
    sep = "-" * len(header)
    print(f"\n{header}")
    print(sep)

    for cluster in sorted(by_cluster):
        rows = by_cluster[cluster]
        cs = summarize(rows, ks)
        line = f"{cluster:<32}"
        for k in ks:
            line += f"  {cs[f'P@{k}']:.3f}  {cs[f'R@{k}']:.3f}"
        line += f"  {cs['MAP']:.3f}"
        print(line)

    print(sep)
    line = f"{'OVERALL':<32}"
    for k in ks:
        line += f"  {overall_summary[f'P@{k}']:.3f}  {overall_summary[f'R@{k}']:.3f}"
    line += f"  {overall_summary['MAP']:.3f}"
    print(line)


def run(ks: list):
    url = os.getenv("UPSTASH_DB_URL")
    token = os.getenv("UPSTASH_READ_ONLY_TOKEN")
    if not url or not token:
        print("[error] UPSTASH_DB_URL or UPSTASH_READ_ONLY_TOKEN not set in backend/.env")
        sys.exit(1)

    if not EVAL_DATASET.exists():
        print(f"[error] {EVAL_DATASET} not found. Run:\n  python evals/datasets/build_eval_dataset.py")
        sys.exit(1)

    with open(EVAL_DATASET) as f:
        eval_cases = json.load(f)

    cluster_lookup = load_cluster_lookup()

    print("Loading PubMedBERT (first run may take a few minutes)...")
    text_model, _, _ = get_medical_models()
    index = Index(url=url, token=token)
    print(f"Ready. Evaluating {len(eval_cases)} queries...\n")

    results = []
    for i, case in enumerate(eval_cases):
        embedding = get_text_embedding(case["cleaned_text"], text_model)
        raw = query_vector_db(embedding.tolist(), index, "text")
        retrieved_ids = [oid for oid in extract_ids(raw) if oid != case["id"]]

        relevant = relevant_ids_for(case["id"], case["cluster"], cluster_lookup)

        row = {"id": case["id"], "cluster": case["cluster"]}
        for k in ks:
            row[f"P@{k}"] = precision_at_k(retrieved_ids, relevant, k)
            row[f"R@{k}"] = recall_at_k(retrieved_ids, relevant, k)
        row["AP"] = average_precision(retrieved_ids, relevant)
        results.append(row)

        if (i + 1) % 10 == 0:
            print(f"  {i + 1}/{len(eval_cases)} evaluated...")

    by_cluster = defaultdict(list)
    for r in results:
        by_cluster[r["cluster"]].append(r)

    overall = summarize(results, ks)

    print(f"\n=== Text Vector Search Eval  (PubMedBERT → Upstash, n={len(results)}) ===")
    print_table(by_cluster, overall, ks)
    print(f"\nRelevance: same cluster label  |  Query cases excluded from results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", default="1,3,5,10",
                        help="Comma-separated k values (default: 1,3,5,10)")
    args = parser.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    run(ks=ks)


if __name__ == "__main__":
    main()
