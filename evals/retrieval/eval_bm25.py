"""
Precision@k and Recall@k eval for BM25 sparse retrieval.

Relevance definition: a corpus case is relevant to a query if it shares the
same cluster label (same-cluster = same pathology category).

Usage (from repo root):
    python evals/retrieval/eval_bm25.py [--ks 1,3,5,10] [--top-k 10]

Requires:
    - data/processed/embedding_input.json
    - evals/datasets/eval_cases.json  (run build_eval_dataset.py first)
"""

import sys
import json
import argparse
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))           # evals.retrieval.metrics
sys.path.insert(0, str(REPO_ROOT / "backend"))  # core.retrieval.bm25_index

from core.retrieval.bm25_index import query_bm25
from evals.retrieval.metrics import precision_at_k, recall_at_k, average_precision, summarize

EVAL_DATASET = REPO_ROOT / "evals" / "datasets" / "eval_cases.json"
FULL_DATA = REPO_ROOT / "data" / "processed" / "embedding_input.json"


def load_cluster_lookup():
    with open(FULL_DATA) as f:
        data = json.load(f)
    return {item["id"]: item.get("cluster", "unknown") for item in data}


def relevant_ids_for(query_id: str, query_cluster: str, cluster_lookup: dict) -> set:
    """All corpus IDs in the same cluster, excluding the query case itself."""
    return {
        cid for cid, cluster in cluster_lookup.items()
        if cluster == query_cluster and cid != query_id
    }


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


def run(ks: list, top_k: int):
    if not EVAL_DATASET.exists():
        print(f"[error] {EVAL_DATASET} not found. Run:\n  python evals/datasets/build_eval_dataset.py")
        sys.exit(1)

    with open(EVAL_DATASET) as f:
        eval_cases = json.load(f)

    cluster_lookup = load_cluster_lookup()

    results = []
    for i, case in enumerate(eval_cases):
        retrieved = query_bm25(case["cleaned_text"], top_k=top_k)
        # Exclude the query case itself — it's in the BM25 corpus and would
        # take up a result slot without counting as a hit.
        retrieved_ids = [oid for oid, _ in retrieved if oid != case["id"]]

        relevant = relevant_ids_for(case["id"], case["cluster"], cluster_lookup)

        row = {"id": case["id"], "cluster": case["cluster"]}
        for k in ks:
            row[f"P@{k}"] = precision_at_k(retrieved_ids, relevant, k)
            row[f"R@{k}"] = recall_at_k(retrieved_ids, relevant, k)
        row["AP"] = average_precision(retrieved_ids, relevant)
        results.append(row)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(eval_cases)} evaluated...")

    by_cluster = defaultdict(list)
    for r in results:
        by_cluster[r["cluster"]].append(r)

    overall = summarize(results, ks)

    print(f"\n=== BM25 Retrieval Eval  (top_k={top_k}, n={len(results)}) ===")
    print_table(by_cluster, overall, ks)
    print(f"\nRelevance: same cluster label  |  Query cases excluded from results")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ks", default="1,3,5,10",
                        help="Comma-separated k values (default: 1,3,5,10)")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Number of results to retrieve (default: 10)")
    args = parser.parse_args()
    ks = [int(x) for x in args.ks.split(",")]
    run(ks=ks, top_k=args.top_k)


if __name__ == "__main__":
    main()
