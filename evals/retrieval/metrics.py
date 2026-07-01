def precision_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    top_k = retrieved_ids[:k]
    hits = sum(1 for r in top_k if r in relevant_ids)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved_ids: list, relevant_ids: set, k: int) -> float:
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for r in top_k if r in relevant_ids)
    return hits / len(relevant_ids)


def average_precision(retrieved_ids: list, relevant_ids: set) -> float:
    """AP for a single query — area under the P-R curve."""
    if not relevant_ids:
        return 0.0
    hits = 0
    ap = 0.0
    for i, r in enumerate(retrieved_ids, 1):
        if r in relevant_ids:
            hits += 1
            ap += hits / i
    return ap / len(relevant_ids)


def summarize(results: list, ks: list) -> dict:
    """Mean P@k, R@k, and MAP across all query results."""
    if not results:
        return {}
    n = len(results)
    summary = {}
    for k in ks:
        summary[f"P@{k}"] = sum(r[f"P@{k}"] for r in results) / n
        summary[f"R@{k}"] = sum(r[f"R@{k}"] for r in results) / n
    summary["MAP"] = sum(r["AP"] for r in results) / n
    return summary
