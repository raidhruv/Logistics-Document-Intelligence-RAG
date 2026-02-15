# src/retrieval_guard.py

SIMILARITY_THRESHOLD = 0.58
TOP_K = 5

def filter_retrieval_results(results: list[dict]) -> list[dict]:
    """
    Filters FAISS retrieval results using a similarity threshold.

    Args:
        results: list of {"score": float, "metadata": dict}

    Returns:
        List of approved results (may be empty)
    """
    approved = [
        r for r in results
        if r["score"] >= SIMILARITY_THRESHOLD
    ]
    return approved


def should_refuse(approved_results: list[dict]) -> bool:
    """
    Decide whether to refuse answering.
    """
    return len(approved_results) == 0
