from src.build_index import build_vector_store
from src.retrieval_guard import filter_retrieval_results, should_refuse

def test_guardrails():
    vector_store, embedder = build_vector_store(
        "data/sample_logistics.pdf"
    )

    # --- Case 1: Answerable question ---
    query_ok = "What is customs clearance in international logistics?"
    q_emb = embedder.embed_query(query_ok)

    results = vector_store.search(q_emb, top_k=3)
    approved = filter_retrieval_results(results)

    assert not should_refuse(approved)
    print("Answerable question approved")

    # --- Case 2: Unanswerable question ---
    query_bad = "What is the CEO's favorite color?"
    q_emb_bad = embedder.embed_query(query_bad)

    results_bad = vector_store.search(q_emb_bad, top_k=3)
    approved_bad = filter_retrieval_results(results_bad)

    assert should_refuse(approved_bad)
    print("Unanswerable question correctly refused")


if __name__ == "__main__":
    test_guardrails()
