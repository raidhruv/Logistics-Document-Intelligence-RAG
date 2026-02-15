from src.build_index import build_vector_store
from src.retrieval_guard import filter_retrieval_results, should_refuse
from src.llm import build_prompt, call_granite


def answer_query(query: str):
    # Build / load vector store
    
    from src.vector_store import FaissVectorStore
    from src.embeddings import EmbeddingGenerator

    vector_store = FaissVectorStore.load("storage")
    embedder = EmbeddingGenerator()

    # Embed query
    q_emb = embedder.embed_query(query)

    # Retrieve top-k
    results = vector_store.search(q_emb, top_k=5)
    
    retrieved_chunks = [r["metadata"]["text"] for r in results]

    context = "\n\n".join(retrieved_chunks)

    # Apply similarity threshold filter
    approved = filter_retrieval_results(results)

    # Guardrail first
    if should_refuse(approved):
       return {
        "answer": "I cannot find the answer in the provided documents.",
        "confidence": 0.0,
        "citations": []
    } 

    # Now safe to compute
    max_score = max(r["score"] for r in approved)
    confidence = float(max_score)

    prompt = build_prompt(query, approved)
    answer = call_granite(prompt)

    return {
        "answer": answer,
        "confidence": confidence,
        "citations": approved
}


