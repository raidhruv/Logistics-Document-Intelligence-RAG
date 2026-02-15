from src.build_index import build_vector_store

def test_retrieval():
    vector_store, embedder = build_vector_store(
    "data/sample_logistics.pdf"
)


    query = "What happens if a shipment is delayed due to customs?"
    query_embedding = embedder.embed_query(query)

    results = vector_store.search(query_embedding, top_k=3)

    assert len(results) > 0

    print("\nTop retrieved chunk:\n")
    print(results[0]["metadata"]["text"])
    print("\nSimilarity score:", results[0]["score"])

if __name__ == "__main__":
    test_retrieval()
