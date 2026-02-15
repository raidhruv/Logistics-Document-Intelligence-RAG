# src/build_index.py

from src.ingestion import ingest_pdf
from src.embeddings import EmbeddingGenerator
from src.vector_store import FaissVectorStore

def build_vector_store(pdf_path):
    chunks = ingest_pdf(pdf_path)

    embedder = EmbeddingGenerator()
    embeddings = embedder.embed_texts([c["text"] for c in chunks])

    vector_store = FaissVectorStore(embeddings.shape[1])
    vector_store.add(embeddings, chunks)

    vector_store.save("storage")   # ← NEW LINE

    return vector_store, embedder


