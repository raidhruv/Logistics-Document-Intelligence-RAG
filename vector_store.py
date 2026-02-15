# src/vector_store.py

import faiss
import numpy as np
import os
import pickle


class FaissVectorStore:
    def __init__(self, embedding_dim: int):
        """
        Uses cosine similarity via inner product on normalized vectors.
        """
        self.index = faiss.IndexFlatIP(embedding_dim)
        self.metadata = []  # maps index → chunk metadata

    def add(self, embeddings: np.ndarray, metadatas: list[dict]):
        assert len(embeddings) == len(metadatas), "Embeddings/metadata mismatch"

        embeddings = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings)   # <-- ADD THIS

        self.index.add(embeddings)
        self.metadata.extend(metadatas)


    def search(self, query_embedding: np.ndarray, top_k: int = 5):
        """
        Returns list of (metadata, score)
        """
        query_embedding = np.array([query_embedding]).astype("float32")
        faiss.normalize_L2(query_embedding)   # <-- ADD THIS

        scores, indices = self.index.search(query_embedding, top_k)


        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "score": float(score),
                "metadata": self.metadata[idx]
            })

        return results
    
    def save(self, folder_path: str):
        os.makedirs(folder_path, exist_ok=True)

        faiss.write_index(self.index, os.path.join(folder_path, "index.faiss"))

        with open(os.path.join(folder_path, "metadata.pkl"), "wb") as f:
            pickle.dump(self.metadata, f)


    @classmethod
    def load(cls, folder_path: str):
        index_path = os.path.join(folder_path, "index.faiss")
        metadata_path = os.path.join(folder_path, "metadata.pkl")

        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            raise ValueError("No persisted index found.")

        index = faiss.read_index(index_path)

        with open(metadata_path, "rb") as f:
            metadata = pickle.load(f)

        obj = cls(index.d)
        obj.index = index
        obj.metadata = metadata

        return obj

