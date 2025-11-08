from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings
from typing import List

class CustomEmbeddings(Embeddings):
    def __init__(self, model_name: str="all-MiniLM-L6-v2"):
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            print(f"Warning: Could not load {model_name}, falling back to all-MiniLM-L6-v2")
            self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        try:
            embeddings = self.model.encode(documents)
            if hasattr(embeddings, 'tolist'):
                return embeddings.tolist()
            else:
                return [emb.tolist() for emb in embeddings]
        except Exception as e:
            print(f"Error in embed_documents: {e}")
            # Return dummy embeddings
            return [[0.0] * 384 for _ in documents]

    def embed_query(self, query: str) -> List[float]:
        try:
            embedding = self.model.encode([query])
            # Ensure a flat list (shape: [dim]) regardless of backend
            if isinstance(embedding, (list, tuple)):
                # embedding is likely a list with a single inner list
                first = embedding[0]
                return first if isinstance(first, list) else list(first)
            if hasattr(embedding, 'shape') and getattr(embedding, 'ndim', 1) == 2:
                return embedding[0].tolist()
            if hasattr(embedding, 'tolist'):
                out = embedding.tolist()
                return out[0] if isinstance(out, list) and out and isinstance(out[0], list) else out
            # Fallback: coerce to list
            return list(embedding)
        except Exception as e:
            print(f"Error in embed_query: {e}")
            # Return dummy embedding
            return [0.0] * 384

