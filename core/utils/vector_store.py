import math
from typing import List


class VectorStoreItem:
    def __init__(self, embedding: List[float], document: str):
        if not isinstance(embedding, list) or not all(
            isinstance(x, (int, float)) for x in embedding
        ):
            raise ValueError("Embedding must be a list of numbers.")
        if not isinstance(document, str):
            raise ValueError("Document must be a string.")

        self.embedding = embedding
        self.document = document
        self.norm = math.sqrt(sum(x * x for x in embedding))  # Precompute norm


class VectorStore:
    def __init__(self):
        self.vector_store: List[VectorStoreItem] = []

    async def add_embedding(self, embedding: List[float], document: str):
        if not isinstance(embedding, list) or not all(
            isinstance(x, (int, float)) for x in embedding
        ):
            raise ValueError("Embedding must be a list of numbers.")
        if not isinstance(document, str):
            raise ValueError("Document must be a string.")

        self.vector_store.append(VectorStoreItem(embedding, document))

    async def search(self, query_embedding: List[float], top_k: int = 3) -> List[str]:
        if not isinstance(query_embedding, list) or not all(
            isinstance(x, (int, float)) for x in query_embedding
        ):
            raise ValueError("Query embedding must be a list of numbers.")

        if len(self.vector_store) == 0:
            return []

        query_norm = math.sqrt(sum(x * x for x in query_embedding))
        if query_norm == 0:
            # Avoid division by zero
            return [item.document for item in self.vector_store[:top_k]]

        scored = []
        for item in self.vector_store:
            dot_product = sum(a * b for a, b in zip(query_embedding, item.embedding))
            score = dot_product / (query_norm * item.norm)
            scored.append((score, item.document))

        # Sort descending by score and take top_k documents
        scored.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scored[:top_k]]

    def cosine_similarity(self, vec_a: List[float], vec_b: List[float]) -> float:
        if len(vec_a) != len(vec_b):
            raise ValueError("Vectors must have the same dimension.")

        dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
        norm_a = math.sqrt(sum(a * a for a in vec_a))
        norm_b = math.sqrt(sum(b * b for b in vec_b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)
