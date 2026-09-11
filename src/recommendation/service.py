from src.retrieval.retriever import Retriever
from src.models.embeddings import EmbeddingStore


class RecommendationService:
    """Service layer for generating personalized recommendations."""

    def __init__(self, retriever=None):
        if retriever is None:
            embedding_store = EmbeddingStore()
            retriever = Retriever(embedding_store)

        self.retriever = retriever

    def recommend(self, customer_id: str, k: int = 10):
        """Generate top-k recommendations for a customer."""

        if not customer_id:
            raise ValueError("customer_id is required")

        if k <= 0:
            raise ValueError("k must be greater than 0")

        recommendations = self.retriever.retrieve(customer_id, k)

        return {
            "customer_id": customer_id,
            "recommendations": recommendations
        }