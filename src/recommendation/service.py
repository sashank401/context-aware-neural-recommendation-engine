from pathlib import Path

from src.retrieval.retriever import Retriever


class RecommendationService:
    """Service layer for generating personalized recommendations."""

    MAX_RECOMMENDATIONS = 100

    def __init__(self, retriever=None, model_dir=None):
        # Allow a retriever to be injected for testing.
        # The trained model is loaded only when recommendations are requested.
        self.retriever = retriever
        self.model_dir = (
            Path(model_dir)
            if model_dir is not None
            else Path(__file__).resolve().parents[2] / "models" / "two_tower"
        )

    def _get_retriever(self):
        """Initialize retrieval with trained embeddings when needed."""
        if self.retriever is None:
            from src.models.numpy_embedding_store import NumpyEmbeddingStore

            embedding_store = NumpyEmbeddingStore(self.model_dir)
            self.retriever = Retriever(embedding_store)

        return self.retriever

    def recommend(self, customer_id: str, k: int = 10):
        """Generate top-k recommendations for a customer."""

        if not customer_id or not customer_id.strip():
            raise ValueError("customer_id is required")

        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be greater than 0")

        if k > self.MAX_RECOMMENDATIONS:
            raise ValueError(
                f"k must not exceed {self.MAX_RECOMMENDATIONS}"
            )

        retriever = self._get_retriever()
        recommendations = retriever.retrieve(customer_id, k)

        return {
            "customer_id": customer_id,
            "recommendations": recommendations,
        }