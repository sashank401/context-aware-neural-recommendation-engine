from src.retrieval.similarity import cosine_similarity


class Retriever:
    """Retrieve relevant items using embedding similarity."""

    def __init__(self, embedding_store=None):
        self.embedding_store = embedding_store

    def retrieve(self, customer_id: str, k: int = 10):
        """Retrieve top-k relevant article IDs."""

        if not customer_id:
            raise ValueError("customer_id is required")

        if k <= 0:
            raise ValueError("k must be greater than 0")

        if self.embedding_store is None:
            return []

        user_embedding = self.embedding_store.get_user_embedding(customer_id)

        if user_embedding is None:
            return []

        scored_items = []

        for article_id, item_embedding in self.embedding_store.item_embeddings.items():
            if item_embedding is None:
                continue

            score = cosine_similarity(user_embedding, item_embedding)
            scored_items.append((article_id, score))

        scored_items.sort(key=lambda x: x[1], reverse=True)

        return [article_id for article_id, _ in scored_items[:k]]