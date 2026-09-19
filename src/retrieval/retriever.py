from src.retrieval.similarity import cosine_similarity


class Retriever:
    """Retrieve relevant items using embedding similarity."""

    def __init__(self, embedding_store=None):
        self.embedding_store = embedding_store

    def retrieve_with_scores(self, customer_id: str, k: int = 10):
        """Retrieve top-k article IDs with similarity scores."""

        if not customer_id:
            raise ValueError("customer_id is required")

        if k <= 0:
            raise ValueError("k must be greater than 0")

        if self.embedding_store is None:
            return []

        user_embedding = self.embedding_store.get_user_embedding(
            customer_id
        )

        if user_embedding is None:
            return []

        scored_items = []

        for article_id, item_embedding in (
            self.embedding_store.item_embeddings.items()
        ):
            if item_embedding is None:
                continue

            try:
                score = cosine_similarity(
                    user_embedding,
                    item_embedding
                )
            except ValueError:
                continue

            scored_items.append(
                {
                    "article_id": article_id,
                    "score": score
                }
            )

        scored_items.sort(
            key=lambda item: (-item["score"], item["article_id"])
        )

        top_items = scored_items[:k]

        for rank, item in enumerate(top_items, start=1):
            item["rank"] = rank

        return top_items

    def retrieve(self, customer_id: str, k: int = 10):
        """Retrieve top-k relevant article IDs."""

        results = self.retrieve_with_scores(
            customer_id,
            k
        )

        return [
            item["article_id"]
            for item in results
        ]