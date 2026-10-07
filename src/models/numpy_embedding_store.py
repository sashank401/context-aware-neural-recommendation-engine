from src.models.embedding_loader import NumpyEmbeddingLoader


class NumpyEmbeddingStore:
    """Expose NumPy embeddings through the retrieval store interface."""

    def __init__(self, model_dir):
        self.loader = NumpyEmbeddingLoader(model_dir).load()

    def get_user_embedding(self, customer_id):
        """Return the embedding for a customer ID."""
        return self.loader.get_user_embedding(customer_id)

    def get_item_embedding(self, article_id):
        """Return the embedding for an article ID."""
        return self.loader.get_item_embedding(article_id)

    @property
    def item_embeddings(self):
        """Return item embeddings keyed by article ID."""

        return {
            str(article_id): self.loader.item_embeddings[index]
            for index, article_id in enumerate(self.loader.item_ids)
        }