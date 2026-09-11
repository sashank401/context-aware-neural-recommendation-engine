class EmbeddingStore:
    """Interface for storing and retrieving user and item embeddings."""

    def __init__(self):
        self.user_embeddings = {}
        self.item_embeddings = {}

    def get_user_embedding(self, customer_id: str):
        return self.user_embeddings.get(customer_id)

    def get_item_embedding(self, article_id: str):
        return self.item_embeddings.get(article_id)

    def add_user_embedding(self, customer_id: str, embedding):
        self.user_embeddings[customer_id] = embedding

    def add_item_embedding(self, article_id: str, embedding):
        self.item_embeddings[article_id] = embedding