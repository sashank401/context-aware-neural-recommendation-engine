class Retriever:
    """Interface for retrieving relevant items for a customer."""

    def __init__(self, embedding_store=None):
        self.embedding_store = embedding_store

    def retrieve(self, customer_id: str, k: int = 10):
        """Retrieve top-k relevant article IDs."""

        if not customer_id:
            raise ValueError("customer_id is required")

        if k <= 0:
            raise ValueError("k must be greater than 0")

        return []
