
import numpy as np

from src.retrieval.similarity import cosine_similarity


class Retriever:
    """Retrieve top-k items using vectorized cosine similarity."""

    def __init__(self, embedding_store=None):
        self.embedding_store = embedding_store

    def retrieve_with_scores(self, customer_id: str, k: int = 10):
        """Return ranked article IDs with cosine similarity scores."""

        if not customer_id:
            raise ValueError("customer_id is required")

        if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
            raise ValueError("k must be a positive integer")

        if self.embedding_store is None:
            return []

        user_embedding = self.embedding_store.get_user_embedding(
            customer_id
        )

        if user_embedding is None:
            return []

        item_embeddings = self.embedding_store.item_embeddings

        if not item_embeddings:
            return []

        # Keep IDs and vectors aligned.
        article_ids = list(item_embeddings.keys())
        vectors = []

        for article_id in article_ids:
            vector = item_embeddings[article_id]
            if vector is None:
                continue

            vector = np.asarray(vector, dtype=float)

            # Ignore malformed vectors instead of failing retrieval.
            if vector.ndim != 1:
                continue

            vectors.append((article_id, vector))

        if not vectors:
            return []

        # Keep only vectors with the user's embedding dimension.
        user_vector = np.asarray(user_embedding, dtype=float)

        if user_vector.ndim != 1:
            return []

        valid = [
            (article_id, vector)
            for article_id, vector in vectors
            if vector.shape == user_vector.shape
        ]

        if not valid:
            return []

        article_ids = [article_id for article_id, _ in valid]
        matrix = np.stack([vector for _, vector in valid])

        # Vectorized cosine similarity across all items.
        user_norm = np.linalg.norm(user_vector)
        item_norms = np.linalg.norm(matrix, axis=1)
        denominators = item_norms * user_norm

        scores = np.zeros(len(article_ids), dtype=float)
        nonzero = denominators > 0

        scores[nonzero] = (
            matrix[nonzero] @ user_vector
        ) / denominators[nonzero]

        # Preserve deterministic tie-breaking by article ID.
        order = sorted(
            range(len(article_ids)),
            key=lambda i: (-scores[i], str(article_ids[i]))
        )[:k]

        return [
            {
                "article_id": str(article_ids[i]),
                "score": float(scores[i]),
                "rank": rank,
            }
            for rank, i in enumerate(order, start=1)
        ]

    def retrieve(self, customer_id: str, k: int = 10):
        """Return top-k article IDs."""

        return [
            item["article_id"]
            for item in self.retrieve_with_scores(customer_id, k)
        ]
