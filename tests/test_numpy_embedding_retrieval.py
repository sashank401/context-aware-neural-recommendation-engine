import numpy as np

from src.models.numpy_embedding_store import NumpyEmbeddingStore
from src.retrieval.retriever import Retriever


def test_numpy_embeddings_work_with_retriever(tmp_path):
    """Verify NumPy embeddings can power recommendation retrieval."""

    user_ids = np.array(["user_1", "user_2"])
    item_ids = np.array(["item_1", "item_2", "item_3"])

    user_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    item_embeddings = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.7, 0.7],
        ]
    )

    np.save(tmp_path / "user_ids.npy", user_ids)
    np.save(tmp_path / "item_ids.npy", item_ids)
    np.save(tmp_path / "user_embeddings.npy", user_embeddings)
    np.save(tmp_path / "item_embeddings.npy", item_embeddings)

    store = NumpyEmbeddingStore(tmp_path)
    retriever = Retriever(store)

    results = retriever.retrieve_with_scores("user_1", k=3)

    assert len(results) == 3
    assert results[0]["article_id"] == "item_1"
    assert results[0]["rank"] == 1
    assert results[0]["score"] > results[1]["score"]