from src.models.embeddings import EmbeddingStore
from src.retrieval.retriever import Retriever


def test_retriever_ranks_items_by_similarity():
    store = EmbeddingStore()

    store.add_user_embedding(
        "customer_001",
        [1, 0]
    )

    store.add_item_embedding(
        "article_A",
        [1, 0]
    )

    store.add_item_embedding(
        "article_B",
        [0.8, 0.2]
    )

    store.add_item_embedding(
        "article_C",
        [-1, 0]
    )

    retriever = Retriever(store)

    result = retriever.retrieve(
        "customer_001",
        k=2
    )

    assert result == [
        "article_A",
        "article_B"
    ]


def test_retriever_returns_scores():
    store = EmbeddingStore()

    store.add_user_embedding(
        "customer_001",
        [1, 0]
    )

    store.add_item_embedding(
        "article_A",
        [1, 0]
    )

    store.add_item_embedding(
        "article_B",
        [0, 1]
    )

    retriever = Retriever(store)

    result = retriever.retrieve_with_scores(
        "customer_001",
        k=2
    )

    assert len(result) == 2
    assert result[0]["article_id"] == "article_A"
    assert result[0]["score"] == 1.0
    assert result[1]["article_id"] == "article_B"
    assert result[1]["score"] == 0.0


def test_retriever_handles_missing_user_embedding():
    store = EmbeddingStore()

    store.add_item_embedding(
        "article_A",
        [1, 0]
    )

    retriever = Retriever(store)

    result = retriever.retrieve(
        "unknown_customer",
        k=5
    )

    assert result == []


def test_retriever_handles_empty_embedding_store():
    store = EmbeddingStore()

    store.add_user_embedding(
        "customer_001",
        [1, 0]
    )

    retriever = Retriever(store)

    result = retriever.retrieve(
        "customer_001",
        k=5
    )

    assert result == []