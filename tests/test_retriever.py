from src.models.embeddings import EmbeddingStore
from src.retrieval.retriever import Retriever


def test_retriever_ranks_items_by_similarity():
    store = EmbeddingStore()

    store.add_user_embedding("customer_001", [1, 0])

    store.add_item_embedding("article_A", [1, 0])
    store.add_item_embedding("article_B", [0.8, 0.2])
    store.add_item_embedding("article_C", [-1, 0])

    retriever = Retriever(store)

    result = retriever.retrieve("customer_001", k=2)

    assert result == ["article_A", "article_B"]