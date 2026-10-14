from unittest.mock import Mock, patch

import pytest

from src.recommendation.service import RecommendationService


def test_recommendation_service():
    retriever = Mock()
    retriever.retrieve.return_value = [
        "707704003",
        "779725005",
        "594978004",
        "566140015",
        "664405005",
    ]

    service = RecommendationService(retriever=retriever)

    result = service.recommend("customer_001", 5)

    assert result["customer_id"] == "customer_001"
    assert "recommendations" in result
    assert len(result["recommendations"]) == 5
    retriever.retrieve.assert_called_once_with("customer_001", 5)


def test_recommendation_service_rejects_empty_customer_id():
    service = RecommendationService()

    with pytest.raises(
        ValueError,
        match="customer_id is required",
    ):
        service.recommend("", 5)


def test_recommendation_service_rejects_zero_k():
    service = RecommendationService()

    with pytest.raises(
        ValueError,
        match="k must be greater than 0",
    ):
        service.recommend("customer_001", 0)


def test_recommendation_service_rejects_excessive_k():
    service = RecommendationService()

    with pytest.raises(
        ValueError,
        match="k must not exceed 100",
    ):
        service.recommend("customer_001", 101)


def test_recommendation_service_rejects_whitespace_customer_id():
    service = RecommendationService()

    with pytest.raises(
        ValueError,
        match="customer_id is required",
    ):
        service.recommend("   ", 5)


def test_service_loads_retriever_lazily():
    service = RecommendationService()

    # The trained embeddings must not load during service initialization.
    assert service.retriever is None

    with patch(
        "src.models.numpy_embedding_store.NumpyEmbeddingStore"
    ) as store_class, patch(
        "src.recommendation.service.Retriever"
    ) as retriever_class:
        retriever_class.return_value.retrieve.return_value = [
            "707704003",
            "779725005",
        ]

        result = service.recommend("real_customer", 2)

        store_class.assert_called_once_with(service.model_dir)
        retriever_class.assert_called_once_with(store_class.return_value)
        retriever_class.return_value.retrieve.assert_called_once_with(
            "real_customer", 2
        )

        assert result == {
            "customer_id": "real_customer",
            "recommendations": [
                "707704003",
                "779725005",
            ],
        }


def test_service_reuses_initialized_retriever():
    retriever = Mock()
    retriever.retrieve.return_value = ["707704003"]

    service = RecommendationService(retriever=retriever)

    first_result = service.recommend("customer_a", 1)
    second_result = service.recommend("customer_b", 1)

    assert first_result["recommendations"] == ["707704003"]
    assert second_result["recommendations"] == ["707704003"]
    assert retriever.retrieve.call_count == 2
