import pytest

from src.recommendation.service import RecommendationService


def test_recommendation_service():
    service = RecommendationService()

    result = service.recommend(
        "customer_001",
        5
    )

    assert result["customer_id"] == "customer_001"
    assert "recommendations" in result


def test_recommendation_service_rejects_empty_customer_id():
    service = RecommendationService()

    with pytest.raises(
        ValueError,
        match="customer_id is required"
    ):
        service.recommend("", 5)


def test_recommendation_service_rejects_zero_k():
    service = RecommendationService()

    with pytest.raises(
        ValueError,
        match="k must be greater than 0"
    ):
        service.recommend(
            "customer_001",
            0
        )


def test_recommendation_service_rejects_excessive_k():
    service = RecommendationService()

    with pytest.raises(
        ValueError,
        match="k must not exceed 100"
    ):
        service.recommend(
            "customer_001",
            101
        )


def test_recommendation_service_rejects_whitespace_customer_id():
    service = RecommendationService()

    with pytest.raises(
        ValueError,
        match="customer_id is required"
    ):
        service.recommend(
            "   ",
            5
        )