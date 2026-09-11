from src.recommendation.service import RecommendationService


def test_recommendation_service():
    service = RecommendationService()

    result = service.recommend("customer_001", 5)

    assert result["customer_id"] == "customer_001"
    assert "recommendations" in result