from fastapi.testclient import TestClient

from src.api.main import app


client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Recommendation Engine API is running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_recommendation_endpoint():
    response = client.get("/recommend/customer_001?k=5")

    assert response.status_code == 200

    data = response.json()

    assert data["customer_id"] == "customer_001"
    assert "recommendations" in data


def test_recommendation_uses_default_k():
    response = client.get("/recommend/customer_001")

    assert response.status_code == 200

    data = response.json()

    assert data["customer_id"] == "customer_001"
    assert isinstance(data["recommendations"], list)
    assert len(data["recommendations"]) <= 10


def test_recommendation_rejects_zero_k():
    response = client.get("/recommend/customer_001?k=0")

    assert response.status_code == 422


def test_recommendation_rejects_negative_k():
    response = client.get("/recommend/customer_001?k=-5")

    assert response.status_code == 422


def test_recommendation_rejects_excessive_k():
    response = client.get("/recommend/customer_001?k=101")

    assert response.status_code == 422