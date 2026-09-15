from fastapi import FastAPI, HTTPException, Query

from src.recommendation.service import RecommendationService


app = FastAPI(
    title="Context-Aware Neural Recommendation Engine",
    version="0.1.0"
)

recommendation_service = RecommendationService()


@app.get("/")
def root():
    return {"message": "Recommendation Engine API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/recommend/{customer_id}")
def recommend(
    customer_id: str,
    k: int = Query(default=10, ge=1, le=100)
):
    try:
        return recommendation_service.recommend(customer_id, k)

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error)
        )