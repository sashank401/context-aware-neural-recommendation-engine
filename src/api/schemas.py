from pydantic import BaseModel


class RecommendationResponse(BaseModel):
    customer_id: str
    recommendations: list[str]