# DTO -> Data Transfer Object
from pydantic import BaseModel


class TextEmbeddingsResponse(BaseModel):
    embedding: list[float]
    ai_model_used: str


class TextSimilarity(BaseModel):
    original_text_position: int
    text: str
    distance_to_query: float


class TextSimilarityResponse(BaseModel):
    original_query: str
    text_similarities: list[TextSimilarity]


class TextToImageScore(BaseModel):
    image_identifier: str
    score: float


class TextToImagesSimilarity(BaseModel):
    search_text: str
    scores: list[TextToImageScore]
