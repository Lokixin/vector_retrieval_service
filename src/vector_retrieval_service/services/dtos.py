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


class TextsToImageSimilarity(BaseModel):
    text: str
    score: float


class TextsToImageSimilarities(BaseModel):
    similarities: list[TextsToImageSimilarity]


class TextToImageScore(BaseModel):
    image_identifier: str | None = ""
    score: float


class TextToImagesSimilarity(BaseModel):
    search_text: str
    scores: list[TextToImageScore]


class ImageEmbeddingsResponse(BaseModel):
    embeddings: list[float]
    ai_model_used: str = "CLIP"
