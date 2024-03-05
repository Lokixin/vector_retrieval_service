from fastapi import APIRouter, Query, Body

from vector_retrieval_service.embedding_retriever.ai_models.model_factory import (
    LanguageModels,
    SearchScoreFunctions,
)
from vector_retrieval_service.service_api.services.dtos import (
    TextEmbeddingsResponse,
    TextSimilarityResponse,
)
from vector_retrieval_service.service_api.services.text_services import (
    get_text_embedding_service,
    get_text_similarity_service,
)


text_embeddings_example = [
    "Tutorial para aprender a programar en Python",
    "Las manzanas son la fruta de moda en españa",
    "Cada día más alumnos suspende matemáticas",
    "El mejor canal de Youtube de programación en Python es Dimas",
]


retrieval_service = APIRouter(
    prefix="/text_processing", tags=["Text Processing Services"]
)


@retrieval_service.get(
    "/get_embeddings", description="Get embeddings for a short plain text"
)
async def get_text_embeddings(
    text: str, requested_model: LanguageModels
) -> TextEmbeddingsResponse:
    return await get_text_embedding_service(text=text, requested_model=requested_model)


@retrieval_service.post(
    "/get_text_similarity",
    description="Compute similarity between a query and a list of texts",
)
async def get_text_similarity(
    requested_model: LanguageModels = LanguageModels.MINI_LM,
    score_function: SearchScoreFunctions = SearchScoreFunctions.EUCLIDEAN,
    query: str = Query(default="Quiero aprender a programar"),
    texts: list[str] = Body(examples=[text_embeddings_example]),
) -> TextSimilarityResponse:
    return await get_text_similarity_service(
        query=query,
        texts=texts,
        requested_model=requested_model,
        score_function=score_function,
    )
