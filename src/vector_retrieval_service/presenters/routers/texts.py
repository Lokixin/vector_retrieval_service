from fastapi import APIRouter, Query, Body, Depends

from vector_retrieval_service.domain.ai_models import (
    LanguageModelInterface,
    ScoreComputer,
)
from vector_retrieval_service.domain.factories import (
    LanguageModels,
    SearchScoreFunctions,
)
from vector_retrieval_service.presenters.dependencies import (
    get_language_model_interface,
    get_score_computer,
)
from vector_retrieval_service.services.dtos import (
    TextEmbeddingsResponse,
    TextSimilarityResponse,
)
from vector_retrieval_service.services.text_services import (
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
    text: str,
    requested_model: LanguageModels,
    llm_interface: LanguageModelInterface = Depends(get_language_model_interface),
) -> TextEmbeddingsResponse:
    return await get_text_embedding_service(
        text=text,
        requested_model=requested_model,
        llm_interface=llm_interface,
    )


@retrieval_service.post(
    "/get_text_similarity",
    description="Compute similarity between a query and a list of texts",
)
async def get_text_similarity(
    requested_model: LanguageModels = LanguageModels.MINI_LM,
    score_function: SearchScoreFunctions = SearchScoreFunctions.EUCLIDEAN,
    query: str = Query(default="Quiero aprender a programar"),
    texts: list[str] = Body(examples=[text_embeddings_example]),
    llm_interface: LanguageModelInterface = Depends(get_language_model_interface),
    score_computer: ScoreComputer = Depends(get_score_computer),
) -> TextSimilarityResponse:
    return await get_text_similarity_service(
        query=query,
        texts=texts,
        score_function=score_function,
        llm_interface=llm_interface,
        score_computer=score_computer,
    )
