from vector_retrieval_service.domain.ai_models import (
    LanguageModelInterface,
    ScoreComputer,
)

from vector_retrieval_service.domain.factories import (
    LanguageModels,
    SearchScoreFunctions,
)
from vector_retrieval_service.services.data_transformers import (
    transform_distances_to_similarity_response,
    transform_embeddings_to_response,
)
from vector_retrieval_service.services.dtos import (
    TextEmbeddingsResponse,
    TextSimilarityResponse,
)


async def get_text_embedding_service(
    text: str,
    requested_model: LanguageModels,
    llm_interface: LanguageModelInterface,
) -> TextEmbeddingsResponse:
    embeddings = await llm_interface.encode(model_input=[text])
    embedding_response = transform_embeddings_to_response(
        embeddings=embeddings, model=requested_model
    )
    return embedding_response


async def get_text_similarity_service(
    query: str,
    texts: list[str],  # TODO: Rename it as corpus
    score_function: SearchScoreFunctions,
    llm_interface: LanguageModelInterface,
    score_computer: ScoreComputer,
) -> TextSimilarityResponse:
    is_tensor_required = score_function != SearchScoreFunctions.EUCLIDEAN
    embeddings = await llm_interface.encode(
        model_input=texts + [query],
        convert_to_tensor=True,
        convert_to_numpy=False,
        normalize_embeddings=False,
    )

    distances = score_computer.compute_score(
        corpus=embeddings[:-1], query=embeddings[-1]
    )

    res = transform_distances_to_similarity_response(
        original_query=query,
        corpus=texts,
        distances=distances,
        sort_reverse=is_tensor_required,
    )
    return res
