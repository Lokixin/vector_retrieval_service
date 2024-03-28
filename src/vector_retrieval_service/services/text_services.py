from vector_retrieval_service.domain.ai_models import LanguageModelInterface, ScoreComputer

from vector_retrieval_service.domain.factories import (
    LanguageModels,
    SearchScoreFunctions,
)
from vector_retrieval_service.services.data_transformers import transform_distances_to_similarity_response
from vector_retrieval_service.services.dtos import (
    TextEmbeddingsResponse,
    TextSimilarityResponse,
)


async def get_text_embedding_service(
    text: str, requested_model: LanguageModels
) -> TextEmbeddingsResponse:
    model_interface = LanguageModelInterface(
        language_model=requested_model
    )
    embeddings = await model_interface.encode(model_input=[text])
    return TextEmbeddingsResponse(
        embedding=embeddings[0].tolist(), ai_model_used=requested_model
    )


async def get_text_similarity_service(
    query: str,
    texts: list[str],  # TODO: Rename it as corpus
    score_function: SearchScoreFunctions,
    requested_model: LanguageModels,
) -> TextSimilarityResponse:
    model_interface = LanguageModelInterface(
        language_model=requested_model
    )
    score_computer = ScoreComputer(score_function)

    texts.append(query)
    is_tensor_required = score_function != SearchScoreFunctions.EUCLIDEAN

    embeddings = await model_interface.encode(
        model_input=texts,
        convert_to_tensor=is_tensor_required,
        convert_to_numpy=not is_tensor_required,
        normalize_embeddings=False,
    )

    distances = {
        text: score_computer.compute_score(embedding, embeddings[-1])
        for embedding, text in zip(embeddings[:-1], texts)
    }
    res = transform_distances_to_similarity_response(
        original_query=query,
        corpus=texts,
        distances=distances,
        sort_reverse=is_tensor_required,
    )
    return res
