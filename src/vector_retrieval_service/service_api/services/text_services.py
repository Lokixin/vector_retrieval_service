import asyncer

from vector_retrieval_service.embedding_retriever.ai_models.model_factory import (
    LLMFactory,
    LanguageModels,
    SearchScoreFunctions,
)
from vector_retrieval_service.service_api.services.dtos import (
    TextEmbeddingsResponse,
    TextSimilarityResponse,
    TextSimilarity,
)


async def get_text_embedding_service(
    text: str, requested_model: LanguageModels
) -> TextEmbeddingsResponse:
    model = LLMFactory.get_model(requested_model)
    embeddings = await asyncer.asyncify(model.encode)(text)
    return TextEmbeddingsResponse(
        embedding=embeddings.tolist(), ai_model_used=requested_model
    )


async def get_text_similarity_service(
    query: str,
    texts: list[str],
    score_function: SearchScoreFunctions,
    requested_model: LanguageModels,
) -> TextSimilarityResponse:
    model = LLMFactory.get_model(requested_model)
    compute_score = LLMFactory.get_score_function(score_function)
    is_tensor_required = score_function != SearchScoreFunctions.EUCLIDEAN
    texts.append(query)
    embeddings = await asyncer.asyncify(model.encode)(
        texts,
        convert_to_tensor=is_tensor_required,
        convert_to_numpy=not is_tensor_required,
        normalize_embeddings=False,
    )
    distances = {
        text: compute_score(embedding, embeddings[-1])
        for embedding, text in zip(embeddings[:-1], texts)
    }
    distances = dict(
        sorted(distances.items(), key=lambda item: item[1], reverse=is_tensor_required)
    )
    similarities = TextSimilarityResponse(
        original_query=query,
        text_similarities=[
            TextSimilarity(
                distance_to_query=distance,
                text=text,
                original_text_position=texts.index(text),
            )
            for text, distance in distances.items()
        ],
    )
    return similarities
