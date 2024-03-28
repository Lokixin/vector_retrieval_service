from typing import Callable

import pytest

from vector_retrieval_service.domain.factories import (
    LanguageModels,
    SearchScoreFunctions,
)
from vector_retrieval_service.services.dtos import (
    TextEmbeddingsResponse,
    TextSimilarityResponse,
    TextSimilarity,
)
from vector_retrieval_service.services.text_services import (
    get_text_embedding_service,
    get_text_similarity_service,
)


@pytest.mark.asyncio
async def test_get_text_embedding_service(
    get_text_embeddings: Callable[[str, str], list[float]],
) -> None:
    # Test setup
    text = "Sample text to encode"
    requested_language_model = LanguageModels.MINI_LM

    # Ejecución de lo que queremos testear
    result = await get_text_embedding_service(
        text=text, requested_model=requested_language_model
    )

    # Comprobación del resultado
    expected_result = TextEmbeddingsResponse(
        embedding=get_text_embeddings(str(LanguageModels.MINI_LM.value), text),
        ai_model_used=LanguageModels.MINI_LM.value,
    )

    assert result == expected_result


@pytest.mark.asyncio
async def test_get_text_similarity_service() -> None:
    query = "Best Programming languages"
    texts = [
        "Python Java Javascript are the best languages for web dev",
        "In italy pasta is very good",
    ]
    requested_language_model = LanguageModels.MINI_LM
    score_function = SearchScoreFunctions.COSINE_SIMILARITY

    result = await get_text_similarity_service(
        query=query,
        texts=texts,
        requested_model=requested_language_model,
        score_function=score_function,
    )

    text_similarities = [
        TextSimilarity(
            original_text_position=0,
            text=texts[0],
            distance_to_query=result.text_similarities[0].distance_to_query,
        ),
        TextSimilarity(
            original_text_position=1,
            text=texts[1],
            distance_to_query=result.text_similarities[1].distance_to_query,
        ),
    ]
    expected_result = TextSimilarityResponse(
        original_query=query, text_similarities=text_similarities
    )

    assert result == expected_result
