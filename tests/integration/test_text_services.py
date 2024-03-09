from typing import Callable

import pytest

from vector_retrieval_service.embedding_retriever.ai_models.model_factory import (
    LanguageModels,
)
from vector_retrieval_service.service_api.services.dtos import TextEmbeddingsResponse
from vector_retrieval_service.service_api.services.text_services import (
    get_text_embedding_service,
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
        embedding=get_text_embeddings(LanguageModels.MINI_LM.value, text),
        ai_model_used=LanguageModels.MINI_LM.value,
    )

    assert result == expected_result
