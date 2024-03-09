from typing import Callable

import pytest_asyncio

from vector_retrieval_service.embedding_retriever.ai_models.model_factory import (
    LLMFactory,
)


@pytest_asyncio.fixture
def get_text_embeddings() -> Callable[[str, str], list[float]]:
    def _get_text_embeddings(model_type: str, text: str) -> list[float]:
        model = LLMFactory.get_model(model_type=model_type)
        model.to("cuda:0")
        embeddings = model.encode(text, device="cuda:0")
        embeddings = embeddings.tolist()
        return embeddings

    return _get_text_embeddings
