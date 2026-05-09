from typing import Callable

import pytest

from vector_retrieval_service.domain import factories
from vector_retrieval_service.config import DEVICE


@pytest.fixture
def get_text_embeddings() -> Callable[[str, str], list[float]]:
    def _get_text_embeddings(model_type: str, text: str) -> list[float]:
        model = factories.make_llm(model_type=model_type)
        model.to(DEVICE)
        embeddings = model.encode(text, device=DEVICE)
        embeddings = embeddings.tolist()
        return embeddings

    return _get_text_embeddings


@pytest.fixture
def get_text_similarity() -> Callable[[str, str, str, str], float]:
    def _get_text_similarity(
        query: str, corpus_text: str, model_type: str, score_function: str
    ) -> float:
        model = factories.make_llm(model_type=model_type)
        model.to(DEVICE)
        query_embedding = model.encode(
            query,
            device=DEVICE,
            normalize_embeddings=False,
            convert_to_tensor=True,
            convert_to_numpy=False,
        )
        corpus_embedding = model.encode(
            corpus_text,
            device=DEVICE,
            normalize_embeddings=False,
            convert_to_tensor=True,
            convert_to_numpy=False,
        )

        compute_score = factories.make_score_function(score_function)
        return float(compute_score(corpus_embedding, query_embedding))

    return _get_text_similarity
