from typing import Any

import asyncer

from vector_retrieval_service.domain import factories
from vector_retrieval_service.domain.factories import LanguageModels, SearchScoreFunctions


class LanguageModelInterface:
    def __init__(self, language_model: LanguageModels) -> None:
        self.language_model = factories.make_llm(language_model)

    async def encode(self, model_input: list[str], convert_to_tensor: bool = False, convert_to_numpy: bool = True, normalize_embeddings: bool = False) -> Any:
        output = await asyncer.asyncify(self.language_model.encode)(
            model_input,
            convert_to_tensor=convert_to_tensor,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
        )
        return output


class ScoreComputer:
    def __init__(self, score: SearchScoreFunctions) -> None:
        self.score_function = factories.make_score_function(score)

    def compute_score(self, query: Any, corpus: Any) -> Any:
        return self.score_function(
            query,
            corpus
        )
