from typing import Any

import numpy as np
import torch

from vector_retrieval_service.domain.ai_models import (
    LanguageModelInterface,
    ScoreComputer,
)
from vector_retrieval_service.domain.factories import LanguageModels


class FakeModelInterface(LanguageModelInterface):
    def __init__(self, language_model: LanguageModels, output: Any) -> None:
        super().__init__(language_model)
        self.output = output

    async def encode(
        self,
        model_input: list[str] | list[Any],
        convert_to_tensor: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> Any:
        if convert_to_numpy:
            return np.array(self.output)
        if convert_to_tensor:
            return torch.tensor(self.output)


class FakeScoreComputer(ScoreComputer):
    def compute_score(self, query: Any, corpus: Any) -> Any:
        return np.linalg.norm(query - corpus)
