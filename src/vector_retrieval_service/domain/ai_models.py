from typing import Any

import asyncer
import torch
from PIL.Image import Image

from vector_retrieval_service.config import DEVICE
from vector_retrieval_service.domain import factories
from vector_retrieval_service.domain.factories import (
    LanguageModels,
    SearchScoreFunctions,
    ImageModels,
)


class LanguageModelInterface:
    def __init__(self, language_model: LanguageModels) -> None:
        self.language_model = factories.make_llm(language_model)

    async def encode(
        self,
        model_input: list[str],
        convert_to_tensor: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> Any:
        output = await asyncer.asyncify(self.language_model.encode)(
            model_input,
            convert_to_tensor=convert_to_tensor,
            convert_to_numpy=convert_to_numpy,
            normalize_embeddings=normalize_embeddings,
        )
        return output


class ImageModelInterface:
    def __init__(self, image_model: ImageModels) -> None:
        self.pre_processor, self.model = factories.make_image_model(
            model_type=image_model
        )

    def _encode(
        self, image: Image, padding: bool = False, text: str | list[str] | None = None
    ) -> Any:
        processed_input = self.pre_processor(
            images=image, return_tensors="pt", padding=padding, text=text
        ).to(DEVICE)
        if not text:
            return self.model.get_image_features(**processed_input).to(DEVICE)
        return self.model(**processed_input)

    async def encode(self, image: Image) -> torch.Tensor:
        return await asyncer.asyncify(self._encode)(image=image)

    async def compute_similarity(
        self, texts: list[str], input_image: Image | list[Image]
    ):
        model_output = await asyncer.asyncify(self._encode)(
            image=input_image,
            text=texts,
            padding=True,
        )
        return model_output.logits_per_text


class ScoreComputer:
    def __init__(self, score: SearchScoreFunctions) -> None:
        self.score_function = factories.make_score_function(score)

    def compute_score(self, corpus: Any, query: Any) -> Any:
        return self.score_function(corpus, query)
