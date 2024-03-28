from abc import ABC, abstractmethod

import asyncer

from vector_retrieval_service.domain import factories


class AbstractLLMModelAdapter(ABC):
    def __init__(self, model_type: str):
        self.model = factories.make_llm(model_type)

    @abstractmethod
    async def encode_text(self, text: str) -> list[float]:
        ...

    @abstractmethod
    async def encode_many(self, texts: list[str]) -> list[list[float]]:
        ...


class LLMModelAdapter(AbstractLLMModelAdapter):
    async def encode_text(self, text: str) -> list[float]:
        embeddings = await asyncer.asyncify(self.model.encode)(text)
        return embeddings.tolist()

    async def encode_many(self, texts: list[str]) -> list[list[float]]:
        return await asyncer.asyncify(self.model.encode)(texts)


class FakeLLMModelAdapter(AbstractLLMModelAdapter):
    def __init__(self, model_type: str, hardcoded_embeddings: list[list[float]]):
        super().__init__(model_type)
        self.hardcoded_embeddings = hardcoded_embeddings or [[1.0, 2.0], [3.0, 4.0]]

    async def encode_text(self, text: str) -> list[float]:
        return self.hardcoded_embeddings[0]

    async def encode_many(self, texts: list[str]) -> list[list[float]]:
        return self.hardcoded_embeddings
