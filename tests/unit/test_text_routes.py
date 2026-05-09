from collections.abc import Iterator
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from vector_retrieval_service.presenters.dependencies import (
    get_language_model_interface,
    get_score_computer,
)
from vector_retrieval_service.presenters.fastapi_app import app


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides.clear()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_healthcheck(client: TestClient) -> None:
    response = client.get("/healthcheck")

    assert response.status_code == 200
    assert response.json() == {"message": "The app is up and running!"}


def test_root_redirects_to_docs(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 302
    assert response.headers["location"] == "docs"


def test_text_embeddings_route_uses_dependency_override(client: TestClient) -> None:
    class FakeLanguageModel:
        async def encode(
            self,
            model_input: list[str],
            convert_to_tensor: bool = False,
            convert_to_numpy: bool = True,
            normalize_embeddings: bool = False,
        ) -> Any:
            assert model_input == ["hello"]
            return torch.tensor([[1.0, 2.0]])

    app.dependency_overrides[get_language_model_interface] = lambda: FakeLanguageModel()

    response = client.get(
        "/text_processing/get_embeddings",
        params={"text": "hello", "requested_model": "mini_lm"},
    )

    assert response.status_code == 200
    assert response.json() == {"embedding": [1.0, 2.0], "ai_model_used": "mini_lm"}


def test_text_similarity_route_uses_dependency_overrides(client: TestClient) -> None:
    class FakeLanguageModel:
        async def encode(
            self,
            model_input: list[str],
            convert_to_tensor: bool = False,
            convert_to_numpy: bool = True,
            normalize_embeddings: bool = False,
        ) -> Any:
            assert model_input == ["one", "two", "query"]
            assert convert_to_tensor is True
            assert convert_to_numpy is False
            return torch.tensor([[1.0], [2.0], [3.0]])

    class FakeScoreComputer:
        def compute_score(self, corpus: Any, query: Any) -> torch.Tensor:
            assert corpus.shape == (2, 1)
            assert query.shape == (1,)
            return torch.tensor([0.2, 0.1])

    app.dependency_overrides[get_language_model_interface] = lambda: FakeLanguageModel()
    app.dependency_overrides[get_score_computer] = lambda: FakeScoreComputer()

    response = client.post(
        "/text_processing/get_text_similarity",
        params={
            "requested_model": "mini_lm",
            "score_function": "euclidean",
            "query": "query",
        },
        json=["one", "two"],
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["original_query"] == "query"
    assert [item["text"] for item in payload["text_similarities"]] == ["two", "one"]
    assert [
        item["original_text_position"] for item in payload["text_similarities"]
    ] == [1, 0]
    assert payload["text_similarities"][0]["distance_to_query"] == pytest.approx(0.1)
    assert payload["text_similarities"][1]["distance_to_query"] == pytest.approx(0.2)
