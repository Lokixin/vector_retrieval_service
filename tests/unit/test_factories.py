import pytest

from vector_retrieval_service.domain import factories
from vector_retrieval_service.domain.factories import (
    ImageModels,
    LanguageModels,
    SearchScoreFunctions,
)


def test_make_llm_rejects_unknown_model_without_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_on_model_load(*args, **kwargs):
        raise AssertionError("model should not load")

    factories.make_llm.cache_clear()
    monkeypatch.setattr(factories, "SentenceTransformer", fail_on_model_load)

    with pytest.raises(ValueError, match="Unknown language model"):
        factories.make_llm("unknown")


def test_make_score_function_rejects_unknown_score_function() -> None:
    with pytest.raises(ValueError, match="Unknown score function"):
        factories.make_score_function("unknown")


def test_make_image_model_rejects_unknown_model_without_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_on_model_load(*args, **kwargs):
        raise AssertionError("model should not load")

    factories.make_image_model.cache_clear()
    monkeypatch.setattr(factories, "CLIPModel", fail_on_model_load)

    with pytest.raises(ValueError, match="Unknown image model"):
        factories.make_image_model("unknown")


def test_factories_accept_valid_string_values(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str) -> None:
            self.model_name = model_name
            self.device = device

    factories.make_llm.cache_clear()
    monkeypatch.setattr(factories, "SentenceTransformer", FakeSentenceTransformer)

    model = factories.make_llm(LanguageModels.MINI_LM.value)

    assert model.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    assert factories.make_score_function(SearchScoreFunctions.DOT_PRODUCT.value) is not None
    assert ImageModels.CLIP_MODEL.value == "clip_model"
