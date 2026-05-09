import torch
from PIL import Image

from vector_retrieval_service.domain import factories
from vector_retrieval_service.domain.ai_models import ImageModelInterface
from vector_retrieval_service.domain.factories import ImageModels, LanguageModels


def test_make_llm_uses_configured_device(monkeypatch) -> None:
    calls = []

    class FakeSentenceTransformer:
        def __init__(self, model_name: str, device: str) -> None:
            calls.append((model_name, device))

    factories.make_llm.cache_clear()
    monkeypatch.setattr(factories, "SentenceTransformer", FakeSentenceTransformer)

    factories.make_llm(LanguageModels.MINI_LM)

    assert calls == [("sentence-transformers/all-MiniLM-L6-v2", factories.DEVICE)]


def test_make_image_model_sets_eval(monkeypatch) -> None:
    class FakeClipModel:
        eval_called = False

        @classmethod
        def from_pretrained(cls, model_name: str) -> "FakeClipModel":
            return cls()

        def to(self, device: str) -> "FakeClipModel":
            return self

        def eval(self) -> None:
            self.eval_called = True

    class FakeClipProcessor:
        @classmethod
        def from_pretrained(cls, model_name: str) -> "FakeClipProcessor":
            return cls()

    factories.make_image_model.cache_clear()
    monkeypatch.setattr(factories, "CLIPModel", FakeClipModel)
    monkeypatch.setattr(factories, "CLIPProcessor", FakeClipProcessor)

    _, model = factories.make_image_model(ImageModels.CLIP_MODEL)

    assert model.eval_called is True


def test_image_encode_runs_in_inference_mode(monkeypatch) -> None:
    class ProcessedInput(dict):
        def to(self, device: str) -> "ProcessedInput":
            return self

    class FakeProcessor:
        def __call__(self, **kwargs) -> ProcessedInput:
            return ProcessedInput()

    class FakeModel:
        def __init__(self) -> None:
            self.used_inference_mode = False

        def get_image_features(self, **kwargs) -> torch.Tensor:
            self.used_inference_mode = torch.is_inference_mode_enabled()
            return torch.tensor([[1.0]])

    fake_model = FakeModel()
    monkeypatch.setattr(
        factories, "make_image_model", lambda model_type: (FakeProcessor(), fake_model)
    )

    model_interface = ImageModelInterface(ImageModels.CLIP_MODEL)

    model_interface._encode(image=Image.new("RGB", (1, 1)))

    assert fake_model.used_inference_mode is True
