from types import SimpleNamespace
from typing import Any

import pytest

from vector_retrieval_service.consumers import event_processor
from vector_retrieval_service.domain.factories import LanguageModels


def test_decode_headers_returns_event_type() -> None:
    headers = [("ignored", b"value"), ("event_type", b"text")]

    assert event_processor._decode_headers(headers) == "text"


def test_decode_headers_raises_when_missing_event_type() -> None:
    with pytest.raises(event_processor.HeaderException):
        event_processor._decode_headers([("ignored", b"value")])


@pytest.mark.asyncio
async def test_process_event_rejects_unknown_event_type() -> None:
    event: Any = SimpleNamespace(headers=[("event_type", b"image")], value={})

    with pytest.raises(event_processor.UnprocessableEvent):
        await event_processor.process_event(event)


@pytest.mark.asyncio
async def test_process_event_handles_valid_text_event(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeLanguageModelInterface:
        def __init__(self, language_model: LanguageModels) -> None:
            assert language_model == LanguageModels.MINI_LM

        async def encode(self, model_input: str) -> list[float]:
            assert model_input == "hello"
            return [1.0, 2.0]

    monkeypatch.setattr(
        event_processor, "LanguageModelInterface", FakeLanguageModelInterface
    )
    event: Any = SimpleNamespace(
        headers=[("event_type", b"text")],
        value={
            "requested_model": LanguageModels.MINI_LM.value,
            "text_to_encode": "hello",
        },
    )

    assert await event_processor.process_event(event) == [1.0, 2.0]


@pytest.mark.asyncio
async def test_process_text_event_rejects_invalid_model_without_loading_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_on_model_load(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("model should not load")

    monkeypatch.setattr(event_processor, "LanguageModelInterface", fail_on_model_load)

    with pytest.raises(ValueError):
        await event_processor.process_text_event(
            {"requested_model": "unknown", "text_to_encode": "hello"}
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "event_value",
    [{"requested_model": LanguageModels.MINI_LM.value}, {"text_to_encode": "hello"}],
)
async def test_process_text_event_rejects_missing_keys_without_loading_model(
    event_value: dict[str, str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_on_model_load(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("model should not load")

    monkeypatch.setattr(event_processor, "LanguageModelInterface", fail_on_model_load)

    with pytest.raises(KeyError):
        await event_processor.process_text_event(event_value)
