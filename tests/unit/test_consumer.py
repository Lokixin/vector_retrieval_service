from types import SimpleNamespace
from typing import Any

import pytest

from vector_retrieval_service.config import AppSettings
from vector_retrieval_service.consumers import consumer


class FakeConsumer:
    def __init__(self, messages: list[Any]) -> None:
        self.messages = messages
        self.commit_count = 0
        self.closed = False

    def __iter__(self):
        return iter(self.messages)

    def commit(self) -> None:
        self.commit_count += 1

    def close(self) -> None:
        self.closed = True


def test_run_consumer_commits_successful_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_handle_message(message: Any) -> list[float]:
        assert message.value == {"text_to_encode": "hello"}
        return [1.0]

    fake_consumer = FakeConsumer(
        [SimpleNamespace(value={"text_to_encode": "hello"}, offset=1)]
    )
    monkeypatch.setattr(consumer, "handle_message", fake_handle_message)

    consumer.run_consumer(fake_consumer)

    assert fake_consumer.commit_count == 1
    assert fake_consumer.closed is True


def test_run_consumer_logs_and_commits_failed_messages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def fake_handle_message(message: Any) -> list[float]:
        raise ValueError("bad message")

    fake_consumer = FakeConsumer([SimpleNamespace(value={}, offset=1)])
    monkeypatch.setattr(consumer, "handle_message", fake_handle_message)

    consumer.run_consumer(fake_consumer)

    assert fake_consumer.commit_count == 1
    assert fake_consumer.closed is True


def test_make_consumer_uses_configured_kafka_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    class FakeKafkaConsumer:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            calls.append((args, kwargs))

    monkeypatch.setattr(consumer, "KafkaConsumer", FakeKafkaConsumer)
    settings = AppSettings(
        device="cpu",
        kafka_topic="custom-topic",
        kafka_bootstrap_servers=["localhost:9093"],
        kafka_group_id="custom-group",
    )

    consumer.make_consumer(settings)

    args, kwargs = calls[0]
    assert args == ("custom-topic",)
    assert kwargs["bootstrap_servers"] == ["localhost:9093"]
    assert kwargs["group_id"] == "custom-group"
    assert kwargs["enable_auto_commit"] is False
