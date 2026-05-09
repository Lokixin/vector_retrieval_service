import json
import os
import time

import pytest
from kafka import KafkaConsumer, KafkaProducer  # type: ignore[import-untyped]

from vector_retrieval_service.config import SETTINGS
from vector_retrieval_service.domain.factories import LanguageModels
from vector_retrieval_service.producers.event_creator import make_text_to_encode_event


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_KAFKA_DOCKER_TESTS") != "1",
    reason="Set RUN_KAFKA_DOCKER_TESTS=1 and start docker compose Kafka first",
)


def test_kafka_docker_round_trips_text_event() -> None:
    producer = KafkaProducer(
        bootstrap_servers=SETTINGS.kafka_bootstrap_servers,
        value_serializer=lambda value: json.dumps(value).encode("utf-8"),
    )
    consumer = KafkaConsumer(
        SETTINGS.kafka_topic,
        bootstrap_servers=SETTINGS.kafka_bootstrap_servers,
        auto_offset_reset="latest",
        consumer_timeout_ms=5000,
        group_id=f"vector-retrieval-service-test-{time.time_ns()}",
        value_deserializer=lambda value: json.loads(value.decode("utf-8")),
    )

    try:
        event = make_text_to_encode_event(
            text_to_encode="hello", requested_model=LanguageModels.MINI_LM
        )
        producer.send(
            SETTINGS.kafka_topic,
            event.model_dump(by_alias=True),
            headers=[("event_type", b"text")],
        ).get(timeout=10)
        producer.flush(timeout=10)

        for message in consumer:
            if message.value.get("_id") == event.event_id:
                assert message.value["text_to_encode"] == "hello"
                assert message.value["requested_model"] == LanguageModels.MINI_LM.value
                return
    finally:
        producer.close()
        consumer.close()

    pytest.fail("Kafka text event was not consumed")
