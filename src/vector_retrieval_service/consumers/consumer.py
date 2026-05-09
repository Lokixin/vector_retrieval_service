import asyncio
import json
import logging
from collections.abc import Iterable
from typing import Any

from kafka import KafkaConsumer  # type: ignore[import-untyped]

from vector_retrieval_service.config import SETTINGS, AppSettings
from vector_retrieval_service.consumers.event_processor import process_event

logger = logging.getLogger(__name__)


def make_consumer(settings: AppSettings = SETTINGS) -> KafkaConsumer:
    return KafkaConsumer(
        settings.kafka_topic,
        bootstrap_servers=settings.kafka_bootstrap_servers,
        group_id=settings.kafka_group_id,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
        enable_auto_commit=False,
    )


async def handle_message(message: Any) -> Any:
    logger.info(
        "Processing Kafka message",
        extra={
            "topic": getattr(message, "topic", SETTINGS.kafka_topic),
            "partition": getattr(message, "partition", None),
            "offset": getattr(message, "offset", None),
        },
    )
    embeddings = await process_event(message)
    logger.info(
        "Kafka message processed",
        extra={
            "topic": getattr(message, "topic", SETTINGS.kafka_topic),
            "partition": getattr(message, "partition", None),
            "offset": getattr(message, "offset", None),
        },
    )
    return embeddings


def run_consumer(consumer: Iterable[Any] | None = None) -> None:
    kafka_consumer = consumer or make_consumer()
    try:
        for message in kafka_consumer:
            try:
                asyncio.run(handle_message(message))
            except Exception:
                logger.exception(
                    "Skipping failed Kafka message",
                    extra={
                        "topic": getattr(message, "topic", SETTINGS.kafka_topic),
                        "partition": getattr(message, "partition", None),
                        "offset": getattr(message, "offset", None),
                    },
                )
            finally:
                commit = getattr(kafka_consumer, "commit", None)
                if commit:
                    commit()
    except KeyboardInterrupt:
        logger.info("Kafka consumer stopped")
    finally:
        close = getattr(kafka_consumer, "close", None)
        if close:
            close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_consumer()
