import asyncio
import logging

from kafka import KafkaConsumer  # type: ignore[import-untyped]
import json

from vector_retrieval_service.consumers.event_processor import process_event

TOPIC = "text_services"
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    consumer = KafkaConsumer(
        "text_services",
        bootstrap_servers=["127.0.0.1:9093"],
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="latest",
    )

    for message in consumer:
        print(f"Message received, computing embeddings: {message.value}")
        embeddings = asyncio.run(process_event(message))
        print(embeddings[:5])
