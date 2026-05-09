from kafka import KafkaProducer  # type: ignore[import-untyped]
import json

from vector_retrieval_service.config import SETTINGS
from vector_retrieval_service.domain.factories import LanguageModels
from vector_retrieval_service.producers.event_creator import make_text_to_encode_event

if __name__ == "__main__":
    producer = KafkaProducer(
        bootstrap_servers=SETTINGS.kafka_bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )
    text_to_encode = "Esto es un texto del cual queremos obtener embeddings"
    event = make_text_to_encode_event(
        text_to_encode=text_to_encode,
        requested_model=LanguageModels.MINI_LM,
    )
    future = producer.send(
        SETTINGS.kafka_topic,
        event.model_dump(by_alias=True),
        headers=[("event_type", "text".encode("utf-8"))],
    )
    result = future.get()
    print(result)
    producer.close()
