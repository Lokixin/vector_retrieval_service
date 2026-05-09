import os
from dataclasses import dataclass

import torch


def _default_device() -> str:
    return "cuda:0" if torch.cuda.is_available() else "cpu"


@dataclass(frozen=True)
class AppSettings:
    device: str
    kafka_topic: str
    kafka_bootstrap_servers: list[str]
    kafka_group_id: str


def load_settings() -> AppSettings:
    bootstrap_servers = os.getenv(
        "KAFKA_BOOTSTRAP_SERVERS", "127.0.0.1:9093"
    ).split(",")
    return AppSettings(
        device=os.getenv("VECTOR_RETRIEVAL_DEVICE", _default_device()),
        kafka_topic=os.getenv("KAFKA_TOPIC", "text_services"),
        kafka_bootstrap_servers=[server.strip() for server in bootstrap_servers],
        kafka_group_id=os.getenv("KAFKA_GROUP_ID", "vector_retrieval_service"),
    )


SETTINGS = load_settings()
DEVICE = SETTINGS.device
