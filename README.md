# Vector Retrieval Service

FastAPI service for text embeddings, text similarity, image embeddings, and CLIP-based text/image similarity. Runtime code lives under `src/vector_retrieval_service`.

## Setup
- Use Python 3.11+ and Poetry.
- Install dependencies with `poetry install`.
- Torch and Torchvision are pinned to CUDA 12.1 wheels from the explicit `pytorch` source in `pyproject.toml`.
- First real model use can download Hugging Face/SentenceTransformer/CLIP weights; tests use fakes and dependency overrides to avoid this.

## Run
- Start the API with `make api`.
- Equivalent command: `poetry run uvicorn vector_retrieval_service.presenters.fastapi_app:app --reload`.
- OpenAPI docs are available at `/docs`; `/` redirects there.
- Healthcheck: `GET /healthcheck`.

## Verify
- Run lint and type checks with `make lint`.
- Run all tests with `poetry run pytest --disable-warnings`.
- Run integration tests with `make integration`.
- Focus one test with `poetry run pytest tests/integration/test_text_services.py::test_get_text_embedding_service --disable-warnings`.

## Kafka
- Local Kafka/Zookeeper are defined in `docker-compose.yaml`.
- Start them with `docker compose up -d`.
- Producer and consumer scripts use topic `text_services` and default to `127.0.0.1:9093`.
- Consumer settings can be overridden with `KAFKA_TOPIC`, `KAFKA_BOOTSTRAP_SERVERS`, and `KAFKA_GROUP_ID`.

## API Notes
- Multimodal upload routes expect repeated `texts` form fields, not comma-separated text splitting.
- Image uploads are validated for image content type, non-empty body, max size, and valid image bytes, then converted to RGB.
