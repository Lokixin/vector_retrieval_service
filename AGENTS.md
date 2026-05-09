# AGENTS.md

## Commands
- Install with `poetry install`; dependencies are locked in `poetry.lock` and CUDA Torch wheels come from the explicit `pytorch` source in `pyproject.toml`.
- Run the API with `make api` or `poetry run uvicorn vector_retrieval_service.presenters.fastapi_app:app --reload`.
- Run lint/type checks with `make lint`; it runs `ruff check src tests` and `mypy src tests`.
- Format with `make reformat`; it only runs `ruff format src tests`.
- Run integration tests with `make integration` or `poetry run pytest ./tests/integration --disable-warnings`.
- Run coverage locally with `poetry run pytest --disable-warnings --cov=vector_retrieval_service --cov-report=term-missing`.
- Focus a single test with `poetry run pytest tests/integration/test_text_services.py::test_get_text_embedding_service --disable-warnings`.
- Run optional Kafka Docker coverage with `RUN_KAFKA_DOCKER_TESTS=1 poetry run pytest tests/integration/test_kafka_docker.py --disable-warnings` after `docker compose up -d`.
- Start local Kafka/Zookeeper with `docker compose up -d`; the compose file intentionally exposes Kafka on `localhost:9093`.

## Architecture
- Package source is under `src/vector_retrieval_service`; Poetry uses the `src` layout.
- FastAPI app entrypoint is `vector_retrieval_service.presenters.fastapi_app:app`.
- Routers live in `presenters/routers`; image routes are in `images.py` and are imported by `fastapi_app.py`.
- `services/` contains async use-case functions; `presenters/dependencies.py` constructs model interfaces for FastAPI dependency injection.
- Model factories in `domain/factories.py` are cached with `lru_cache` and load Hugging Face/SentenceTransformer/CLIP models.
- `config.AppSettings` centralizes `VECTOR_RETRIEVAL_DEVICE`, `KAFKA_TOPIC`, `KAFKA_BOOTSTRAP_SERVERS`, and `KAFKA_GROUP_ID`; producer/consumer scripts read from `config.SETTINGS`.
- CI lives in `.github/workflows/ci.yaml` and runs `make lint` plus `poetry run pytest --disable-warnings` on Python 3.11.

## Test And Runtime Gotchas
- `config.DEVICE` controls model device selection; keep tests using fakes/overrides so they do not download Hugging Face models.
- Image/multimodal route tests override FastAPI dependencies; preserve that pattern when adding route coverage.
- Multimodal upload routes expect repeated `texts` form fields, not comma-splitting.
