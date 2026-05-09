# AGENTS.md

## Commands
- Install with `poetry install`; dependencies are locked in `poetry.lock` and CUDA Torch wheels come from the explicit `pytorch` source in `pyproject.toml`.
- Run the API with `make api` or `poetry run uvicorn vector_retrieval_service.presenters.fastapi_app:app --reload`.
- Run lint/type checks with `make lint`; it runs `ruff check src tests` and `mypy src tests`.
- Format with `make reformat`; it only runs `ruff format src tests`.
- Run integration tests with `make integration` or `poetry run pytest ./tests/integration --disable-warnings`.
- Focus a single test with `poetry run pytest tests/integration/test_text_services.py::test_get_text_embedding_service --disable-warnings`.

## Architecture
- Package source is under `src/vector_retrieval_service`; Poetry uses the `src` layout.
- FastAPI app entrypoint is `vector_retrieval_service.presenters.fastapi_app:app`.
- Routers live in `presenters/routers`; the image router filename is intentionally misspelled as `iamges.py` and is imported that way by `fastapi_app.py`.
- `services/` contains async use-case functions; `presenters/dependencies.py` constructs model interfaces for FastAPI dependency injection.
- Model factories in `domain/factories.py` are cached with `lru_cache` and load Hugging Face/SentenceTransformer/CLIP models.
- Kafka producer/consumer scripts use topic `text_services` and connect to `127.0.0.1:9093`; `docker-compose.yaml` provides Kafka and Zookeeper with the external listener on port `9093`.

## Test And Runtime Gotchas
- `config.DEVICE` falls back to CPU, but `domain/factories.make_llm` and `tests/integration/conftest.py` hard-code `cuda:0`; do not assume the whole repo is CPU-safe.
- The current integration fake `FakeModelInterface` calls `LanguageModelInterface.__init__`, so tests can still instantiate/download a real SentenceTransformer model before using fake outputs.
- There are no unit tests beyond package placeholders; current executable tests are under `tests/integration`.
