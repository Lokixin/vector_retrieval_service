ALL_PACKAGES := src tests

.PHONY: reformat lint api

lint:
	poetry run ruff check $(ALL_PACKAGES) &
	poetry run mypy .

reformat:
	poetry run ruff format $(ALL_PACKAGES)

api:
	poetry run uvicorn vector_retrieval_service.service_api.fastapi_app:app --reload