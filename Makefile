ALL_PACKAGES := src tests

.PHONY: reformat lint api

lint:
	poetry run ruff check $(ALL_PACKAGES) &
	poetry run mypy $(ALL_PACKAGES)

reformat:
	poetry run ruff format $(ALL_PACKAGES)

api:
	poetry run uvicorn vector_retrieval_service.presenters.fastapi_app:app --reload

integration:
	poetry run pytest ./tests/integration --disable-warnings
