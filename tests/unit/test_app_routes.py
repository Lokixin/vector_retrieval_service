from fastapi.testclient import TestClient

from vector_retrieval_service.presenters.fastapi_app import app
from vector_retrieval_service.presenters.routers.images import image_router


def test_app_registers_expected_routes() -> None:
    client = TestClient(app)

    response = client.get("/openapi.json")

    assert response.status_code == 200
    paths = response.json()["paths"]
    assert "/healthcheck" in paths
    assert "/text_processing/get_embeddings" in paths
    assert "/text_processing/get_text_similarity" in paths
    assert "/image_services/extract_image_embeddings" in paths
    assert "/multimodal/texts_and_image_scores" in paths
    assert "/multimodal/texts_and_images_scores" in paths


def test_image_router_imports_from_correct_module() -> None:
    assert image_router.prefix == "/image_services"
