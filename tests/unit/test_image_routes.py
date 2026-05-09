from io import BytesIO
from collections.abc import Iterator

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from vector_retrieval_service.presenters.dependencies import get_image_model_interface
from vector_retrieval_service.presenters.fastapi_app import app


def make_png(mode: str = "RGB") -> bytes:
    image = Image.new(mode, (1, 1))
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture
def client() -> Iterator[TestClient]:
    app.dependency_overrides.clear()
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_extract_image_embeddings_validates_and_converts_to_rgb(
    client: TestClient,
) -> None:
    class FakeImageModelInterface:
        def __init__(self) -> None:
            self.image_mode: str | None = None

        async def encode(self, image: Image.Image) -> torch.Tensor:
            self.image_mode = image.mode
            return torch.tensor([[1.0, 2.0]])

    fake_model = FakeImageModelInterface()
    app.dependency_overrides[get_image_model_interface] = lambda: fake_model

    response = client.post(
        "/image_services/extract_image_embeddings",
        params={"requested_model": "clip_model"},
        files={"uploaded_image": ("image.png", make_png("L"), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["embeddings"] == [1.0, 2.0]
    assert fake_model.image_mode == "RGB"


@pytest.mark.parametrize(
    ("file_content", "content_type", "expected_status"),
    [
        (b"", "image/png", 400),
        (b"not an image", "image/png", 400),
        (make_png(), "text/plain", 415),
        (b"x" * (5 * 1024 * 1024 + 1), "image/png", 413),
    ],
    ids=["empty", "corrupt", "wrong-content-type", "too-large"],
)
def test_extract_image_embeddings_rejects_bad_uploads(
    client: TestClient,
    file_content: bytes,
    content_type: str,
    expected_status: int,
) -> None:
    app.dependency_overrides[get_image_model_interface] = lambda: None

    response = client.post(
        "/image_services/extract_image_embeddings",
        params={"requested_model": "clip_model"},
        files={"uploaded_image": ("image.png", file_content, content_type)},
    )

    assert response.status_code == expected_status


def test_multimodal_routes_use_shared_upload_validation(client: TestClient) -> None:
    app.dependency_overrides[get_image_model_interface] = lambda: None

    response = client.post(
        "/multimodal/texts_and_image_scores",
        params={"requested_model": "clip_model"},
        data={"texts": "cat,dog"},
        files={"image": ("bad.png", b"not an image", "image/png")},
    )

    assert response.status_code == 400


def test_multimodal_multi_image_route_uses_shared_upload_validation(
    client: TestClient,
) -> None:
    app.dependency_overrides[get_image_model_interface] = lambda: None

    response = client.post(
        "/multimodal/texts_and_images_scores",
        params={"requested_model": "clip_model"},
        data={"texts": "cat,dog"},
        files={"images": ("bad.png", b"not an image", "image/png")},
    )

    assert response.status_code == 400


def test_multimodal_route_accepts_repeated_text_form_fields(
    client: TestClient,
) -> None:
    class FakeImageModelInterface:
        async def compute_similarity(
            self, texts: list[str], input_image: Image.Image | list[Image.Image]
        ) -> torch.Tensor:
            assert texts == ["cat", "dog"]
            return torch.tensor([[0.1], [0.9]])

    app.dependency_overrides[get_image_model_interface] = lambda: FakeImageModelInterface()

    response = client.post(
        "/multimodal/texts_and_image_scores",
        params={"requested_model": "clip_model"},
        data={"texts": ["cat", "dog"]},
        files={"image": ("image.png", make_png(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {
        "similarities": [
            {"text": "dog", "score": 0.8999999761581421},
            {"text": "cat", "score": 0.10000000149011612},
        ]
    }


def test_multimodal_route_preserves_commas_inside_text(
    client: TestClient,
) -> None:
    class FakeImageModelInterface:
        async def compute_similarity(
            self, texts: list[str], input_image: Image.Image | list[Image.Image]
        ) -> torch.Tensor:
            assert texts == ["cat,dog"]
            return torch.tensor([[0.5]])

    app.dependency_overrides[get_image_model_interface] = lambda: FakeImageModelInterface()

    response = client.post(
        "/multimodal/texts_and_image_scores",
        params={"requested_model": "clip_model"},
        data={"texts": "cat,dog"},
        files={"image": ("image.png", make_png(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json() == {"similarities": [{"text": "cat,dog", "score": 0.5}]}


def test_multimodal_multi_image_route_accepts_repeated_text_form_fields(
    client: TestClient,
) -> None:
    class FakeImageModelInterface:
        async def compute_similarity(
            self, texts: list[str], input_image: Image.Image | list[Image.Image]
        ) -> torch.Tensor:
            assert texts == ["cat", "dog"]
            assert isinstance(input_image, list)
            return torch.tensor([[0.2, 0.8], [0.7, 0.3]])

    app.dependency_overrides[get_image_model_interface] = lambda: FakeImageModelInterface()

    response = client.post(
        "/multimodal/texts_and_images_scores",
        params={"requested_model": "clip_model"},
        data={"texts": ["cat", "dog"]},
        files=[
            ("images", ("first.png", make_png(), "image/png")),
            ("images", ("second.png", make_png(), "image/png")),
        ],
    )

    assert response.status_code == 200
    assert response.json() == [
        {
            "search_text": "cat",
            "scores": [
                {"image_identifier": "second.png", "score": 0.800000011920929},
                {"image_identifier": "first.png", "score": 0.20000000298023224},
            ],
        },
        {
            "search_text": "dog",
            "scores": [
                {"image_identifier": "first.png", "score": 0.699999988079071},
                {"image_identifier": "second.png", "score": 0.30000001192092896},
            ],
        },
    ]
