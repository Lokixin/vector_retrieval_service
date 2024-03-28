from io import BytesIO
from typing import Any

from PIL import Image
from fastapi import APIRouter, UploadFile

from vector_retrieval_service.domain.factories import ImageModels
from vector_retrieval_service.services.dtos import TextToImagesSimilarity
from vector_retrieval_service.services.multimodal_services import (
    get_text_and_image_similitude_service,
    get_texts_and_image_similitude_service,
)

multimodal_router = APIRouter(
    prefix="/multimodal", tags=["Multimodal Processing Services"]
)


@multimodal_router.post("/texts_and_image_scores")
async def get_text_and_image_similitude(
    image: UploadFile,
    texts: list[str],
    requested_model: ImageModels,
) -> dict[str, Any]:
    image_bytes = BytesIO(await image.read())
    pillow_image = Image.open(image_bytes)
    res = await get_text_and_image_similitude_service(
        image=pillow_image, texts=texts[0].split(","), requested_model=requested_model
    )
    return res


@multimodal_router.post("/texts_and_images_scores")
async def get_texts_and_images_similitude(
    images: list[UploadFile],
    texts: list[str],
    requested_model: ImageModels,
) -> list[TextToImagesSimilarity]:
    image_names = [image.filename for image in images]
    pillow_images = [Image.open(BytesIO(await image.read())) for image in images]
    res = await get_texts_and_image_similitude_service(
        images=pillow_images,
        texts=texts[0].split(","),
        requested_model=requested_model,
        image_ids=image_names,
    )
    return res
