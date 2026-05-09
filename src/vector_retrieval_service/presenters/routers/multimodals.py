from fastapi import APIRouter, UploadFile, Depends, Form, HTTPException, status

from vector_retrieval_service.domain.ai_models import ImageModelInterface
from vector_retrieval_service.domain.factories import ImageModels
from vector_retrieval_service.presenters.dependencies import get_image_model_interface
from vector_retrieval_service.presenters.image_uploads import read_image_upload
from vector_retrieval_service.services.dtos import (
    TextToImagesSimilarity,
    TextsToImageSimilarities,
)
from vector_retrieval_service.services.multimodal_services import (
    get_text_and_image_similitude_service,
    get_texts_and_image_similitude_service,
)

multimodal_router = APIRouter(
    prefix="/multimodal", tags=["Multimodal Processing Services"]
)


def _validate_texts(texts: list[str]) -> list[str]:
    clean_texts = [text.strip() for text in texts if text.strip()]
    if not clean_texts:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one non-empty text is required",
        )
    return clean_texts


@multimodal_router.post("/texts_and_image_scores")
async def get_text_and_image_similitude(
    image: UploadFile,
    requested_model: ImageModels,
    texts: list[str] = Form(...),
    model_interface: ImageModelInterface = Depends(get_image_model_interface),
) -> TextsToImageSimilarities:
    texts = _validate_texts(texts)
    pillow_image = await read_image_upload(image)
    res = await get_text_and_image_similitude_service(
        image=pillow_image, texts=texts, model_interface=model_interface
    )
    return res


@multimodal_router.post("/texts_and_images_scores")
async def get_texts_and_images_similitude(
    images: list[UploadFile],
    requested_model: ImageModels,
    texts: list[str] = Form(...),
    model_interface: ImageModelInterface = Depends(get_image_model_interface),
) -> list[TextToImagesSimilarity]:
    texts = _validate_texts(texts)
    image_names = [image.filename for image in images]
    pillow_images = [await read_image_upload(image) for image in images]
    res = await get_texts_and_image_similitude_service(
        images=pillow_images,
        texts=texts,
        image_ids=image_names,
        model_interface=model_interface,
    )
    return res
