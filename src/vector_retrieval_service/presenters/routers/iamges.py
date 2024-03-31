import logging
from io import BytesIO

from fastapi import APIRouter, UploadFile, Depends
from PIL import Image

from vector_retrieval_service.domain.ai_models import ImageModelInterface
from vector_retrieval_service.domain.factories import ImageModels
from vector_retrieval_service.presenters.dependencies import get_image_model_interface
from vector_retrieval_service.services.dtos import ImageEmbeddingsResponse
from vector_retrieval_service.services.image_services import (
    get_image_embeddings_services,
)

logger = logging.getLogger(__name__)
image_router = APIRouter(prefix="/image_services", tags=["Image Services"])


@image_router.post("/extract_image_embeddings")
async def extract_image_embeddings(
    uploaded_image: UploadFile,
    requested_model: ImageModels,
    model_interface: ImageModelInterface = Depends(get_image_model_interface),
) -> ImageEmbeddingsResponse:
    image_bytes = BytesIO(await uploaded_image.read())
    pillow_image = Image.open(image_bytes)
    result = await get_image_embeddings_services(
        image=pillow_image, model_interface=model_interface
    )
    return result
