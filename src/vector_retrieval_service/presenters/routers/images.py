import logging

from fastapi import APIRouter, UploadFile, Depends

from vector_retrieval_service.domain.ai_models import ImageModelInterface
from vector_retrieval_service.domain.factories import ImageModels
from vector_retrieval_service.presenters.dependencies import get_image_model_interface
from vector_retrieval_service.presenters.image_uploads import read_image_upload
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
    pillow_image = await read_image_upload(uploaded_image)
    result = await get_image_embeddings_services(
        image=pillow_image, model_interface=model_interface
    )
    return result
