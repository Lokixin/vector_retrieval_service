from PIL import Image

from vector_retrieval_service.domain.ai_models import ImageModelInterface
from vector_retrieval_service.services.data_transformers import (
    transform_image_embeddings_to_response,
)
from vector_retrieval_service.services.dtos import ImageEmbeddingsResponse


async def get_image_embeddings_services(
    image: Image.Image, model_interface: ImageModelInterface
) -> ImageEmbeddingsResponse:
    image_embedding = await model_interface.encode(image=image)
    embedding_response = transform_image_embeddings_to_response(
        embeddings=image_embedding
    )
    return embedding_response
