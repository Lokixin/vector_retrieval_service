from PIL import Image

from vector_retrieval_service.domain.ai_models import ImageModelInterface
from vector_retrieval_service.services.data_transformers import (
    transform_texts_to_image_similarity_to_response,
    transform_texts_to_images_similarities_to_response,
)
from vector_retrieval_service.services.dtos import (
    TextToImagesSimilarity,
    TextsToImageSimilarities,
)


async def get_text_and_image_similitude_service(
    image: Image.Image,
    texts: list[str],
    model_interface: ImageModelInterface,
) -> TextsToImageSimilarities:
    logits_per_text = await model_interface.compute_similarity(
        texts=texts, input_image=image
    )
    texts_to_image_similarities = transform_texts_to_image_similarity_to_response(
        logits_per_text=logits_per_text, texts=texts
    )
    return texts_to_image_similarities


async def get_texts_and_image_similitude_service(
    images: list[Image.Image],
    texts: list[str],
    image_ids: list[str | None],
    model_interface: ImageModelInterface,
) -> list[TextToImagesSimilarity]:
    logits_per_text = await model_interface.compute_similarity(
        texts=texts, input_image=images
    )
    return transform_texts_to_images_similarities_to_response(
        texts=texts,
        image_ids=image_ids,
        logits=logits_per_text,
    )
