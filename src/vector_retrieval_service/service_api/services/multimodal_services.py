from typing import Any

import asyncer
from PIL import Image

from vector_retrieval_service.embedding_retriever.ai_models.model_factory import (
    ImageModels,
    LLMFactory,
    DEVICE,
)
from vector_retrieval_service.service_api.services.dtos import (
    TextToImagesSimilarity,
    TextToImageScore,
)


async def get_text_and_image_similitude_service(
    image: Image, texts: list[str], requested_model: ImageModels
) -> dict[str, Any]:
    clip_processor, clip_model = LLMFactory.get_image_model(requested_model)
    input_to_model = await asyncer.asyncify(clip_processor)(
        text=texts,
        images=image,
        return_tensors="pt",
        padding=True,
    )
    input_to_model.to(DEVICE)
    model_output = await asyncer.asyncify(clip_model)(**input_to_model)
    logits_per_text = model_output.logits_per_text.detach().cpu().numpy().tolist()
    score_by_text = {
        input_text: score[0] for input_text, score in zip(texts, logits_per_text)
    }
    sorted_results = dict(
        sorted(score_by_text.items(), key=lambda item: item[1], reverse=True)
    )
    return sorted_results


async def get_texts_and_image_similitude_service(
    images: Image, texts: list[str], requested_model: ImageModels, image_ids: list[str]
) -> list[TextToImagesSimilarity]:
    clip_processor, clip_model = LLMFactory.get_image_model(requested_model)
    input_to_model = await asyncer.asyncify(clip_processor)(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    )
    input_to_model.to(DEVICE)
    model_output = await asyncer.asyncify(clip_model)(**input_to_model)
    logits_per_text = model_output.logits_per_text.detach().cpu().numpy().tolist()
    scores = []
    for idx, text_to_images_scores in enumerate(logits_per_text):
        score_by_text = {
            image_id: score for image_id, score in zip(image_ids, text_to_images_scores)
        }
        sorted_results = dict(
            sorted(score_by_text.items(), key=lambda item: item[1], reverse=True)
        )
        text_to_images_similarity = TextToImagesSimilarity(
            search_text=texts[idx],
            scores=[
                TextToImageScore(image_identifier=identifier, score=score)
                for identifier, score in sorted_results.items()
            ],
        )
        scores.append(text_to_images_similarity)
    return scores
