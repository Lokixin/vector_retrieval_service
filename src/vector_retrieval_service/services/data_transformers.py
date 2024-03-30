from typing import Any

import numpy as np
import torch

from vector_retrieval_service.domain.factories import LanguageModels
from vector_retrieval_service.services.dtos import (
    TextSimilarityResponse,
    TextSimilarity,
    TextEmbeddingsResponse,
    ImageEmbeddingsResponse,
    TextsToImageSimilarities,
    TextsToImageSimilarity,
    TextToImageScore,
    TextToImagesSimilarity,
)


def transform_distances_to_similarity_response(
    original_query: str,
    corpus: list[str],
    distances: Any,
    sort_reverse: bool,
) -> TextSimilarityResponse:
    text_similarities = [
        TextSimilarity(
            distance_to_query=float(distances[idx]),
            text=text,
            original_text_position=idx,
        )
        for idx, text in enumerate(corpus)
    ]

    sorted_similarities = sorted(
        text_similarities, key=lambda item: item.distance_to_query, reverse=sort_reverse
    )

    similarities = TextSimilarityResponse(
        original_query=original_query, text_similarities=sorted_similarities
    )

    return similarities


def transform_embeddings_to_response(
    embeddings: list[np.ndarray], model: LanguageModels
) -> TextEmbeddingsResponse:
    return TextEmbeddingsResponse(embedding=embeddings[0].tolist(), ai_model_used=model)


def transform_image_embeddings_to_response(
    embeddings: torch.Tensor,
) -> ImageEmbeddingsResponse:
    embedding_list = embeddings.cpu().detach().tolist()[0]
    return ImageEmbeddingsResponse(embeddings=embedding_list)


def transform_texts_to_image_similarity_to_response(
    logits_per_text: torch.Tensor, texts: list[str]
) -> TextsToImageSimilarities:
    logits_per_text = logits_per_text.cpu().detach().tolist()
    score_by_text = [
        TextsToImageSimilarity(text=input_text, score=score[0])
        for input_text, score in zip(texts, logits_per_text)
    ]
    sorted_results = sorted(score_by_text, key=lambda item: item.score, reverse=True)
    return TextsToImageSimilarities(similarities=sorted_results)


def transform_texts_to_images_similarities_to_response(
    texts: list[str], image_ids: list[str], logits: torch.Tensor
) -> list[TextToImagesSimilarity]:
    scores = []
    logits = logits.cpu().detach().tolist()
    for idx, text_to_images_scores in enumerate(logits):
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
