from typing import Any

from vector_retrieval_service.services.dtos import TextSimilarityResponse, TextSimilarity


def transform_distances_to_similarity_response(original_query: str, corpus: list[str], distances: dict[str, Any], sort_reverse: bool) -> TextSimilarityResponse:
    sorted_distances = dict(
        sorted(distances.items(), key=lambda item: item[1], reverse=sort_reverse)
    )
    similarities = TextSimilarityResponse(
        original_query=original_query,
        text_similarities=[
            TextSimilarity(
                distance_to_query=float(distance),
                text=text,
                original_text_position=corpus.index(text),
            )
            for text, distance in sorted_distances.items()
        ],
    )
    return similarities
