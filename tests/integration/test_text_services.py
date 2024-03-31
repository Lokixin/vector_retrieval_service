import pytest

from integration.fakes import FakeModelInterface
from vector_retrieval_service.domain.ai_models import ScoreComputer
from vector_retrieval_service.domain.factories import (
    LanguageModels,
    SearchScoreFunctions,
)
from vector_retrieval_service.services.dtos import (
    TextEmbeddingsResponse,
    TextSimilarityResponse,
    TextSimilarity,
)
from vector_retrieval_service.services.text_services import (
    get_text_embedding_service,
    get_text_similarity_service,
)


def compare_text_similarities(sim_1: TextSimilarity, sim_2: TextSimilarity) -> bool:
    are_equal = (
        sim_1.text == sim_2.text
        and sim_1.original_text_position == sim_2.original_text_position
        and sim_1.distance_to_query == pytest.approx(sim_2.distance_to_query, 0.01)
    )
    return are_equal


@pytest.mark.asyncio
async def test_get_text_embedding_service() -> None:
    # Test setup
    text = "Sample text to encode"
    requested_language_model = LanguageModels.MINI_LM
    expected_output = [[1.0, 2.0, 3.0]]
    llm_interface = FakeModelInterface(
        language_model=requested_language_model, output=expected_output
    )

    # Ejecución de lo que queremos testear
    result = await get_text_embedding_service(
        text=text,
        requested_model=requested_language_model,
        llm_interface=llm_interface,
    )

    # Comprobación del resultado
    expected_result = TextEmbeddingsResponse(
        embedding=expected_output[0],
        ai_model_used=LanguageModels.MINI_LM.value,
    )
    assert result == expected_result


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "score_function",
    [
        SearchScoreFunctions.COSINE_SIMILARITY,
        SearchScoreFunctions.EUCLIDEAN,
        SearchScoreFunctions.DOT_PRODUCT,
    ],
)
async def test_get_text_similarity_service(score_function) -> None:
    query = "Best Programming languages"
    texts = [
        "Python Java Javascript are the best languages for web dev",
        "In italy pasta is very good",
    ]
    language_model = LanguageModels.MINI_LM
    model_output = [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
        [1.0, 2.0, 3.0],
    ]
    llm_interface = FakeModelInterface(
        language_model=language_model, output=model_output
    )
    model_output = await llm_interface.encode(
        model_input=[], convert_to_tensor=True, convert_to_numpy=False
    )

    score_computer = ScoreComputer(score=score_function)

    result = await get_text_similarity_service(
        query=query,
        texts=texts,
        score_function=score_function,
        score_computer=score_computer,
        llm_interface=llm_interface,
    )

    expected_scores = score_computer.compute_score(
        query=model_output[-1], corpus=model_output[:-1]
    )
    text_similarities = [
        TextSimilarity(
            original_text_position=0,
            text=texts[0],
            distance_to_query=float(expected_scores[0]),
        ),
        TextSimilarity(
            original_text_position=1,
            text=texts[1],
            distance_to_query=float(expected_scores[1]),
        ),
    ]
    if score_function == SearchScoreFunctions.DOT_PRODUCT:
        text_similarities.reverse()

    expected_result = TextSimilarityResponse(
        original_query=query, text_similarities=text_similarities
    )

    assert result.original_query == expected_result.original_query
    assert all(
        compare_text_similarities(obtained_similarity, expected_similarity)
        for obtained_similarity, expected_similarity in zip(
            result.text_similarities, expected_result.text_similarities
        )
    )
