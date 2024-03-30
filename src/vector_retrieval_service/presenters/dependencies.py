from vector_retrieval_service.domain.ai_models import (
    LanguageModelInterface,
    ScoreComputer,
    ImageModelInterface,
)
from vector_retrieval_service.domain.factories import (
    LanguageModels,
    SearchScoreFunctions,
    ImageModels,
)


def get_language_model_interface(
    requested_model: LanguageModels,
) -> LanguageModelInterface:
    return LanguageModelInterface(language_model=requested_model)


def get_score_computer(score_function: SearchScoreFunctions) -> ScoreComputer:
    return ScoreComputer(score=score_function)


def get_image_model_interface(requested_model: ImageModels) -> ImageModelInterface:
    return ImageModelInterface(image_model=requested_model)
