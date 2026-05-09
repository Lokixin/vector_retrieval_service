import typing
from enum import StrEnum, auto
from functools import lru_cache

from sentence_transformers import SentenceTransformer, util  # type: ignore
from transformers import PreTrainedModel, CLIPModel, CLIPProcessor  # type: ignore

from vector_retrieval_service.config import DEVICE
from vector_retrieval_service.domain.scores import compute_euclidean_distance


class LanguageModels(StrEnum):
    MINI_LM = auto()
    MPNET_BASE_V2 = auto()


class ImageModels(StrEnum):
    CLIP_MODEL = auto()
    FAKE = auto()


class SearchScoreFunctions(StrEnum):
    EUCLIDEAN = auto()
    COSINE_SIMILARITY = auto()
    DOT_PRODUCT = auto()


@lru_cache
def make_llm(model_type: LanguageModels | str) -> SentenceTransformer:
    try:
        language_model = LanguageModels(model_type)
    except ValueError as exc:
        raise ValueError(f"Unknown language model: {model_type}") from exc

    match language_model:
        case LanguageModels.MINI_LM:
            mini_lm = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", device=DEVICE
            )
            return mini_lm
        case LanguageModels.MPNET_BASE_V2:
            mpnet_v2 = SentenceTransformer(
                "sentence-transformers/all-mpnet-base-v2", device=DEVICE
            )
            return mpnet_v2


def make_score_function(
    score_function: SearchScoreFunctions | str,
) -> typing.Callable[..., typing.Any]:
    try:
        score_function_type = SearchScoreFunctions(score_function)
    except ValueError as exc:
        raise ValueError(f"Unknown score function: {score_function}") from exc

    match score_function_type:
        case SearchScoreFunctions.EUCLIDEAN:
            return compute_euclidean_distance
        case SearchScoreFunctions.COSINE_SIMILARITY:
            return util.cos_sim
        case SearchScoreFunctions.DOT_PRODUCT:
            return util.dot_score


@lru_cache
def make_image_model(
    model_type: ImageModels | str,
) -> tuple[PreTrainedModel, PreTrainedModel]:
    try:
        image_model = ImageModels(model_type)
    except ValueError as exc:
        raise ValueError(f"Unknown image model: {model_type}") from exc

    match image_model:
        case ImageModels.CLIP_MODEL | ImageModels.FAKE:
            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(
                DEVICE
            )
            clip_model.eval()
            clip_processor = CLIPProcessor.from_pretrained(
                "openai/clip-vit-base-patch32"
            )
            return clip_processor, clip_model
