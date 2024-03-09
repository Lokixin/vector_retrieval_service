import logging
import typing
from enum import StrEnum, auto
from functools import lru_cache

import numpy as np
from numpy._typing import _32Bit
from numpy.typing import NDArray
import torch
from numpy import floating, dtype, ndarray
from sentence_transformers import SentenceTransformer  # type: ignore
from sentence_transformers import util
from transformers import CLIPModel, CLIPProcessor, PreTrainedModel  # type: ignore

logger = logging.getLogger(__name__)


DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


class LanguageModels(StrEnum):
    MINI_LM = auto()
    MPNET_BASE_V2 = auto()


class ImageModels(StrEnum):
    CLIP_MODEL = auto()
    CLIP_PROCESSOR = auto()


class SearchScoreFunctions(StrEnum):
    EUCLIDEAN = auto()
    COSINE_SIMILARITY = auto()
    DOT_PRODUCT = auto()


def compute_euclidean_distance(
    query_embedding: NDArray[np.float32], corpus_embedding: NDArray[np.float32]
) -> floating:
    euclidean_distance = np.linalg.norm(corpus_embedding - query_embedding)
    return euclidean_distance


class LLMFactory:
    @staticmethod
    @lru_cache
    def get_model(model_type: LanguageModels | str) -> SentenceTransformer:
        match model_type:
            case LanguageModels.MINI_LM:
                mini_lm = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2", device="cuda:0"
                )
                return mini_lm
            case LanguageModels.MPNET_BASE_V2:
                mpnet_v2 = SentenceTransformer(
                    "sentence-transformers/all-mpnet-base-v2", device="cuda:0"
                )
                return mpnet_v2
            case _:
                mini_lm = SentenceTransformer(
                    "sentence-transformers/all-MiniLM-L6-v2", device="cuda:0"
                )
                return mini_lm

    @staticmethod
    def get_score_function(
        score_function: SearchScoreFunctions | str,
    ) -> typing.Callable[
        [
            ndarray[typing.Any, dtype[floating[_32Bit]]],
            ndarray[typing.Any, dtype[floating[_32Bit]]],
        ],
        floating[typing.Any],
    ]:
        match score_function:
            case SearchScoreFunctions.EUCLIDEAN:
                return compute_euclidean_distance
            case SearchScoreFunctions.COSINE_SIMILARITY:
                return util.cos_sim
            case SearchScoreFunctions.DOT_PRODUCT:
                return util.dot_score
            case _:
                return util.dot_score

    @staticmethod
    @lru_cache
    def get_image_model(
        model_type: ImageModels | str,
    ) -> tuple[PreTrainedModel, PreTrainedModel]:
        match model_type:
            case ImageModels.CLIP_MODEL:
                clip_model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32"
                ).to(DEVICE)
                clip_processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                return clip_processor, clip_model
            case _:
                clip_model = CLIPModel.from_pretrained(
                    "openai/clip-vit-base-patch32"
                ).to(DEVICE)
                clip_processor = CLIPProcessor.from_pretrained(
                    "openai/clip-vit-base-patch32"
                )
                return clip_processor, clip_model
