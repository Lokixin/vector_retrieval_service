import numpy as np
from numpy import floating
from numpy._typing import NDArray


def compute_euclidean_distance(
    query_embedding: NDArray[np.float32], corpus_embedding: NDArray[np.float32]
) -> floating:
    euclidean_distance = np.linalg.norm(corpus_embedding - query_embedding)
    return euclidean_distance
