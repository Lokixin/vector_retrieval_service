import numpy as np
import torch
from numpy._typing import NDArray


def compute_euclidean_distance(
    query_embedding: NDArray[np.float32], corpus_embedding: NDArray[np.float32]
) -> NDArray[np.float32]:
    euclidean_distances = torch.norm(corpus_embedding - query_embedding, dim=1)
    return euclidean_distances
