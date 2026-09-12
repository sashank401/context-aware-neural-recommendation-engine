import numpy as np


def cosine_similarity(vector_a, vector_b):
    """Calculate cosine similarity between two vectors."""

    a = np.asarray(vector_a, dtype=float)
    b = np.asarray(vector_b, dtype=float)

    if a.ndim != 1 or b.ndim != 1:
        raise ValueError("Both vectors must be one-dimensional")

    if a.shape != b.shape:
        raise ValueError("Vectors must have the same dimensions")

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return float(np.dot(a, b) / (norm_a * norm_b))