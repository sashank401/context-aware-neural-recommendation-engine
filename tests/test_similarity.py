from src.retrieval.similarity import cosine_similarity


def test_cosine_similarity_identical_vectors():
    assert cosine_similarity([1, 0], [1, 0]) == 1.0


def test_cosine_similarity_opposite_vectors():
    assert cosine_similarity([1, 0], [-1, 0]) == -1.0


def test_cosine_similarity_perpendicular_vectors():
    assert cosine_similarity([1, 0], [0, 1]) == 0.0