import numpy as np
import pytest

from src.models.embedding_loader import NumpyEmbeddingLoader


def create_embedding_files(tmp_path):
    np.save(
        tmp_path / "user_embeddings.npy",
        np.array([
            [1.0, 0.0],
            [0.0, 1.0],
        ])
    )

    np.save(
        tmp_path / "user_ids.npy",
        np.array(["user_a", "user_b"])
    )

    np.save(
        tmp_path / "item_embeddings.npy",
        np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ])
    )

    np.save(
        tmp_path / "item_ids.npy",
        np.array(["item_a", "item_b", "item_c"])
    )


def test_load_embeddings(tmp_path):
    create_embedding_files(tmp_path)

    loader = NumpyEmbeddingLoader(tmp_path).load()

    assert loader.user_embeddings.shape == (2, 2)
    assert loader.item_embeddings.shape == (3, 2)


def test_lookup_embeddings(tmp_path):
    create_embedding_files(tmp_path)

    loader = NumpyEmbeddingLoader(tmp_path).load()

    user_embedding = loader.get_user_embedding("user_a")
    item_embedding = loader.get_item_embedding("item_c")

    np.testing.assert_array_equal(
        user_embedding,
        np.array([1.0, 0.0])
    )

    np.testing.assert_array_equal(
        item_embedding,
        np.array([1.0, 1.0])
    )


def test_unknown_id_returns_none(tmp_path):
    create_embedding_files(tmp_path)

    loader = NumpyEmbeddingLoader(tmp_path).load()

    assert loader.get_user_embedding("unknown_user") is None
    assert loader.get_item_embedding("unknown_item") is None


def test_missing_embedding_file_raises_error(tmp_path):
    create_embedding_files(tmp_path)

    (tmp_path / "item_ids.npy").unlink()

    with pytest.raises(FileNotFoundError, match="item_ids.npy"):
        NumpyEmbeddingLoader(tmp_path).load()


def test_mismatched_embedding_dimensions_raise_error(tmp_path):
    create_embedding_files(tmp_path)

    np.save(
        tmp_path / "item_embeddings.npy",
        np.array([
            [1.0, 0.0, 0.5],
            [0.0, 1.0, 0.5],
            [1.0, 1.0, 0.5],
        ])
    )

    with pytest.raises(
        ValueError,
        match="embedding dimensions must match"
    ):
        NumpyEmbeddingLoader(tmp_path).load()
