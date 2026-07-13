import numpy as np
import pytest

from app.ml import embeddings


def _fake_embed(texts, model_name=None):
    return np.asarray([[len(text), sum(map(ord, text)) % 17] for text in texts], dtype="float32")


def test_embedding_cache_is_aligned_and_reusable(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(embeddings, "_EMB_PATH", tmp_path / "vectors.npy")
    monkeypatch.setattr(embeddings, "_IDS_PATH", tmp_path / "ids.npy")
    monkeypatch.setattr(embeddings, "_CHUNK", 2)
    monkeypatch.setattr(embeddings, "embed_texts", _fake_embed)

    first = embeddings.embed_with_cache(["a", "b", "c"], ["one", "two", "three"])
    second = embeddings.embed_with_cache(["c", "a"], ["three", "one"])
    assert first.shape == (3, 2)
    np.testing.assert_array_equal(second, first[[2, 0]])


def test_embedding_cache_rejects_incomplete_pair(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(embeddings, "_EMB_PATH", tmp_path / "vectors.npy")
    monkeypatch.setattr(embeddings, "_IDS_PATH", tmp_path / "ids.npy")
    np.save(tmp_path / "vectors.npy", np.zeros((1, 2), dtype="float32"))
    with pytest.raises(RuntimeError, match="incomplete"):
        embeddings.embed_with_cache(["a"], ["one"])


def test_embedding_cache_rejects_duplicate_request_ids(tmp_path, monkeypatch):
    monkeypatch.setattr(embeddings, "MODELS_DIR", tmp_path)
    with pytest.raises(ValueError, match="unique"):
        embeddings.embed_with_cache(["a", "a"], ["one", "one"])
