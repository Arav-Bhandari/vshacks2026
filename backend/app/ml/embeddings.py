"""Sentence embeddings for trial text, cached for cheap retrains."""
import os

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import MODELS_DIR

EMB_DIM = 384
MODEL_NAME = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")

_SLUG = MODEL_NAME.replace("/", "_")
_EMB_PATH = MODELS_DIR / f"embeddings_{_SLUG}.npy"
_IDS_PATH = MODELS_DIR / f"embeddings_ncts_{_SLUG}.npy"

_cache: dict = {}


def _device() -> str:
    if os.getenv("EMB_DEVICE"):
        return os.environ["EMB_DEVICE"]
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _model(name: str | None = None) -> SentenceTransformer:
    key = name or MODEL_NAME
    if key not in _cache:
        import torch
        torch.set_num_threads(int(os.getenv("EMB_THREADS", "4")))
        _cache[key] = SentenceTransformer(key, device=_device())
    return _cache[key]


def _intervention_names(interventions) -> list[str]:
    if isinstance(interventions, list):
        return [str(i) for i in interventions if i]
    names = []
    for part in str(interventions or "").split("; "):
        name = part.split(":", 1)[-1].strip() if ":" in part else part.strip()
        if name:
            names.append(name)
    return names


def build_text(conditions, interventions, title) -> str:
    if isinstance(conditions, list):
        conditions = ", ".join(str(c) for c in conditions)
    names = _intervention_names(interventions)
    return f"{conditions or ''}. {', '.join(names)}. {title or ''}".strip()


def embed_texts(texts: list[str], model_name: str | None = None) -> np.ndarray:
    vecs = _model(model_name).encode(
        texts, batch_size=int(os.getenv("EMB_BATCH", "128")),
        show_progress_bar=False, convert_to_numpy=True,
    )
    return vecs.astype("float32")


def embed_one(text: str, model_name: str | None = None) -> np.ndarray:
    return _model(model_name).encode([text], convert_to_numpy=True)[0].astype("float32")


_CHUNK = 20000


def embed_with_cache(nct_ids: list[str], texts: list[str]) -> np.ndarray:
    """Embeddings aligned to nct_ids; grow-only cache, resumable."""
    cached_emb = None
    cached_ids: dict = {}
    if _EMB_PATH.exists() and _IDS_PATH.exists():
        cached_emb = np.load(_EMB_PATH)
        ids_arr = np.load(_IDS_PATH, allow_pickle=True)
        cached_ids = {nid: i for i, nid in enumerate(ids_arr)}

    missing = [i for i, nid in enumerate(nct_ids) if nid not in cached_ids]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    for s in range(0, len(missing), _CHUNK):
        block = missing[s:s + _CHUNK]
        vecs = embed_texts([texts[i] for i in block])
        cached_emb = vecs if cached_emb is None else np.vstack([cached_emb, vecs])
        for i in block:
            cached_ids[nct_ids[i]] = len(cached_ids)
        ids_sorted = sorted(cached_ids, key=cached_ids.get)
        np.save(_EMB_PATH, cached_emb)
        np.save(_IDS_PATH, np.array(ids_sorted, dtype=object))
        print(f"embedded {len(cached_ids)} cached rows", flush=True)

    if cached_emb is None:
        return np.zeros((0, EMB_DIM), dtype="float32")
    idx = [cached_ids[nid] for nid in nct_ids]
    return cached_emb[idx]
