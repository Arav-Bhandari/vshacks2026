"""Cache trial text embeddings."""
import os
from contextlib import contextmanager

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import MODELS_DIR

EMB_DIM = 384
MODEL_NAME = os.getenv("EMB_MODEL", "all-MiniLM-L6-v2")

_SLUG = MODEL_NAME.replace("/", "_")
_EMB_PATH = MODELS_DIR / f"embeddings_v2_{_SLUG}.npy"
_IDS_PATH = MODELS_DIR / f"embeddings_ncts_v2_{_SLUG}.npy"

_cache: dict = {}


@contextmanager
def _cache_lock():
    """Lock the embedding cache."""
    import fcntl

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    lock_path = MODELS_DIR / f".embeddings_v2_{_SLUG}.lock"
    with open(lock_path, "a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


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


def _display_value(value) -> str:
    """Extract readable text."""
    if isinstance(value, dict):
        for key in ("name", "label", "title", "description", "value"):
            if value.get(key):
                return str(value[key]).strip()
        return ""
    return str(value or "").strip()


def _intervention_names(interventions) -> list[str]:
    if isinstance(interventions, (list, tuple)):
        return [name for item in interventions if (name := _display_value(item))]
    if isinstance(interventions, dict):
        name = _display_value(interventions)
        return [name] if name else []
    names = []
    for part in str(interventions or "").split("; "):
        name = part.split(":", 1)[-1].strip() if ":" in part else part.strip()
        if name:
            names.append(name)
    return names


def _joined_text(values) -> str:
    if values is None:
        return ""
    if isinstance(values, str):
        values = [part.strip() for part in values.split(";") if part.strip()]
    elif isinstance(values, dict):
        values = [values]
    return ", ".join(text for item in values if (text := _display_value(item)))


def build_text(
    conditions,
    interventions,
    title,
    primary_outcomes=None,
    secondary_outcomes=None,
    primary_outcome_timeframes=None,
    secondary_outcome_timeframes=None,
) -> str:
    if isinstance(conditions, (list, tuple)):
        conditions = ", ".join(text for item in conditions if (text := _display_value(item)))
    elif isinstance(conditions, dict):
        conditions = _display_value(conditions)
    names = _intervention_names(interventions)
    parts = [f"{conditions or ''}. {', '.join(names)}. {_display_value(title)}".strip()]
    optional_sections = (
        ("Primary outcomes", primary_outcomes),
        ("Secondary outcomes", secondary_outcomes),
        ("Primary outcome timeframes", primary_outcome_timeframes),
        ("Secondary outcome timeframes", secondary_outcome_timeframes),
    )
    for label, values in optional_sections:
        joined = _joined_text(values)
        if joined:
            parts.append(f"{label}: {joined}")
    return ". ".join(parts)


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
    """Read or build aligned embeddings."""
    if len(nct_ids) != len(texts):
        raise ValueError("embedding IDs and texts must have identical lengths")
    if len(set(nct_ids)) != len(nct_ids):
        raise ValueError("embedding cache IDs must be unique")

    with _cache_lock():
        cache_exists = _EMB_PATH.exists() or _IDS_PATH.exists()
        if cache_exists and not (_EMB_PATH.exists() and _IDS_PATH.exists()):
            raise RuntimeError("embedding cache is incomplete; remove both v2 cache files and rebuild")

        cached_emb = None
        ids_order: list[str] = []
        if cache_exists:
            cached_emb = np.load(_EMB_PATH, mmap_mode="r")
            ids_arr = np.load(_IDS_PATH, allow_pickle=False)
            ids_order = [str(value) for value in ids_arr.tolist()]
            if cached_emb.ndim != 2 or len(cached_emb) != len(ids_order):
                raise RuntimeError("embedding cache IDs and matrix have inconsistent lengths")
            if len(set(ids_order)) != len(ids_order):
                raise RuntimeError("embedding cache contains duplicate IDs")

        cached_ids = {nid: index for index, nid in enumerate(ids_order)}
        missing = [i for i, nid in enumerate(nct_ids) if nid not in cached_ids]
        if not missing:
            if cached_emb is None:
                return np.zeros((0, EMB_DIM), dtype="float32")
            return np.asarray(cached_emb[[cached_ids[nid] for nid in nct_ids]])

        first_block = missing[:_CHUNK]
        first_vectors = embed_texts([texts[i] for i in first_block])
        dim = int(first_vectors.shape[1])
        if cached_emb is not None and cached_emb.shape[1] != dim:
            raise RuntimeError(
                f"embedding cache dimension {cached_emb.shape[1]} does not match encoder dimension {dim}"
            )

        total = len(ids_order) + len(missing)
        emb_temp = _EMB_PATH.with_name(f".{_EMB_PATH.name}.{os.getpid()}.tmp")
        ids_temp = _IDS_PATH.with_name(f".{_IDS_PATH.name}.{os.getpid()}.tmp")
        output = np.lib.format.open_memmap(
            emb_temp, mode="w+", dtype="float32", shape=(total, dim)
        )
        if cached_emb is not None:
            output[: len(ids_order)] = cached_emb

        cursor = len(ids_order)
        for start in range(0, len(missing), _CHUNK):
            block = missing[start : start + _CHUNK]
            vectors = (
                first_vectors
                if start == 0
                else embed_texts([texts[i] for i in block])
            )
            if vectors.shape[1] != dim:
                raise RuntimeError("embedding encoder returned inconsistent dimensions")
            output[cursor : cursor + len(block)] = vectors
            ids_order.extend(nct_ids[i] for i in block)
            cursor += len(block)
            print(f"embedded {cursor}/{total} cached rows", flush=True)
        output.flush()
        del output

        max_chars = max((len(value) for value in ids_order), default=1)
        with open(ids_temp, "wb") as file:
            np.save(file, np.asarray(ids_order, dtype=f"<U{max_chars}"), allow_pickle=False)
        os.replace(emb_temp, _EMB_PATH)
        os.replace(ids_temp, _IDS_PATH)

        completed = np.load(_EMB_PATH, mmap_mode="r")
        final_ids = {nid: index for index, nid in enumerate(ids_order)}
        return np.asarray(completed[[final_ids[nid] for nid in nct_ids]])
