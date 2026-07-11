"""Build the embedding cache for all training rows; resumable."""
import pickle
import time

from app.config import MODELS_DIR
from app.ml.embeddings import build_text, embed_with_cache

_TEXTS_PKL = MODELS_DIR / "train_texts.pkl"


def load_ids_texts():
    if _TEXTS_PKL.exists():
        with open(_TEXTS_PKL, "rb") as f:
            return pickle.load(f)
    from app.ml.train import _load_db_rows
    rows = _load_db_rows()
    ids = [r["nct_id"] for r in rows]
    texts = [
        build_text(r.get("conditions"), r.get("interventions"), r.get("title"))
        for r in rows
    ]
    with open(_TEXTS_PKL, "wb") as f:
        pickle.dump((ids, texts), f)
    return ids, texts


if __name__ == "__main__":
    t0 = time.time()
    ids, texts = load_ids_texts()
    print(f"rows {len(ids)} ready in {time.time()-t0:.0f}s", flush=True)
    emb = embed_with_cache(ids, texts)
    print(f"cache complete shape={emb.shape} total {time.time()-t0:.0f}s", flush=True)
