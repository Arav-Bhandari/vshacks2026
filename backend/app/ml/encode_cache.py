"""Build the trial embedding cache."""
import hashlib
import time

from app.ml.embeddings import embed_with_cache


def load_ids_texts():
    from app.ml.train import _load_db_rows, _row_embedding_text

    rows = _load_db_rows()
    texts = [_row_embedding_text(row) for row in rows]
    ids = [
        f"{row['nct_id']}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
        for row, text in zip(rows, texts)
    ]
    return ids, texts


if __name__ == "__main__":
    t0 = time.time()
    ids, texts = load_ids_texts()
    print(f"rows {len(ids)} ready in {time.time()-t0:.0f}s", flush=True)
    emb = embed_with_cache(ids, texts)
    print(f"cache complete shape={emb.shape} total {time.time()-t0:.0f}s", flush=True)
