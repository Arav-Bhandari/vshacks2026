"""Benchmark the sentence encoder."""
from __future__ import annotations

import json
import os
import time

import numpy as np
from sklearn.metrics import r2_score

from app.ml.embeddings import MODEL_NAME
from app.ml.features import STRUCT_FEATURES
from app.ml.stacking import RidgeTextHead
from app.ml.train import _build_matrix, _load_db_rows, _rolling_origin_splits


def benchmark() -> dict:
    started = time.time()
    rows = _load_db_rows()
    X, y = _build_matrix(rows)
    embeddings = X[:, len(STRUCT_FEATURES):]
    folds, test_idx, details = _rolling_origin_splits(rows)
    alphas = [float(value) for value in os.getenv("EMB_BENCHMARK_ALPHAS", "1,10,100").split(",")]
    search = []
    best = None
    for alpha in alphas:
        scores = []
        for fold in folds:
            head = RidgeTextHead.fit(embeddings[fold["train"]], y[fold["train"]], alpha)
            prediction = head.predict(embeddings[fold["validation"]])
            scores.append(float(r2_score(y[fold["validation"]], prediction)))
        mean = float(np.mean(scores))
        search.append({"alpha": alpha, "fold_r2_log": scores, "mean_r2_log": mean})
        if best is None or mean > best[0]:
            best = (mean, alpha)
    assert best is not None
    latest = max(folds, key=lambda fold: fold["validation_year"])
    reference = np.sort(np.concatenate([latest["train"], latest["validation"]]))
    head = RidgeTextHead.fit(embeddings[reference], y[reference], best[1])
    test_prediction = head.predict(embeddings[test_idx])
    return {
        "encoder": MODEL_NAME,
        "embedding_dim": int(embeddings.shape[1]),
        "n_rows": len(rows),
        "selection": search,
        "selected_alpha": best[1],
        "test_year": details["test_year"],
        "test_r2_log": float(r2_score(y[test_idx], test_prediction)),
        "seconds": round(time.time() - started, 1),
    }


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2))
