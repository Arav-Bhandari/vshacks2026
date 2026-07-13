"""Train the censored-duration AFT model."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from app.config import MODELS_DIR
from app.database.db import get_db
from app.ml.embeddings import MODEL_NAME, embed_with_cache
from app.ml.features import STRUCT_FEATURES, to_vector, trial_row_features
from app.ml.train import _row_embedding_text, _start_year, _xgb_device

_ACTIVE = {
    "RECRUITING", "NOT_YET_RECRUITING", "ACTIVE_NOT_RECRUITING",
    "ENROLLING_BY_INVITATION",
}
_ENDED = {"COMPLETED", "TERMINATED", "WITHDRAWN", "SUSPENDED"}


def _parse_date(value) -> dt.date | None:
    text = str(value or "")
    try:
        if len(text) >= 10:
            return dt.date.fromisoformat(text[:10])
        if len(text) >= 7:
            return dt.date(int(text[:4]), int(text[5:7]), 1)
    except ValueError:
        return None
    return None


def _months(start: dt.date, end: dt.date) -> float:
    return max(1.0 / 30.4375, (end - start).days / 30.4375)


def _label_for_row(row: dict, observed_today: dt.date) -> tuple[float, float] | None:
    start = _parse_date(row.get("start_date"))
    if not start or start >= observed_today:
        return None
    status = str(row.get("status") or "")
    completion = _parse_date(row.get("completion_date"))
    completion_type = str(row.get("completion_date_type") or "").upper()
    if status in _ENDED and completion and completion > start and completion_type in {"", "ACTUAL"}:
        duration = _months(start, completion)
        return (duration, duration) if 1.0 < duration < 240.0 else None
    if status in _ACTIVE:
        observed_at = _parse_date(row.get("fetched_at")) or observed_today
        if observed_at <= start:
            return None
        return _months(start, observed_at), math.inf
    return None


def _load_survival_rows() -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Load rows and AFT duration bounds."""
    with get_db() as db:
        raw = db.execute(
            "SELECT * FROM trials WHERE start_date IS NOT NULL "
            "AND status IN ("
            + ",".join("?" for _ in sorted(_ACTIVE | _ENDED))
            + ") ORDER BY nct_id",
            tuple(sorted(_ACTIVE | _ENDED)),
        ).fetchall()

    today = dt.date.today()
    rows: list[dict] = []
    lower: list[float] = []
    upper: list[float] = []
    for sqlite_row in raw:
        row = dict(sqlite_row)
        label = _label_for_row(row, today)
        if label is None:
            continue
        lo, hi = label
        rows.append(row)
        lower.append(lo)
        upper.append(hi)
    return rows, np.asarray(lower, dtype=np.float32), np.asarray(upper, dtype=np.float32)


def _matrix(rows: list[dict], use_embeddings: bool) -> tuple[np.ndarray, int]:
    struct = np.asarray(
        [[trial_row_features(row)[name] for name in STRUCT_FEATURES] for row in rows],
        dtype=np.float32,
    )
    if not use_embeddings:
        return struct, 0
    texts = [_row_embedding_text(row) for row in rows]
    import hashlib

    ids = [
        f"{row['nct_id']}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
        for row, text in zip(rows, texts)
    ]
    embeddings = embed_with_cache(ids, texts)
    return np.concatenate([struct, embeddings], axis=1), int(embeddings.shape[1])


def _dmat(X: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> xgb.DMatrix:
    matrix = xgb.DMatrix(X)
    matrix.set_float_info("label_lower_bound", lower)
    matrix.set_float_info("label_upper_bound", upper)
    return matrix


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict:
    return {
        "r2_raw": float(r2_score(actual, prediction)),
        "mae_months": float(mean_absolute_error(actual, prediction)),
        "rmse_months": float(np.sqrt(mean_squared_error(actual, prediction))),
    }


def train_survival(save: bool = True, deep: bool = False) -> dict:
    started = time.time()
    rows, lower, upper = _load_survival_rows()
    if len(rows) < 500:
        raise RuntimeError(f"only {len(rows)} usable survival rows found")
    use_embeddings = os.getenv("AFT_USE_EMBEDDINGS", "0") == "1"
    X, emb_dim = _matrix(rows, use_embeddings)
    years = np.asarray([_start_year(row) or 0 for row in rows])
    train_end = int(os.getenv("AFT_TRAIN_END", "2016"))
    val_year = int(os.getenv("AFT_VAL_YEAR", str(train_end + 1)))
    test_year = int(os.getenv("AFT_TEST_YEAR", str(val_year + 1)))
    train_idx = np.flatnonzero((years > 0) & (years <= train_end))
    val_idx = np.flatnonzero(years == val_year)
    test_idx = np.flatnonzero(years == test_year)
    if min(map(len, (train_idx, val_idx, test_idx))) < 10:
        raise RuntimeError("AFT temporal cohorts are too small")

    configs = [
        {"max_depth": 5, "learning_rate": 0.04, "min_child_weight": 10,
         "subsample": 0.85, "colsample_bytree": 0.8, "reg_lambda": 4.0,
         "aft_loss_distribution": "normal", "aft_loss_distribution_scale": 1.2},
        {"max_depth": 7, "learning_rate": 0.025, "min_child_weight": 20,
         "subsample": 0.8, "colsample_bytree": 0.7, "reg_lambda": 8.0,
         "aft_loss_distribution": "logistic", "aft_loss_distribution_scale": 1.0},
    ]
    if deep:
        configs.extend([
            {"max_depth": 8, "learning_rate": 0.018, "min_child_weight": 30,
             "subsample": 0.75, "colsample_bytree": 0.65, "reg_lambda": 12.0,
             "reg_alpha": 0.5, "aft_loss_distribution": "normal",
             "aft_loss_distribution_scale": 0.8},
            {"max_depth": 10, "learning_rate": 0.012, "min_child_weight": 50,
             "subsample": 0.7, "colsample_bytree": 0.6, "reg_lambda": 16.0,
             "reg_alpha": 1.0, "aft_loss_distribution": "extreme",
             "aft_loss_distribution_scale": 1.2},
        ])
    budget = int(os.getenv("AFT_N_ESTIMATORS", "8000" if deep else "2500"))
    common = {
        "objective": "survival:aft", "eval_metric": "aft-nloglik",
        "tree_method": "hist", "device": _xgb_device(), "seed": 42,
    }
    dtrain = _dmat(X[train_idx], lower[train_idx], upper[train_idx])
    dval = _dmat(X[val_idx], lower[val_idx], upper[val_idx])
    search = []
    best = None
    for index, config in enumerate(configs):
        history: dict = {}
        model = xgb.train(
            {**common, **config}, dtrain, num_boost_round=budget,
            evals=[(dval, "validation")], evals_result=history,
            early_stopping_rounds=int(os.getenv("AFT_EARLY_STOPPING", "100")),
            verbose_eval=False,
        )
        score = float(min(history["validation"]["aft-nloglik"]))
        item = {"config_index": index, "params": config, "validation_aft_nloglik": score,
                "best_iteration": int(model.best_iteration)}
        search.append(item)
        if best is None or score < best[0]:
            best = (score, model, config)
    assert best is not None
    _, selected, params = best

    test_uncensored = test_idx[np.isfinite(upper[test_idx])]
    prediction = selected.predict(xgb.DMatrix(X[test_uncensored]), iteration_range=(0, selected.best_iteration + 1))
    test_metrics = _metrics(lower[test_uncensored], prediction)

    dall = _dmat(X, lower, upper)
    final = xgb.train(
        {**common, **params}, dall, num_boost_round=int(selected.best_iteration) + 1,
        verbose_eval=False,
    )
    censored = int(np.isinf(upper).sum())
    meta = {
        "schema_version": 1,
        "objective": "survival:aft",
        "n_rows": len(rows),
        "n_uncensored": len(rows) - censored,
        "n_right_censored": censored,
        "struct_features": STRUCT_FEATURES,
        "emb_dim": emb_dim,
        "emb_model": MODEL_NAME if use_embeddings else None,
        "feature_order": STRUCT_FEATURES + [f"emb_{i}" for i in range(emb_dim)],
        "temporal_split": {"train_end": train_end, "validation_year": val_year,
                           "test_year": test_year, "counts": {
                               "train": len(train_idx), "validation": len(val_idx),
                               "test": len(test_idx), "test_uncensored": len(test_uncensored)}},
        "test_metrics_uncensored": test_metrics,
        "params": params,
        "n_estimators_used": int(selected.best_iteration) + 1,
        "search_results": search,
        "train_seconds": round(time.time() - started, 1),
        "limitations": [
            "Registry features are latest snapshots, not historical as-of snapshots.",
            "The production regressor is not replaced automatically by this experiment.",
        ],
    }
    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        final.save_model(MODELS_DIR / "duration_aft_model.json")
        (MODELS_DIR / "duration_aft_meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep", action="store_true")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()
    print(json.dumps(train_survival(save=not args.no_save, deep=args.deep), indent=2))


if __name__ == "__main__":
    main()
