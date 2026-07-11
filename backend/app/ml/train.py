"""Train an XGBoost regressor predicting log1p(duration_months)."""
import json
import os
import random
import time

import numpy as np
import xgboost as xgb
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from app.config import MODELS_DIR
from app.database.db import get_db
from app.ml.embeddings import embed_with_cache, build_text, EMB_DIM, MODEL_NAME
from app.ml.features import STRUCT_FEATURES, trial_row_features, to_vector

MIN_ROWS = 500
HEAVY = os.getenv("TRAIN_HEAVY") == "1"

_GRID = [
    dict(max_depth=4, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, min_child_weight=5),
    dict(max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=10),
    dict(max_depth=8, learning_rate=0.03, subsample=0.7, colsample_bytree=0.7, min_child_weight=20),
]

_HEAVY_GRID = _GRID + [
    dict(max_depth=6, learning_rate=0.03, subsample=0.9, colsample_bytree=0.6, min_child_weight=5),
    dict(max_depth=10, learning_rate=0.02, subsample=0.7, colsample_bytree=0.6, min_child_weight=30),
    dict(max_depth=12, learning_rate=0.02, subsample=0.6, colsample_bytree=0.5, min_child_weight=50),
    dict(max_depth=8, learning_rate=0.015, subsample=0.8, colsample_bytree=0.7, min_child_weight=20),
]


def _xgb_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_db_rows() -> list[dict]:
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM trials WHERE duration_months IS NOT NULL "
            "AND duration_months > 1 AND duration_months < 240 "
            "AND enrollment IS NOT NULL AND status = 'COMPLETED'"
        ).fetchall()
    return [dict(r) for r in rows]


def _build_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    nct_ids = [r["nct_id"] for r in rows]
    texts = [build_text(r.get("conditions"), r.get("interventions"), r.get("title")) for r in rows]
    emb = embed_with_cache(nct_ids, texts)

    X = np.array(
        [to_vector(trial_row_features(r), emb[i]) for i, r in enumerate(rows)],
        dtype="float32",
    )
    y = np.log1p(np.array([r["duration_months"] for r in rows], dtype="float64"))
    return X, y


def _synthetic_matrix(n: int = 400, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        phase = rng.choice([0, 1, 2, 3, 4])
        enrollment = max(10, int(rng.gauss(300, 250)))
        struct = {
            "phase_num": phase,
            "log_enrollment": np.log1p(enrollment),
            "n_arms": rng.choice([1, 2, 3, 4]),
            "n_endpoints_primary": rng.randint(1, 4),
            "n_endpoints_secondary": rng.randint(0, 8),
            "has_secondary_outcomes": rng.choice([0, 1]),
            "n_inclusion": rng.randint(1, 15),
            "n_exclusion": rng.randint(0, 15),
            "randomized": rng.choice([0, 1]),
            "blinded": rng.choice([0, 1]),
            "n_conditions": rng.randint(1, 3),
            "n_interventions": rng.randint(1, 4),
            "is_drug": rng.choice([0, 1]),
            "is_biological": 0,
            "is_device": 0,
            "is_behavioral": 0,
            "is_procedure": 0,
            "is_other_intervention": 0,
            "is_academic": rng.choice([0, 1]),
            "start_year": rng.randint(2005, 2023),
        }
        emb = [rng.gauss(0, 1) for _ in range(EMB_DIM)]
        duration = max(2, min(230, 6 + phase * 6 + enrollment * 0.02 + rng.gauss(0, 4)))
        X.append(to_vector(struct, emb))
        y.append(np.log1p(duration))
    return np.array(X, dtype="float32"), np.array(y, dtype="float64")


def _fit_one(params, X_train, y_train, X_val, y_val, n_estimators):
    model = xgb.XGBRegressor(
        n_estimators=n_estimators,
        tree_method="hist",
        device=_xgb_device(),
        early_stopping_rounds=50 if HEAVY else 30,
        eval_metric="rmse",
        random_state=42,
        **params,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    val_r2 = r2_score(y_val, model.predict(X_val))
    return model, val_r2


def train(save: bool = True, quick: bool = False) -> dict:
    start = time.time()

    if quick:
        X, y = _synthetic_matrix()
        synthetic = True
        n_estimators = 40
        grid = _GRID[:1]
    else:
        rows = _load_db_rows()
        synthetic = len(rows) < MIN_ROWS
        X, y = _synthetic_matrix() if synthetic else _build_matrix(rows)
        n_estimators = 6000 if HEAVY else 2000
        grid = _HEAVY_GRID if HEAVY else _GRID

    X_train, X_tmp, y_train, y_tmp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42)

    best_model, best_val_r2, best_params = None, -np.inf, None
    for params in grid:
        model, val_r2 = _fit_one(params, X_train, y_train, X_val, y_val, n_estimators)
        if val_r2 > best_val_r2:
            best_model, best_val_r2, best_params = model, val_r2, params

    pred_log = best_model.predict(X_test)
    r2_log = r2_score(y_test, pred_log)
    r2_raw = r2_score(np.expm1(y_test), np.expm1(np.clip(pred_log, 0, None)))

    emb_dim = X.shape[1] - len(STRUCT_FEATURES)
    names = STRUCT_FEATURES + [f"emb_{i}" for i in range(emb_dim)]
    gain = best_model.get_booster().get_score(importance_type="gain")
    named_gain = {names[int(k[1:])]: v for k, v in gain.items()}
    top_gain = sorted(named_gain.items(), key=lambda p: p[1], reverse=True)[:15]

    meta = {
        "n_rows": int(len(X)),
        "r2_log": round(float(r2_log), 4),
        "r2_raw": round(float(r2_raw), 4),
        "val_r2_log": round(float(best_val_r2), 4),
        "timestamp": time.time(),
        "synthetic": synthetic,
        "train_seconds": round(time.time() - start, 1),
        "params": best_params,
        "n_estimators_used": int(best_model.best_iteration or n_estimators),
        "struct_features": STRUCT_FEATURES,
        "emb_dim": int(emb_dim),
        "emb_model": MODEL_NAME,
        "heavy": HEAVY,
        "xgb_device": _xgb_device(),
        "top_gain_features": top_gain,
    }

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        best_model.save_model(str(MODELS_DIR / "duration_model.json"))
        with open(MODELS_DIR / "train_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    return meta


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2, default=str))
