"""Train an XGBoost regressor predicting trial duration_months."""
import json
import random
import time

import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from app.config import MODELS_DIR
from app.database.db import get_db
from app.ml.features import FEATURE_ORDER, trial_row_features

MIN_ROWS = 500


def _load_db_rows() -> tuple[list[list[float]], list[float]]:
    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM trials WHERE duration_months IS NOT NULL "
            "AND duration_months > 1 AND duration_months < 240 "
            "AND enrollment IS NOT NULL"
        ).fetchall()
    X, y = [], []
    for r in rows:
        row = dict(r)
        feats = trial_row_features(row)
        X.append([feats[k] for k in FEATURE_ORDER])
        y.append(row["duration_months"])
    return X, y


def _synthetic_data(n: int = 3000, seed: int = 42) -> tuple[list[list[float]], list[float]]:
    rng = random.Random(seed)
    X, y = [], []
    for _ in range(n):
        phase = rng.choice([1, 2, 3, 4])
        enrollment = max(10, int(rng.gauss(300, 250)))
        n_arms = rng.choice([1, 2, 2, 3, 4])
        n_ep_primary = rng.randint(1, 4)
        n_ep_secondary = rng.randint(0, 8)
        n_inclusion = rng.randint(2, 15)
        n_exclusion = rng.randint(2, 15)
        n_visits = rng.randint(3, 20)
        randomized = rng.choice([0, 1])
        blinded = rng.choice([0, 1])
        n_conditions = rng.randint(1, 3)
        n_interventions = rng.randint(1, 4)

        duration = (
            6
            + phase * 6
            + enrollment * 0.02
            + n_arms * 1.5
            + n_ep_primary * 0.8
            + n_ep_secondary * 0.3
            + n_inclusion * 0.3
            + n_exclusion * 0.3
            + n_visits * 0.4
            + randomized * 2
            + blinded * 1.5
            + rng.gauss(0, 4)
        )
        duration = max(2, min(230, duration))

        X.append([
            phase, enrollment, n_arms, n_ep_primary, n_ep_secondary,
            n_inclusion, n_exclusion, n_visits, randomized, blinded,
            n_conditions, n_interventions,
        ])
        y.append(duration)
    return X, y


def train(save: bool = True) -> dict:
    X, y = _load_db_rows()
    synthetic = len(X) < MIN_ROWS
    if synthetic:
        X, y = _synthetic_data()

    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, random_state=42,
    )
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))

    meta = {
        "n_rows": len(X),
        "r2": round(float(r2), 4),
        "timestamp": time.time(),
        "synthetic": synthetic,
        "feature_order": FEATURE_ORDER,
    }

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        model.save_model(str(MODELS_DIR / "duration_model.json"))
        with open(MODELS_DIR / "train_meta.json", "w") as f:
            json.dump(meta, f, indent=2)

    return meta


if __name__ == "__main__":
    result = train()
    print(json.dumps(result, indent=2))
