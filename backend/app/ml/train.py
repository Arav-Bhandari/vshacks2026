"""Train and evaluate the trial duration model."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import inspect
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from app.config import MODELS_DIR
from app.database.db import get_db
from app.ml.embeddings import EMB_DIM, MODEL_NAME, build_text, embed_with_cache
from app.ml.features import STRUCT_FEATURES, to_vector, trial_row_features
from app.ml.stacking import (
    NeighborDurationBaseline,
    RidgeTextHead,
    apply_blend,
    coverage_preflight,
    fit_nonnegative_blend,
)

MIN_ROWS = 500
SEED = 42
HEAVY = os.getenv("TRAIN_DEEP", os.getenv("TRAIN_HEAVY", "0")) == "1"


_GRID = [
    dict(
        max_depth=4,
        learning_rate=0.06,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=5,
        reg_alpha=0.0,
        reg_lambda=2.0,
        gamma=0.0,
    ),
    dict(
        max_depth=6,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.80,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=3.0,
        gamma=0.0,
    ),
    dict(
        max_depth=8,
        learning_rate=0.025,
        subsample=0.75,
        colsample_bytree=0.75,
        min_child_weight=20,
        reg_alpha=0.5,
        reg_lambda=5.0,
        gamma=0.05,
    ),
    dict(
        max_depth=5,
        learning_rate=0.035,
        subsample=0.90,
        colsample_bytree=0.65,
        min_child_weight=15,
        reg_alpha=1.0,
        reg_lambda=8.0,
        gamma=0.1,
    ),
]

_DEEP_GRID = _GRID + [
    dict(
        max_depth=6,
        learning_rate=0.025,
        subsample=0.90,
        colsample_bytree=0.60,
        min_child_weight=5,
        reg_alpha=0.05,
        reg_lambda=1.0,
        gamma=0.0,
    ),
    dict(
        max_depth=8,
        learning_rate=0.018,
        subsample=0.85,
        colsample_bytree=0.70,
        min_child_weight=12,
        reg_alpha=0.25,
        reg_lambda=4.0,
        gamma=0.02,
    ),
    dict(
        max_depth=10,
        learning_rate=0.015,
        subsample=0.75,
        colsample_bytree=0.60,
        min_child_weight=30,
        reg_alpha=1.0,
        reg_lambda=8.0,
        gamma=0.10,
    ),
    dict(
        max_depth=12,
        learning_rate=0.012,
        subsample=0.65,
        colsample_bytree=0.55,
        min_child_weight=50,
        reg_alpha=2.0,
        reg_lambda=12.0,
        gamma=0.20,
    ),
    dict(
        max_depth=0,
        max_leaves=48,
        grow_policy="lossguide",
        learning_rate=0.025,
        subsample=0.85,
        colsample_bytree=0.75,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=4.0,
        gamma=0.02,
    ),
    dict(
        max_depth=0,
        max_leaves=96,
        grow_policy="lossguide",
        learning_rate=0.018,
        subsample=0.75,
        colsample_bytree=0.65,
        min_child_weight=20,
        reg_alpha=0.5,
        reg_lambda=8.0,
        gamma=0.05,
    ),
]


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    value = int(os.getenv(name, str(default)))
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}, got {value}")
    return value


def _xgb_device() -> str:
    override = os.getenv("TRAIN_DEVICE")
    if override:
        return override
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def _load_db_rows() -> list[dict]:
    """Load completed trials with usable duration labels."""

    with get_db() as db:
        rows = db.execute(
            "SELECT * FROM trials WHERE duration_months IS NOT NULL "
            "AND duration_months > 1 AND duration_months < 240 "
            "AND enrollment IS NOT NULL AND status = 'COMPLETED' "
            "AND (start_date_type IS NULL OR start_date_type = 'ACTUAL') "
            "AND (completion_date_type IS NULL OR completion_date_type = 'ACTUAL') "
            "ORDER BY nct_id"
        ).fetchall()
    return [dict(r) for r in rows]


def _row_embedding_text(row: dict) -> str:
    """Build embedding text from the available fields."""

    def structured(json_key: str, fallback_key: str):
        value = row.get(json_key)
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
                if parsed:
                    return parsed
            except (TypeError, ValueError, json.JSONDecodeError):
                pass
        return row.get(fallback_key)

    values = {
        "conditions": structured("conditions_json", "conditions"),
        "interventions": structured("interventions_json", "interventions"),
        "title": row.get("title"),
        "primary_outcomes": structured("primary_outcomes_json", "primary_outcomes"),
        "secondary_outcomes": structured("secondary_outcomes_json", "secondary_outcomes"),
        "primary_outcome_timeframes": structured(
            "primary_outcome_timeframes_json", "primary_outcome_timeframes"
        ),
        "secondary_outcome_timeframes": structured(
            "secondary_outcome_timeframes_json", "secondary_outcome_timeframes"
        ),
    }
    parameters = inspect.signature(build_text).parameters
    kwargs = {name: value for name, value in values.items() if name in parameters}
    return build_text(**kwargs)


def _build_matrix(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    texts = [_row_embedding_text(r) for r in rows]
    cache_ids = [
        f"{row['nct_id']}:{hashlib.sha256(text.encode()).hexdigest()[:16]}"
        for row, text in zip(rows, texts)
    ]
    emb = embed_with_cache(cache_ids, texts)

    X = np.asarray(
        [to_vector(trial_row_features(r), emb[i]) for i, r in enumerate(rows)],
        dtype="float32",
    )
    y = np.log1p(np.asarray([r["duration_months"] for r in rows], dtype="float64"))
    return X, y


def _synthetic_dataset(
    n: int = 400, seed: int = SEED
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    rng = random.Random(seed)
    X, y, rows = [], [], []
    for i in range(n):
        phase = rng.choice([0, 1, 2, 3, 4])
        enrollment = max(10, int(rng.gauss(300, 250)))
        struct = {name: 0.0 for name in STRUCT_FEATURES}
        struct.update(
            {
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
                "is_academic": rng.choice([0, 1]),
                "start_year": rng.randint(2005, 2021),
                "is_interventional": 1 if phase else 0,
                "study_type_missing": 0 if phase else 1,
            }
        )
        emb = [rng.gauss(0, 1) for _ in range(EMB_DIM)]
        duration = max(2, min(230, 6 + phase * 6 + enrollment * 0.02 + rng.gauss(0, 4)))
        X.append(to_vector(struct, emb))
        y.append(np.log1p(duration))
        rows.append(
            {
                "nct_id": f"SYNTHETIC-{i:06d}",
                "start_date": f"{int(struct['start_year']):04d}-01-01",
                "completion_date": None,
                "duration_months": duration,
                "sponsor": f"Synthetic Sponsor {i % 20}",
            }
        )
    return (
        np.asarray(X, dtype="float32"),
        np.asarray(y, dtype="float64"),
        rows,
    )


def _synthetic_matrix(n: int = 400, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    """Build the legacy synthetic matrix."""

    X, y, _ = _synthetic_dataset(n=n, seed=seed)
    return X, y


def _start_year(row: dict) -> int | None:
    match = re.match(r"\s*(\d{4})", str(row.get("start_date") or ""))
    return int(match.group(1)) if match else None


def _stable_random_split(indices: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    train_val, test = train_test_split(indices, test_size=0.10, random_state=seed)
    train, val = train_test_split(train_val, test_size=1 / 9, random_state=seed)
    return {
        "train": np.sort(np.asarray(train, dtype=int)),
        "validation": np.sort(np.asarray(val, dtype=int)),
        "test": np.sort(np.asarray(test, dtype=int)),
    }


def _stable_group_split(
    rows: list[dict], indices: np.ndarray, seed: int
) -> dict[str, np.ndarray] | None:
    """Split rows while holding sponsors out."""

    groups = np.asarray(
        [
            _normalise_group(rows[int(index)].get("sponsor"))
            if _normalise_group(rows[int(index)].get("sponsor")) != "unknown"
            else f"unknown:{rows[int(index)].get('nct_id')}"
            for index in indices
        ]
    )
    if len(set(groups)) < 10:
        return None
    outer = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=seed)
    train_val_pos, test_pos = next(outer.split(indices, groups=groups))
    train_val = indices[train_val_pos]
    train_val_groups = groups[train_val_pos]
    inner = GroupShuffleSplit(n_splits=1, test_size=1 / 9, random_state=seed)
    train_pos, val_pos = next(inner.split(train_val, groups=train_val_groups))
    return {
        "train": np.sort(np.asarray(train_val[train_pos], dtype=int)),
        "validation": np.sort(np.asarray(train_val[val_pos], dtype=int)),
        "test": np.sort(np.asarray(indices[test_pos], dtype=int)),
    }


def _parse_years(value: str) -> list[int]:
    years = sorted({int(part.strip()) for part in value.split(",") if part.strip()})
    if not years:
        raise ValueError("at least one rolling validation year is required")
    return years


def _rolling_origin_splits(
    rows: list[dict],
) -> tuple[list[dict[str, Any]], np.ndarray, dict[str, Any]]:
    """Create expanding-window folds."""

    validation_years = _parse_years(
        os.getenv("TRAIN_ROLLING_VALIDATION_YEARS", "2015,2016,2017")
    )
    test_year = _env_int("TRAIN_TEMPORAL_TEST_YEAR", 2018, minimum=1900)
    if any(year >= test_year for year in validation_years):
        raise ValueError("rolling validation years must all precede the temporal test year")
    years = np.asarray([_start_year(row) or -1 for row in rows])
    minimum = _env_int("TRAIN_MIN_TEMPORAL_COHORT_ROWS", 10)
    folds: list[dict[str, Any]] = []
    unavailable = []
    for validation_year in validation_years:
        train_idx = np.flatnonzero((years > 0) & (years < validation_year))
        val_idx = np.flatnonzero(years == validation_year)
        if len(train_idx) < minimum or len(val_idx) < minimum:
            unavailable.append(
                {
                    "validation_year": validation_year,
                    "train_rows": int(len(train_idx)),
                    "validation_rows": int(len(val_idx)),
                }
            )
            continue
        folds.append(
            {
                "validation_year": validation_year,
                "train": train_idx,
                "validation": val_idx,
            }
        )
    test_idx = np.flatnonzero(years == test_year)
    if len(folds) < 2:
        raise RuntimeError(
            "Rolling-origin selection requires at least two usable validation years; "
            "adjust TRAIN_ROLLING_VALIDATION_YEARS for this dataset."
        )
    if len(test_idx) < minimum:
        raise RuntimeError(
            f"Temporal test year {test_year} has only {len(test_idx)} rows; at least {minimum} required."
        )
    details = {
        "strategy": "expanding_window_rolling_origin",
        "validation_years_requested": validation_years,
        "validation_years_used": [fold["validation_year"] for fold in folds],
        "test_year": test_year,
        "test_rows": int(len(test_idx)),
        "unavailable_folds": unavailable,
        "selection_uses_test": False,
    }
    return folds, test_idx, details


def _ridge_alphas() -> list[float]:
    value = os.getenv("TRAIN_RIDGE_ALPHAS", "0.1,1,10,100")
    alphas = sorted({float(part.strip()) for part in value.split(",") if part.strip()})
    if not alphas or any(alpha <= 0 for alpha in alphas):
        raise ValueError("TRAIN_RIDGE_ALPHAS must contain positive comma-separated values")
    return alphas


def _select_ridge_alpha(
    embeddings: np.ndarray,
    y: np.ndarray,
    folds: list[dict[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    results = []
    best_alpha, best_score = None, -np.inf
    for alpha in _ridge_alphas():
        fold_results = []
        for fold in folds:
            head = RidgeTextHead.fit(
                embeddings[fold["train"]], y[fold["train"]], alpha
            )
            prediction = head.predict(embeddings[fold["validation"]])
            metrics = _metrics(y[fold["validation"]], prediction)
            fold_results.append(
                {
                    "validation_year": fold["validation_year"],
                    "metrics": _round_metrics(metrics),
                }
            )
        score = float(
            np.mean([result["metrics"]["log"]["r2"] for result in fold_results])
        )
        results.append(
            {"alpha": alpha, "mean_validation_r2_log": score, "folds": fold_results}
        )
        if score > best_score:
            best_alpha, best_score = alpha, score
    assert best_alpha is not None
    return float(best_alpha), _round_floats(results)


def _stack_predictions(
    blend: dict[str, Any], xgb_log: np.ndarray, text_log: np.ndarray
) -> np.ndarray:
    return apply_blend(blend, xgb_log=xgb_log, text_log=text_log)


def _raw_prediction_metrics(
    y_log: np.ndarray, predicted_raw: np.ndarray
) -> dict[str, float]:
    return _scale_metrics(np.expm1(y_log), np.asarray(predicted_raw, dtype=float))


def _pipeline_metrics(
    y_log: np.ndarray,
    xgb_log: np.ndarray,
    text_log: np.ndarray,
    stack_blend: dict[str, Any],
    calibrator: dict[str, Any],
    neighbor_raw: np.ndarray | None = None,
    baseline_blend: dict[str, Any] | None = None,
) -> dict[str, Any]:
    stacked_log = _stack_predictions(stack_blend, xgb_log, text_log)
    calibrated_raw = _apply_calibrator(stacked_log, calibrator)
    result = {
        "xgb": _round_metrics(_metrics(y_log, xgb_log)),
        "text_ridge": _round_metrics(_metrics(y_log, text_log)),
        "stacked": _round_metrics(_metrics(y_log, stacked_log, calibrator)),
    }
    if neighbor_raw is not None:
        result["neighbor_baseline_raw"] = _round_floats(
            _raw_prediction_metrics(y_log, neighbor_raw)
        )
    if neighbor_raw is not None and baseline_blend is not None:
        blended_raw = apply_blend(
            baseline_blend,
            model_raw=calibrated_raw,
            baseline_raw=neighbor_raw,
        )
        result["stacked_with_neighbor_baseline_raw"] = _round_floats(
            _raw_prediction_metrics(y_log, np.maximum(1.0, blended_raw))
        )
    return result


def _model_kwargs(params: dict, n_estimators: int, seed: int) -> dict:
    return {
        "n_estimators": int(n_estimators),
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "device": _xgb_device(),
        "eval_metric": "rmse",
        "random_state": seed,
        "n_jobs": _env_int("TRAIN_N_JOBS", max(1, os.cpu_count() or 1)),
        **params,
    }


def _fit_one(
    params,
    X_train,
    y_train,
    X_val,
    y_val,
    n_estimators,
    *,
    seed: int = SEED,
    early_stopping_rounds: int | None = None,
):
    early_stopping_rounds = early_stopping_rounds or _env_int(
        "TRAIN_EARLY_STOPPING_ROUNDS", 100 if HEAVY else 50
    )
    model = xgb.XGBRegressor(
        **_model_kwargs(params, n_estimators, seed),
        early_stopping_rounds=early_stopping_rounds,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    pred_log = model.predict(X_val)
    metrics = _metrics(y_val, pred_log)
    return model, metrics["log"]["r2"]


def _fit_calibrator(y_log: np.ndarray, pred_log: np.ndarray) -> dict[str, Any]:
    """Fit month-scale calibration on validation data."""

    actual = np.expm1(y_log)
    raw_pred = np.expm1(np.clip(pred_log, 0, None))
    smearing = float(np.mean(np.exp(np.clip(y_log - pred_log, -10, 10))))
    smearing_pred = np.exp(np.clip(pred_log, 0, None)) * smearing - 1.0

    design = np.column_stack([raw_pred, np.ones_like(raw_pred)])
    slope, intercept = np.linalg.lstsq(design, actual, rcond=None)[0]
    affine_valid = bool(np.isfinite(slope) and np.isfinite(intercept) and slope > 0)
    affine_pred = slope * raw_pred + intercept if affine_valid else raw_pred

    candidates = {
        "none": raw_pred,
        "smearing": smearing_pred,
        "affine_raw": affine_pred,
    }
    scores = {name: float(r2_score(actual, pred)) for name, pred in candidates.items()}
    if not affine_valid:
        scores["affine_raw"] = -math.inf
    method = max(scores, key=scores.get)
    return {
        "method": method,
        "fitted_on": "validation",
        "smearing_factor": smearing,
        "slope": float(slope) if affine_valid else 1.0,
        "intercept": float(intercept) if affine_valid else 0.0,
        "validation_r2_raw_by_method": {
            name: (round(score, 6) if np.isfinite(score) else None)
            for name, score in scores.items()
        },
    }


def _apply_calibrator(pred_log: np.ndarray, calibrator: dict[str, Any] | None) -> np.ndarray:
    pred_log = np.clip(np.asarray(pred_log, dtype=float), 0, None)
    raw = np.expm1(pred_log)
    if not calibrator or calibrator.get("method") == "none":
        return raw
    if calibrator.get("method") == "smearing":
        calibrated = np.exp(pred_log) * float(calibrator["smearing_factor"]) - 1.0
        return np.maximum(1.0, calibrated)
    if calibrator.get("method") == "affine_raw":
        calibrated = float(calibrator["slope"]) * raw + float(calibrator["intercept"])
        return np.maximum(1.0, calibrated)
    raise ValueError(f"Unknown calibration method: {calibrator.get('method')}")


def _scale_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        "r2": float(r2_score(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
        "mae": float(mean_absolute_error(actual, predicted)),
    }


def _metrics(
    y_log: np.ndarray,
    pred_log: np.ndarray,
    calibrator: dict[str, Any] | None = None,
) -> dict[str, dict[str, float]]:
    pred_log = np.asarray(pred_log, dtype=float)
    actual_raw = np.expm1(y_log)
    raw_uncalibrated = np.expm1(np.clip(pred_log, 0, None))
    return {
        "log": _scale_metrics(y_log, pred_log),
        "raw_uncalibrated": _scale_metrics(actual_raw, raw_uncalibrated),
        "raw_calibrated": _scale_metrics(
            actual_raw, _apply_calibrator(pred_log, calibrator)
        ),
    }


def _best_iteration(model: xgb.XGBRegressor, budget: int) -> int:
    """Return the tree count for refitting."""

    try:
        return int(model.best_iteration) + 1
    except (AttributeError, TypeError):
        return int(budget)


def _fit_refit_model(
    params: dict,
    X: np.ndarray,
    y: np.ndarray,
    n_estimators: int,
    seed: int,
) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor(**_model_kwargs(params, n_estimators, seed))
    model.fit(X, y, verbose=False)
    return model


def _save_xgb_atomic(model: xgb.XGBRegressor, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
    model.save_model(str(temporary))
    os.replace(temporary, path)


def _write_json_atomic(path: Path, value: Any, *, compact: bool = False) -> None:
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(temporary, "w") as file:
        if compact:
            json.dump(value, file, separators=(",", ":"))
        else:
            json.dump(value, file, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


def _normalise_group(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "unknown").lower()).strip()


def _group_leakage_audit(rows: list[dict], split: dict[str, np.ndarray]) -> dict[str, Any]:
    groups = {
        name: {_normalise_group(rows[int(i)].get("sponsor")) for i in indices}
        for name, indices in split.items()
    }
    train = groups["train"]
    val = groups["validation"]
    test = groups["test"]
    return {
        "group_key": "normalised_sponsor",
        "unique_groups": {name: len(values) for name, values in groups.items()},
        "train_validation_overlap": len(train & val),
        "train_test_overlap": len(train & test),
        "validation_test_overlap": len(val & test),
        "note": (
            "The legacy split is row-random and therefore has sponsor-group leakage. "
            "These overlap counts make that limitation explicit; the temporal backtest "
            "is the preferred generalisation metric."
        ),
    }


def _dataset_fingerprint(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            (json.dumps(row, sort_keys=True, separators=(",", ":"), default=str) + "\n").encode()
        )
    return digest.hexdigest()


def _label_date_type_audit(rows: list[dict]) -> dict[str, dict[str, int]]:
    """Count the duration endpoint date types."""

    def counts(column: str) -> dict[str, int]:
        result: dict[str, int] = {}
        for row in rows:
            value = str(row.get(column) or "NULL").upper()
            result[value] = result.get(value, 0) + 1
        return dict(sorted(result.items()))

    return {
        "start_date_type": counts("start_date_type"),
        "completion_date_type": counts("completion_date_type"),
    }


def _source_coverage(rows: list[dict], indices: np.ndarray) -> dict[str, Any]:
    fields = [
        "conditions_json",
        "interventions_json",
        "primary_outcomes_json",
        "secondary_outcomes_json",
        "eligibility_inclusion_count",
        "eligibility_exclusion_count",
        "site_count",
        "country_count",
        "snapshot_type",
    ]
    selected = [rows[int(index)] for index in indices]
    rates = {
        field: round(
            sum(row.get(field) not in (None, "", [], {}) for row in selected) / len(selected),
            6,
        )
        for field in fields
    }
    snapshot_counts: dict[str, int] = {}
    enrollment_type_counts: dict[str, int] = {}
    for row in selected:
        snapshot = str(row.get("snapshot_type") or "UNKNOWN").upper()
        snapshot_counts[snapshot] = snapshot_counts.get(snapshot, 0) + 1
        enrollment_type = str(row.get("enrollment_type") or "UNKNOWN").upper()
        enrollment_type_counts[enrollment_type] = enrollment_type_counts.get(enrollment_type, 0) + 1
    return {
        "field_nonmissing_rates": rates,
        "snapshot_type_counts": dict(sorted(snapshot_counts.items())),
        "enrollment_type_counts": dict(sorted(enrollment_type_counts.items())),
    }


def _file_sha256(path: Path) -> str | None:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return None


def _git_provenance() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]

    def run(*args: str) -> str | None:
        try:
            return subprocess.run(
                ["git", *args],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            return None

    status = run("status", "--porcelain")
    tracked_files = [
        Path(__file__),
        Path(__file__).with_name("features.py"),
        Path(__file__).with_name("embeddings.py"),
    ]
    return {
        "git_commit": run("rev-parse", "HEAD"),
        "git_dirty": bool(status) if status is not None else None,
        "code_sha256": {
            path.name: _file_sha256(path) for path in tracked_files
        },
    }


def _round_floats(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _round_floats(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_round_floats(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return round(float(value), 6)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _round_metrics(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return _round_floats(metrics)


def _manifest(
    rows: list[dict],
    random_split: dict[str, np.ndarray],
    temporal_split: dict[str, np.ndarray] | None,
    group_split: dict[str, np.ndarray] | None,
    rolling_folds: list[dict[str, Any]] | None,
    temporal_test: np.ndarray | None,
    fingerprint: str,
    seed: int,
) -> dict[str, Any]:
    ids = [str(row.get("nct_id")) for row in rows]

    def id_split(split: dict[str, np.ndarray] | None) -> dict[str, list[str]] | None:
        if split is None:
            return None
        return {
            name: sorted(ids[int(i)] for i in indices)
            for name, indices in split.items()
        }

    payload = {
        "version": 2,
        "dataset_sha256": fingerprint,
        "seed": seed,
        "legacy_random": id_split(random_split),
        "mature_temporal": id_split(temporal_split),
        "sponsor_grouped": id_split(group_split),
        "rolling_origin": [
            {
                "validation_year": fold["validation_year"],
                "train": sorted(ids[int(i)] for i in fold["train"]),
                "validation": sorted(ids[int(i)] for i in fold["validation"]),
            }
            for fold in (rolling_folds or [])
        ],
        "temporal_test": (
            sorted(ids[int(i)] for i in temporal_test)
            if temporal_test is not None
            else None
        ),
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["manifest_sha256"] = hashlib.sha256(canonical).hexdigest()
    return payload


def train(
    save: bool = True,
    quick: bool = False,
    deep: bool | None = None,
    allow_demo_save: bool = False,
) -> dict:
    """Train the duration stack."""

    started = time.time()
    seed = _env_int("TRAIN_SEED", SEED, minimum=0)
    deep = HEAVY if deep is None else bool(deep)
    if quick and save and not allow_demo_save:
        raise ValueError(
            "Refusing to save a synthetic demo model. Pass allow_demo_save=True "
            "or CLI --allow-demo-save explicitly."
        )

    if quick:
        X, y, rows = _synthetic_dataset(seed=seed)
        synthetic = True
        n_estimators = _env_int("TRAIN_N_ESTIMATORS", 60)
        grid = _GRID[:1]
    else:
        rows = _load_db_rows()
        if len(rows) < MIN_ROWS:
            raise RuntimeError(
                f"Only {len(rows)} eligible real rows were found; at least {MIN_ROWS} "
                "are required. Use train(quick=True, save=False) only for a demo run."
            )
        synthetic = False
        X, y = _build_matrix(rows)
        n_estimators = _env_int("TRAIN_N_ESTIMATORS", 12000 if deep else 3500)
        grid = _DEEP_GRID if deep else _GRID

    max_configs = _env_int("TRAIN_MAX_CONFIGS", len(grid))
    grid = grid[:max_configs]
    early_stopping_rounds = _env_int(
        "TRAIN_EARLY_STOPPING_ROUNDS", 100 if deep else 50
    )
    selection_metric = os.getenv("TRAIN_SELECTION_METRIC", "raw_r2")
    if selection_metric not in {"raw_r2", "log_r2"}:
        raise ValueError("TRAIN_SELECTION_METRIC must be raw_r2 or log_r2")

    years = np.asarray([_start_year(row) or 9999 for row in rows])
    maturity_years = _env_int("TRAIN_PRODUCTION_MATURITY_YEARS", 5)
    maturity_cutoff = dt.date.today().year - maturity_years
    eligible = np.flatnonzero(years <= maturity_cutoff)
    if len(eligible) < (30 if synthetic else MIN_ROWS):
        raise RuntimeError(
            f"Only {len(eligible)} maturity-eligible rows start on/before {maturity_cutoff}."
        )

    rolling_folds, temporal_test_idx, rolling_details = _rolling_origin_splits(rows)
    latest_fold = max(rolling_folds, key=lambda fold: fold["validation_year"])
    train_idx = latest_fold["train"]
    val_idx = latest_fold["validation"]
    test_idx = temporal_test_idx
    temporal_split = {"train": train_idx, "validation": val_idx, "test": test_idx}
    random_split = _stable_random_split(eligible, seed)
    group_split = _stable_group_split(rows, eligible, seed)

    n_struct = len(STRUCT_FEATURES)
    embeddings = X[:, n_struct:]
    coverage = coverage_preflight(X[eligible, :n_struct], list(STRUCT_FEATURES), len(eligible))
    coverage["source"] = _source_coverage(rows, eligible)
    if coverage["source"]["snapshot_type_counts"].get("UNKNOWN", 0) == len(eligible):
        coverage["warnings"].append(
            "No versioned/initial snapshot provenance is populated for the training cohort"
        )
    if coverage["warnings"]:
        warnings.warn("; ".join(coverage["warnings"]), RuntimeWarning, stacklevel=2)

    search_results = []
    best_params: dict[str, Any] | None = None
    best_score = -np.inf
    for config_index, params in enumerate(grid):
        fold_results = []
        for fold in rolling_folds:
            print(
                f"xgb search {config_index + 1}/{len(grid)} "
                f"validation_year={fold['validation_year']}",
                file=sys.stderr,
                flush=True,
            )
            model, _ = _fit_one(
                params,
                X[fold["train"]],
                y[fold["train"]],
                X[fold["validation"]],
                y[fold["validation"]],
                n_estimators,
                seed=seed,
                early_stopping_rounds=early_stopping_rounds,
            )
            prediction = model.predict(X[fold["validation"]])
            metrics = _metrics(y[fold["validation"]], prediction)
            score = (
                metrics["raw_uncalibrated"]["r2"]
                if selection_metric == "raw_r2"
                else metrics["log"]["r2"]
            )
            fold_results.append(
                {
                    "validation_year": fold["validation_year"],
                    "n_estimators_used": _best_iteration(model, n_estimators),
                    "selection_score": float(score),
                    "metrics": _round_metrics(metrics),
                }
            )
        mean_score = float(np.mean([fold["selection_score"] for fold in fold_results]))
        search_results.append(
            {
                "config_index": config_index,
                "params": params,
                "mean_rolling_selection_score": mean_score,
                "folds": _round_floats(fold_results),
            }
        )
        if mean_score > best_score:
            best_score, best_params = mean_score, params
    assert best_params is not None

    best_alpha, ridge_search = _select_ridge_alpha(embeddings, y, rolling_folds)
    print(f"selected Ridge alpha={best_alpha}", file=sys.stderr, flush=True)

    selected_model, _ = _fit_one(
        best_params,
        X[train_idx],
        y[train_idx],
        X[val_idx],
        y[val_idx],
        n_estimators,
        seed=seed,
        early_stopping_rounds=early_stopping_rounds,
    )
    best_trees = _best_iteration(selected_model, n_estimators)
    selected_text_head = RidgeTextHead.fit(embeddings[train_idx], y[train_idx], best_alpha)
    validation_xgb_log = selected_model.predict(X[val_idx])
    validation_text_log = selected_text_head.predict(embeddings[val_idx])
    stack_blend = fit_nonnegative_blend(
        [validation_xgb_log, validation_text_log],
        y[val_idx],
        ["xgb_log", "text_log"],
    )
    validation_stacked_log = _stack_predictions(
        stack_blend, validation_xgb_log, validation_text_log
    )
    calibration = _fit_calibrator(y[val_idx], validation_stacked_log)
    validation_model_raw = _apply_calibrator(validation_stacked_log, calibration)

    neighbor = NeighborDurationBaseline(
        n_neighbors=_env_int("TRAIN_NEIGHBOR_K", 25),
        max_reference=_env_int("TRAIN_NEIGHBOR_MAX_REFERENCE", 50_000),
        seed=seed,
    ).fit(embeddings[train_idx], np.expm1(y[train_idx]))
    print("fit temporal neighbor baseline", file=sys.stderr, flush=True)
    validation_neighbor_raw = neighbor.predict(embeddings[val_idx])
    raw_baseline_blend = fit_nonnegative_blend(
        [validation_model_raw, validation_neighbor_raw],
        np.expm1(y[val_idx]),
        ["model_raw", "baseline_raw"],
    )
    baseline_blend = {
        "model_weight": raw_baseline_blend["weights"]["model_raw"],
        "baseline_weight": raw_baseline_blend["weights"]["baseline_raw"],
        "intercept": raw_baseline_blend["intercept"],
        "nonnegative_weights": True,
        "fitted_on": f"temporal_validation_{latest_fold['validation_year']}",
        "baseline_source": "embedding_neighbor_duration",
    }

    reference_idx = np.sort(np.concatenate([train_idx, val_idx]))
    test_neighbor = NeighborDurationBaseline(
        n_neighbors=_env_int("TRAIN_NEIGHBOR_K", 25),
        max_reference=_env_int("TRAIN_NEIGHBOR_MAX_REFERENCE", 50_000),
        seed=seed,
    ).fit(embeddings[reference_idx], np.expm1(y[reference_idx]))
    test_neighbor_raw = test_neighbor.predict(embeddings[test_idx])
    selected_test_xgb_log = selected_model.predict(X[test_idx])
    selected_test_text_log = selected_text_head.predict(embeddings[test_idx])
    temporal_test_metrics = _pipeline_metrics(
        y[test_idx],
        selected_test_xgb_log,
        selected_test_text_log,
        stack_blend,
        calibration,
        test_neighbor_raw,
        raw_baseline_blend,
    )
    temporal_validation_metrics = _pipeline_metrics(
        y[val_idx],
        validation_xgb_log,
        validation_text_log,
        stack_blend,
        calibration,
        validation_neighbor_raw,
        raw_baseline_blend,
    )

    final_model = _fit_refit_model(
        best_params, X[reference_idx], y[reference_idx], best_trees, seed
    )
    final_text_head = RidgeTextHead.fit(
        embeddings[reference_idx], y[reference_idx], best_alpha
    )
    print("refit generic XGBoost and Ridge stack", file=sys.stderr, flush=True)
    final_test_xgb_log = final_model.predict(X[test_idx])
    final_test_text_log = final_text_head.predict(embeddings[test_idx])
    production_refit_metrics = _pipeline_metrics(
        y[test_idx],
        final_test_xgb_log,
        final_test_text_log,
        stack_blend,
        calibration,
        test_neighbor_raw,
        raw_baseline_blend,
    )

    random_model, _ = _fit_one(
        best_params,
        X[random_split["train"]],
        y[random_split["train"]],
        X[random_split["validation"]],
        y[random_split["validation"]],
        n_estimators,
        seed=seed,
        early_stopping_rounds=early_stopping_rounds,
    )
    random_text = RidgeTextHead.fit(
        embeddings[random_split["train"]], y[random_split["train"]], best_alpha
    )
    random_val_xgb = random_model.predict(X[random_split["validation"]])
    random_val_text = random_text.predict(embeddings[random_split["validation"]])
    random_stack = fit_nonnegative_blend(
        [random_val_xgb, random_val_text],
        y[random_split["validation"]],
        ["xgb_log", "text_log"],
    )
    random_calibration = _fit_calibrator(
        y[random_split["validation"]],
        _stack_predictions(random_stack, random_val_xgb, random_val_text),
    )
    random_test_xgb = random_model.predict(X[random_split["test"]])
    random_test_text = random_text.predict(embeddings[random_split["test"]])
    legacy_metrics = _pipeline_metrics(
        y[random_split["test"]],
        random_test_xgb,
        random_test_text,
        random_stack,
        random_calibration,
    )

    if group_split is None:
        grouped_evaluation: dict[str, Any] = {"available": False}
    else:
        print("fitting sponsor-held-out diagnostic", file=sys.stderr, flush=True)
        grouped_model, _ = _fit_one(
            best_params,
            X[group_split["train"]],
            y[group_split["train"]],
            X[group_split["validation"]],
            y[group_split["validation"]],
            n_estimators,
            seed=seed,
            early_stopping_rounds=early_stopping_rounds,
        )
        grouped_validation = grouped_model.predict(X[group_split["validation"]])
        grouped_calibration = _fit_calibrator(
            y[group_split["validation"]], grouped_validation
        )
        grouped_evaluation = {
            "available": True,
            "strategy": "normalised_sponsor_group_held_out_80_10_10",
            "counts": {name: int(len(idx)) for name, idx in group_split.items()},
            "metrics": _round_metrics(
                _metrics(
                    y[group_split["test"]],
                    grouped_model.predict(X[group_split["test"]]),
                    grouped_calibration,
                )
            ),
        }

    phase_index = STRUCT_FEATURES.index("phase_num")
    interventional_index = STRUCT_FEATURES.index("is_interventional")
    specialist_mask = (
        np.isin(np.rint(X[:, phase_index]).astype(int), [2, 3])
        & (X[:, interventional_index] > 0.5)
    )
    specialist_split = {
        name: indices[specialist_mask[indices]] for name, indices in temporal_split.items()
    }
    specialist_model: xgb.XGBRegressor | None = None
    specialist_meta: dict[str, Any]
    if min(len(indices) for indices in specialist_split.values()) < 10:
        specialist_meta = {
            "available": False,
            "reason": "Fewer than 10 Phase II/III interventional rows in a temporal split.",
        }
    else:
        print("fitting Phase II/III specialist", file=sys.stderr, flush=True)
        specialist_selected, _ = _fit_one(
            best_params,
            X[specialist_split["train"]],
            y[specialist_split["train"]],
            X[specialist_split["validation"]],
            y[specialist_split["validation"]],
            n_estimators,
            seed=seed,
            early_stopping_rounds=early_stopping_rounds,
        )
        specialist_trees = _best_iteration(specialist_selected, n_estimators)
        specialist_validation = specialist_selected.predict(X[specialist_split["validation"]])
        specialist_calibration = _fit_calibrator(
            y[specialist_split["validation"]], specialist_validation
        )
        specialist_pre_metrics = _metrics(
            y[specialist_split["test"]],
            specialist_selected.predict(X[specialist_split["test"]]),
            specialist_calibration,
        )
        specialist_refit_idx = np.sort(
            np.concatenate([specialist_split["train"], specialist_split["validation"]])
        )
        specialist_model = _fit_refit_model(
            best_params,
            X[specialist_refit_idx],
            y[specialist_refit_idx],
            specialist_trees,
            seed,
        )
        specialist_artifact_metrics = _metrics(
            y[specialist_split["test"]],
            specialist_model.predict(X[specialist_split["test"]]),
            specialist_calibration,
        )
        specialist_meta = {
            "available": True,
            "artifact": "duration_model_phase23.json",
            "routing": {
                "study_type": "INTERVENTIONAL",
                "phase_num_in": [2, 3],
                "fallback": "generic_stack",
            },
            "prediction_pipeline": "xgb_log_then_specialist_calibration",
            "calibration": _round_floats(specialist_calibration),
            "counts": {name: int(len(idx)) for name, idx in specialist_split.items()},
            "n_estimators_used": specialist_trees,
            "pre_refit_test_metrics": _round_metrics(specialist_pre_metrics),
            "artifact_test_metrics": _round_metrics(specialist_artifact_metrics),
        }

    emb_dim = X.shape[1] - n_struct
    names = list(STRUCT_FEATURES) + [f"emb_{i}" for i in range(emb_dim)]
    gain = final_model.get_booster().get_score(importance_type="gain")
    named_gain = {
        names[int(key[1:])]: float(value)
        for key, value in gain.items()
        if key.startswith("f") and int(key[1:]) < len(names)
    }
    top_gain = sorted(named_gain.items(), key=lambda pair: pair[1], reverse=True)[:20]

    fingerprint = _dataset_fingerprint(rows)
    split_manifest = _manifest(
        rows,
        random_split,
        temporal_split,
        group_split,
        rolling_folds,
        test_idx,
        fingerprint,
        seed,
    )
    provenance = {
        **_git_provenance(),
        "dataset_sha256": fingerprint,
        "dataset_source": "sqlite.trials",
        "dataset_filter": (
            "duration_months in (1,240), enrollment non-null, status=COMPLETED, "
            "start/completion date types ACTUAL or legacy NULL"
        ),
        "label_date_type_counts": _label_date_type_audit(rows),
        "completion_conditioned": True,
        "design_time_snapshot_available": bool(
            coverage["source"]["snapshot_type_counts"].get("INITIAL", 0)
        ),
        "registry_snapshot": (
            "initial_or_versioned_features_with_current_actual_labels"
            if coverage["source"]["snapshot_type_counts"].get("INITIAL", 0)
            else "latest_record_at_fetch_time"
        ),
        "split_manifest_sha256": split_manifest["manifest_sha256"],
    }

    stacked_artifact_metrics = production_refit_metrics["stacked"]
    text_artifact_meta = {
        "artifact": "duration_text_ridge.npz",
        "alpha": best_alpha,
        "embedding_dim": emb_dim,
        "target": "log1p_duration_months",
    }
    meta = {
        "schema_version": 3,
        "n_rows": int(len(X)),
        "n_rows_production_eligible": int(len(eligible)),
        "r2_log": round(float(stacked_artifact_metrics["log"]["r2"]), 4),
        "r2_raw": round(float(stacked_artifact_metrics["raw_calibrated"]["r2"]), 4),
        "r2_raw_uncalibrated": round(
            float(stacked_artifact_metrics["raw_uncalibrated"]["r2"]), 4
        ),
        "r2_raw_with_baseline": round(
            float(
                production_refit_metrics["stacked_with_neighbor_baseline_raw"]["r2"]
            ),
            4,
        ),
        "val_r2_log": round(
            float(temporal_validation_metrics["stacked"]["log"]["r2"]), 4
        ),
        "timestamp": time.time(),
        "synthetic": synthetic,
        "train_seconds": round(time.time() - started, 1),
        "params": best_params,
        "selection_strategy": "rolling_origin",
        "selection_metric": selection_metric,
        "selection_score": round(best_score, 6),
        "n_estimators_budget": n_estimators,
        "n_estimators_used": best_trees,
        "struct_features": list(STRUCT_FEATURES),
        "feature_order": names,
        "emb_dim": int(emb_dim),
        "emb_model": MODEL_NAME,
        "embedding_recipe_version": 2,
        "deep": deep,
        "heavy": deep,
        "xgb_device": _xgb_device(),
        "calibration": _round_floats(calibration),
        "stacking": {
            "prediction_space": "log1p_duration_months",
            "component_order": ["xgb_log", "text_log"],
            "weights": _round_floats(stack_blend["weights"]),
            "intercept": round(float(stack_blend["intercept"]), 6),
            "nonnegative_weights": True,
            "fitted_on": f"temporal_validation_{latest_fold['validation_year']}",
            "text_head": text_artifact_meta,
            "serving_order": [
                "xgb_log_and_text_ridge_log",
                "nonnegative_log_stack",
                "expm1_and_calibration",
                "optional_baseline_blend",
            ],
        },
        "baseline_blend": _round_floats(baseline_blend),
        "neighbor_baseline": {
            "validation": neighbor.metadata(),
            "test": test_neighbor.metadata(),
            "leakage_control": "validation queries train-only; test queries train+validation",
        },
        "coverage_preflight": coverage,
        "specialists": {"phase23_interventional": specialist_meta},
        "evaluations": {
            "mature_temporal": {
                **rolling_details,
                "primary_validation_year": latest_fold["validation_year"],
                "counts": {name: int(len(idx)) for name, idx in temporal_split.items()},
                "validation_metrics": temporal_validation_metrics,
                "pre_refit_test_metrics": temporal_test_metrics,
                "artifact_test_metrics": production_refit_metrics,
                "interpretation": (
                    "Primary score: mature rolling-origin selection with an untouched "
                    "test year; still conditioned on completed trials."
                ),
            },
            "legacy_random": {
                "strategy": "deterministic_random_80_10_10_comparison_only",
                "counts": {name: int(len(idx)) for name, idx in random_split.items()},
                "metrics": legacy_metrics,
                "influenced_selection": False,
                "group_leakage_audit": _group_leakage_audit(rows, random_split),
            },
            "sponsor_grouped": grouped_evaluation,
        },
        "search_results": _round_floats(search_results),
        "ridge_search_results": ridge_search,
        "refit": {
            "strategy": "latest_temporal_train_plus_validation",
            "n_rows": int(len(reference_idx)),
            "test_year": rolling_details["test_year"],
            "test_rows_excluded": int(len(test_idx)),
        },
        "maturity": {
            "comparison_horizon_years": maturity_years,
            "comparison_start_year_cutoff": maturity_cutoff,
        },
        "provenance": provenance,
        "top_gain_features": _round_floats(top_gain),
        "limitations": [
            "Labels currently exist only for completed trials; active trials are not modeled as right-censored.",
            "The legacy random score is not a prospective performance estimate.",
            "The source contains latest registry records rather than registration-time feature snapshots.",
        ],
    }

    if save:
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        _save_xgb_atomic(final_model, MODELS_DIR / "duration_model.json")
        ridge_artifact_meta = final_text_head.save(MODELS_DIR / "duration_text_ridge.npz")
        meta["stacking"]["text_head"].update(ridge_artifact_meta)
        meta["artifacts"] = {
            "generic_xgb": {
                "artifact": "duration_model.json",
                "artifact_sha256": _file_sha256(MODELS_DIR / "duration_model.json"),
            },
            "text_ridge": ridge_artifact_meta,
        }
        if specialist_model is not None:
            _save_xgb_atomic(
                specialist_model, MODELS_DIR / "duration_model_phase23.json"
            )
            specialist_sha = _file_sha256(
                MODELS_DIR / "duration_model_phase23.json"
            )
            specialist_meta["artifact_sha256"] = specialist_sha
            meta["artifacts"]["phase23_interventional_xgb"] = {
                "artifact": "duration_model_phase23.json",
                "artifact_sha256": specialist_sha,
            }
        _write_json_atomic(MODELS_DIR / "train_meta.json", meta)
        _write_json_atomic(
            MODELS_DIR / "training_split_manifest.json", split_manifest, compact=True
        )

    return meta


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="explicit synthetic demo mode")
    parser.add_argument(
        "--allow-demo-save",
        action="store_true",
        help="explicitly allow --quick to write synthetic model artifacts",
    )
    parser.add_argument("--deep", action="store_true", help="use the deep search grid")
    parser.add_argument("--no-save", action="store_true", help="do not write model artifacts")
    args = parser.parse_args()
    result = train(
        save=not args.no_save,
        quick=args.quick,
        deep=args.deep or None,
        allow_demo_save=args.allow_demo_save,
    )
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    _main()
