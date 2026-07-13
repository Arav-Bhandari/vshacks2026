"""Serve duration predictions and SHAP values."""
import json
import hashlib
import math
from pathlib import Path

import numpy as np
import shap
import xgboost as xgb

from app.config import MODELS_DIR
from app.ml.embeddings import build_text, embed_one
from app.ml.features import STRUCT_FEATURES, usdm_features, to_vector
from app.ml.stacking import RidgeTextHead

_EXPLANATIONS = {
    "phase_num": "Later phases run longer trials",
    "log_enrollment": "Larger enrollment slows recruitment",
    "n_arms": "More arms adds operational time",
    "n_endpoints_primary": "More primary endpoints adds analysis time",
    "n_endpoints_secondary": "More secondary endpoints adds follow-up burden",
    "has_secondary_outcomes": "Secondary outcomes extend follow-up",
    "n_inclusion": "Stricter inclusion criteria slows recruitment",
    "n_exclusion": "More exclusion criteria slows recruitment",
    "randomized": "Randomization adds setup overhead",
    "blinded": "Blinding adds coordination overhead",
    "n_conditions": "Multiple conditions complicate the protocol",
    "n_interventions": "More interventions adds logistics",
    "is_drug": "Drug intervention",
    "is_biological": "Biological intervention adds handling steps",
    "is_device": "Device intervention adds regulatory steps",
    "is_behavioral": "Behavioral intervention affects visit cadence",
    "is_procedure": "Procedural intervention affects scheduling",
    "is_other_intervention": "Uncommon intervention type",
    "is_academic": "Academic sponsorship affects pacing",
    "start_year": "Start year captures era effects",
    "planned_duration_months": "The protocol's planned follow-up informs total duration",
    "primary_outcome_timeframe_months": "Longer primary endpoint follow-up extends the trial",
    "secondary_outcome_timeframe_months": "Longer secondary endpoint follow-up extends the trial",
    "n_visits": "More scheduled visits add operational time",
    "n_procedures": "More procedures increase operational complexity",
    "n_sites": "Site count changes startup and recruitment pacing",
    "n_countries": "More countries add regulatory and operational coordination",
    "age_range_years": "The eligible age range affects recruitment",
    "intervention_model_missing": "An unspecified intervention model increases uncertainty",
}

_CONDITION_PROFILE_EXPLANATION = "Disease area and intervention profile"

_REFERENCE_DURATION = 24.0

_cache: dict = {}
_SUPPORTED_SCHEMA_VERSION = 3
_MAX_DURATION_MONTHS = 240.0


def _model_path():
    return MODELS_DIR / "duration_model.json"


def _meta_path():
    return MODELS_DIR / "train_meta.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_artifact(path: Path, expected_sha: str | None = None) -> None:
    if not path.exists():
        raise FileNotFoundError(f"required duration-model artifact not found: {path}")
    if expected_sha and _sha256(path) != expected_sha:
        raise ValueError(f"duration-model artifact checksum mismatch: {path.name}")


def _artifact_signature(paths: list[Path]) -> tuple:
    return tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths if path.exists()
    )


def _load_xgb(path: Path) -> xgb.XGBRegressor:
    model = xgb.XGBRegressor()
    model.load_model(str(path))
    model.set_params(device="cpu")
    model.get_booster().set_param({"device": "cpu"})
    return model


def _load_model():
    meta_path = _meta_path()
    if not meta_path.exists():
        raise FileNotFoundError(
            f"duration model metadata not found at {meta_path}; run the training command "
            "and publish the complete artifact set"
        )
    with open(meta_path) as file:
        meta = json.load(file)
    if int(meta.get("schema_version", -1)) != _SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported duration model schema {meta.get('schema_version')}; "
            f"expected {_SUPPORTED_SCHEMA_VERSION}"
        )

    artifacts = meta.get("artifacts") or {}
    generic_info = artifacts.get("generic_xgb") or {}
    ridge_info = artifacts.get("text_ridge") or meta.get("stacking", {}).get("text_head") or {}
    model_path = MODELS_DIR / str(generic_info.get("artifact") or "duration_model.json")
    ridge_path = MODELS_DIR / str(ridge_info.get("artifact") or "duration_text_ridge.npz")
    required = [meta_path, model_path, ridge_path]

    specialist_info = artifacts.get("phase23_interventional_xgb") or {}
    specialist_meta = (meta.get("specialists") or {}).get("phase23_interventional") or {}
    specialist_path = None
    if specialist_meta.get("available"):
        specialist_path = MODELS_DIR / str(
            specialist_info.get("artifact") or specialist_meta.get("artifact")
        )
        required.append(specialist_path)

    signature = _artifact_signature(required)
    if _cache.get("signature") == signature:
        return _cache["model"], _cache["meta"]

    _require_artifact(model_path, generic_info.get("artifact_sha256"))
    _require_artifact(ridge_path, ridge_info.get("artifact_sha256"))
    model = _load_xgb(model_path)
    ridge = RidgeTextHead.load(ridge_path)
    specialist = None
    if specialist_path is not None:
        _require_artifact(
            specialist_path,
            specialist_info.get("artifact_sha256") or specialist_meta.get("artifact_sha256"),
        )
        specialist = _load_xgb(specialist_path)

    _validate_model_schema(model, meta)
    if len(ridge.coef_) != int(meta["emb_dim"]):
        raise ValueError("text Ridge artifact embedding dimension does not match metadata")
    if specialist is not None and specialist.get_booster().num_features() != model.get_booster().num_features():
        raise ValueError("Phase II/III specialist feature schema does not match generic model")

    _cache.clear()
    _cache.update(
        signature=signature,
        model=model,
        meta=meta,
        ridge=ridge,
        specialist=specialist,
        explainer=shap.TreeExplainer(model),
        specialist_explainer=(shap.TreeExplainer(specialist) if specialist is not None else None),
    )
    return model, meta


def _validate_model_schema(model, meta: dict) -> None:
    """Validate the model feature schema."""
    required = {
        "schema_version", "struct_features", "feature_order", "emb_dim",
        "emb_model", "embedding_recipe_version", "stacking", "calibration", "artifacts",
    }
    missing = sorted(required - set(meta))
    if missing:
        raise ValueError(f"duration model metadata is incomplete: {', '.join(missing)}")
    meta_struct = meta["struct_features"]
    if list(meta_struct) != STRUCT_FEATURES:
        raise ValueError(
            "duration model structured feature schema does not match serving code; retrain the model"
        )

    model_dim = int(model.get_booster().num_features())
    inferred_emb_dim = model_dim - len(STRUCT_FEATURES)
    try:
        expected_emb_dim = int(meta["emb_dim"])
    except (TypeError, ValueError):
        raise ValueError("duration model metadata contains an invalid embedding dimension") from None
    expected_order = STRUCT_FEATURES + [f"emb_{i}" for i in range(expected_emb_dim)]
    meta_order = meta["feature_order"]
    if list(meta_order) != expected_order:
        raise ValueError("duration model feature order does not match serving code; retrain the model")

    expected_dim = len(STRUCT_FEATURES) + expected_emb_dim
    if expected_emb_dim <= 0 or model_dim != expected_dim:
        raise ValueError(
            f"duration model expects {model_dim} features but metadata specifies {expected_dim}; "
            "retrain the model"
        )
    if int(meta["embedding_recipe_version"]) != 2:
        raise ValueError("unsupported embedding text recipe; retrain the model")
    if not str(meta["emb_model"]).strip():
        raise ValueError("duration model metadata does not name its embedding encoder")


def _positive_number(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) and number > 0 else None


def _calibrate(raw_prediction: float, meta: dict) -> float:
    calibration = meta.get("calibration") or {}
    if calibration.get("method") == "affine_raw":
        try:
            slope = float(calibration.get("slope", 1.0))
            intercept = float(calibration.get("intercept", 0.0))
            calibrated = slope * raw_prediction + intercept
            if math.isfinite(calibrated):
                return max(1.0, calibrated)
        except (TypeError, ValueError):
            pass

    if calibration.get("method") == "smearing":
        try:
            smearing = float(calibration.get("factor", calibration.get("smearing_factor", 1.0)))
            corrected = (raw_prediction + 1.0) * smearing - 1.0
            if math.isfinite(corrected) and smearing > 0:
                return max(1.0, corrected)
        except (TypeError, ValueError):
            pass

    try:
        smearing = float(meta.get("smearing_factor", 1.0))
    except (TypeError, ValueError):
        smearing = 1.0
    if not math.isfinite(smearing) or smearing <= 0:
        smearing = 1.0
    return max(1.0, (raw_prediction + 1.0) * smearing - 1.0)


def _blend_with_baseline(prediction: float, baseline_duration: float | None, meta: dict) -> float:
    blend = meta.get("baseline_blend") or {}
    if baseline_duration is None or not blend:
        return prediction
    try:
        model_weight = float(blend.get("model_weight", 1.0))
        baseline_weight = float(blend.get("baseline_weight", 0.0))
        intercept = float(blend.get("intercept", 0.0))
        blended = model_weight * prediction + baseline_weight * baseline_duration + intercept
    except (TypeError, ValueError):
        return prediction
    return max(1.0, blended) if math.isfinite(blended) else prediction


def _log_to_raw(prediction_log: float) -> float:
    if not math.isfinite(prediction_log):
        raise ValueError("duration model produced a non-finite log prediction")
    bounded = min(max(0.0, prediction_log), math.log1p(_MAX_DURATION_MONTHS))
    return float(math.expm1(bounded))


def _use_phase23_specialist(struct: dict, meta: dict) -> bool:
    specialist = (meta.get("specialists") or {}).get("phase23_interventional") or {}
    if not specialist.get("available"):
        return False
    phase = int(round(float(struct.get("phase_num", 0))))
    return phase in {2, 3} and float(struct.get("is_interventional", 0)) > 0.5


def _generic_log_prediction(model, x: np.ndarray, meta: dict) -> float:
    xgb_log = float(model.predict(x)[0])
    text_head = _cache.get("ridge")
    if text_head is None:
        raise RuntimeError("duration text head is not loaded")
    text_log = float(text_head.predict(x[:, len(STRUCT_FEATURES):])[0])
    stacking = meta["stacking"]
    weights = stacking.get("weights") or {}
    if set(weights) != {"xgb_log", "text_log"}:
        raise ValueError("duration stacking metadata has an invalid component set")
    prediction = (
        float(stacking.get("intercept", 0.0))
        + float(weights["xgb_log"]) * xgb_log
        + float(weights["text_log"]) * text_log
    )
    if not math.isfinite(prediction):
        raise ValueError("duration stack produced a non-finite prediction")
    return prediction


def _build_vector(usdm: dict, burden: dict | None, meta: dict) -> tuple[np.ndarray, dict]:
    study = (usdm or {}).get("study") or {}
    primary_outcomes, secondary_outcomes = [], []
    primary_timeframes, secondary_timeframes = [], []
    for objective in study.get("objectives") or []:
        if not isinstance(objective, dict):
            continue
        secondary = "secondary" in str(objective.get("level") or "").lower()
        outcomes = secondary_outcomes if secondary else primary_outcomes
        timeframes = secondary_timeframes if secondary else primary_timeframes
        for endpoint in objective.get("endpoints") or []:
            outcomes.append(endpoint)
            if isinstance(endpoint, dict):
                timeframe = endpoint.get("timeframe") or endpoint.get("timeFrame")
                if timeframe:
                    timeframes.append(timeframe)
    text = build_text(
        study.get("conditions"),
        study.get("interventions"),
        study.get("name") or study.get("title"),
        primary_outcomes=primary_outcomes,
        secondary_outcomes=secondary_outcomes,
        primary_outcome_timeframes=primary_timeframes,
        secondary_outcome_timeframes=secondary_timeframes,
    )
    emb = embed_one(text, meta.get("emb_model"))
    if meta.get("emb_dim") is not None and len(emb) != int(meta["emb_dim"]):
        raise ValueError(
            f"embedding encoder returned {len(emb)} dimensions; model expects {meta['emb_dim']}"
        )
    struct = usdm_features(usdm, burden)
    return np.array([to_vector(struct, emb)], dtype="float32"), struct


def predict_duration_risk(usdm: dict, baseline: dict | None = None, burden: dict | None = None) -> dict:
    model, meta = _load_model()

    x, struct = _build_vector(usdm, burden, meta)
    specialist_used = _use_phase23_specialist(struct, meta)
    if specialist_used:
        prediction_model = _cache.get("specialist")
        if prediction_model is None:
            raise RuntimeError("Phase II/III specialist declared but not loaded")
        prediction_log = float(prediction_model.predict(x)[0])
        calibration_meta = (
            meta["specialists"]["phase23_interventional"].get("calibration") or {}
        )
        predicted = _calibrate(
            _log_to_raw(prediction_log), {"calibration": calibration_meta}
        )
        explainer = _cache["specialist_explainer"]
    else:
        prediction_model = model
        prediction_log = _generic_log_prediction(model, x, meta)
        predicted = _calibrate(_log_to_raw(prediction_log), meta)
        explainer = _cache["explainer"]

    baseline_duration = _positive_number(
        baseline.get("expected_duration_months") if baseline else None
    )
    predicted = _blend_with_baseline(predicted, baseline_duration, meta)
    predicted = min(_MAX_DURATION_MONTHS, max(1.0, predicted))

    reference = baseline_duration or _REFERENCE_DURATION
    relative_excess = (predicted - reference) / reference
    overrun_risk_pct = int(round(100 / (1 + math.exp(-4 * relative_excess))))
    overrun_risk_pct = max(0, min(100, overrun_risk_pct))

    shap_values = explainer.shap_values(x)[0]

    n_struct = len(STRUCT_FEATURES)
    struct_pairs = list(zip(STRUCT_FEATURES, shap_values[:n_struct]))
    condition_profile = float(np.sum(shap_values[n_struct:]))
    struct_pairs.append(("condition_profile", condition_profile))

    ranked = sorted(struct_pairs, key=lambda p: abs(p[1]), reverse=True)[:5]
    shap_top5 = [
        {
            "feature": name,
            "impact": round(float(val), 3),
            "direction": "increases" if val >= 0 else "decreases",
            "explanation": _EXPLANATIONS.get(
                name,
                _CONDITION_PROFILE_EXPLANATION if name == "condition_profile"
                else name.replace("_", " ").capitalize(),
            ),
        }
        for name, val in ranked
    ]

    return {
        "predicted_duration_months": round(predicted, 1),
        "overrun_risk_pct": overrun_risk_pct,
        "baseline_duration_months": baseline_duration,
        "shap_top5": shap_top5,
        "model_meta": meta,
    }
