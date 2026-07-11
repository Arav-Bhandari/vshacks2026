"""Duration prediction + overrun risk + SHAP explanation for a protocol."""
import json
import math

import numpy as np
import shap
import xgboost as xgb

from app.config import MODELS_DIR
from app.ml.features import FEATURE_ORDER, usdm_features
from app.ml.train import train

_EXPLANATIONS = {
    "phase_num": "Later phases run longer trials",
    "enrollment": "Larger enrollment slows recruitment",
    "n_arms": "More arms adds operational time",
    "n_endpoints_primary": "More primary endpoints adds analysis time",
    "n_endpoints_secondary": "More secondary endpoints adds follow-up burden",
    "n_inclusion": "Stricter inclusion criteria slows recruitment",
    "n_exclusion": "More exclusion criteria slows recruitment",
    "n_visits": "More visits extends the schedule",
    "randomized": "Randomization adds setup overhead",
    "blinded": "Blinding adds coordination overhead",
    "n_conditions": "Multiple conditions complicate the protocol",
    "n_interventions": "More interventions adds logistics",
}

_REFERENCE_DURATION = 24.0

_cache: dict = {}


def _model_path():
    return MODELS_DIR / "duration_model.json"


def _meta_path():
    return MODELS_DIR / "train_meta.json"


def _load_model():
    if "model" in _cache:
        return _cache["model"], _cache["meta"]

    if not _model_path().exists():
        train()

    model = xgb.XGBRegressor()
    model.load_model(str(_model_path()))

    meta = {}
    if _meta_path().exists():
        with open(_meta_path()) as f:
            meta = json.load(f)

    _cache["model"] = model
    _cache["meta"] = meta
    return model, meta


def predict_duration_risk(usdm: dict, baseline: dict | None = None, burden: dict | None = None) -> dict:
    model, meta = _load_model()

    feats = usdm_features(usdm, burden)
    x = np.array([[feats[k] for k in FEATURE_ORDER]], dtype=float)

    predicted = float(model.predict(x)[0])
    predicted = max(1.0, predicted)

    baseline_duration = None
    if baseline:
        baseline_duration = baseline.get("expected_duration_months")

    reference = baseline_duration or _REFERENCE_DURATION
    relative_excess = (predicted - reference) / reference
    overrun_risk_pct = int(round(100 / (1 + math.exp(-4 * relative_excess))))
    overrun_risk_pct = max(0, min(100, overrun_risk_pct))

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x)[0]

    ranked = sorted(
        zip(FEATURE_ORDER, shap_values), key=lambda p: abs(p[1]), reverse=True
    )[:5]
    shap_top5 = [
        {
            "feature": name,
            "impact": round(float(val), 3),
            "direction": "increases" if val >= 0 else "decreases",
            "explanation": _EXPLANATIONS.get(name, name),
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
