"""Duration prediction + overrun risk + SHAP explanation for a protocol."""
import json
import math

import numpy as np
import shap
import xgboost as xgb

from app.config import MODELS_DIR
from app.ml.embeddings import build_text, embed_one
from app.ml.features import FEATURE_ORDER, STRUCT_FEATURES, usdm_features, to_vector
from app.ml.train import train

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
}

_CONDITION_PROFILE_EXPLANATION = "Disease area and intervention profile"

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
        train(quick=True)

    model = xgb.XGBRegressor()
    model.load_model(str(_model_path()))

    meta = {}
    if _meta_path().exists():
        with open(_meta_path()) as f:
            meta = json.load(f)

    _cache["model"] = model
    _cache["meta"] = meta
    return model, meta


def _build_vector(usdm: dict, burden: dict | None, meta: dict) -> np.ndarray:
    study = (usdm or {}).get("study") or {}
    text = build_text(
        study.get("conditions"), study.get("interventions"), study.get("name") or study.get("title")
    )
    emb = embed_one(text, meta.get("emb_model"))
    struct = usdm_features(usdm, burden)
    return np.array([to_vector(struct, emb)], dtype="float32")


def predict_duration_risk(usdm: dict, baseline: dict | None = None, burden: dict | None = None) -> dict:
    model, meta = _load_model()

    x = _build_vector(usdm, burden, meta)
    predicted = float(np.expm1(model.predict(x)[0]))
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

    n_struct = len(STRUCT_FEATURES)
    struct_pairs = list(zip(FEATURE_ORDER[:n_struct], shap_values[:n_struct]))
    condition_profile = float(np.sum(shap_values[n_struct:]))
    struct_pairs.append(("condition_profile", condition_profile))

    ranked = sorted(struct_pairs, key=lambda p: abs(p[1]), reverse=True)[:5]
    shap_top5 = [
        {
            "feature": name,
            "impact": round(float(val), 3),
            "direction": "increases" if val >= 0 else "decreases",
            "explanation": _EXPLANATIONS.get(name, _CONDITION_PROFILE_EXPLANATION),
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
