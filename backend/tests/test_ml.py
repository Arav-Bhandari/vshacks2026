import pytest

pytest.importorskip("xgboost")

from app.ml.features import FEATURE_ORDER, trial_row_features, usdm_features
from app.ml.train import train
from app.ml.predictor import predict_duration_risk

USDM = {
    "study": {
        "phase": "PHASE2",
        "population": {
            "plannedEnrollment": 200,
            "criteria": {"inclusion": ["a", "b", "c"], "exclusion": ["d", "e"]},
        },
        "arms": [{"name": "arm1"}, {"name": "arm2"}],
        "objectives": [
            {"level": "primary", "endpoints": ["e1"]},
            {"level": "secondary", "endpoints": ["e2", "e3"]},
        ],
        "scheduleOfActivities": {"visits": [{}] * 5},
        "design": {"allocation": "randomized", "masking": "double"},
        "conditions": ["cond1"],
        "interventions": ["drug1", "drug2"],
    }
}

TRIAL_ROW = {
    "phase": "PHASE2",
    "enrollment": 200,
    "arms": 2,
    "primary_outcomes": "e1",
    "secondary_outcomes": "e2; e3",
    "eligibility": "a; b; c Exclusion Criteria: d; e",
    "allocation": "randomized",
    "masking": "double",
    "conditions": "cond1",
    "interventions": "drug1; drug2",
}


def test_features_same_keys_and_order():
    f1 = usdm_features(USDM)
    f2 = trial_row_features(TRIAL_ROW)
    assert list(f1.keys()) == FEATURE_ORDER
    assert list(f2.keys()) == FEATURE_ORDER


def test_train_and_predict(tmp_path, monkeypatch):
    import app.ml.train as train_mod
    import app.ml.predictor as predictor_mod

    monkeypatch.setattr(train_mod, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(predictor_mod, "MODELS_DIR", tmp_path)
    predictor_mod._cache.clear()

    meta = train()
    assert meta["n_rows"] > 0

    result = predict_duration_risk(USDM, baseline={"expected_duration_months": 18})

    assert set(result.keys()) == {
        "predicted_duration_months",
        "overrun_risk_pct",
        "baseline_duration_months",
        "shap_top5",
        "model_meta",
    }
    assert 0 <= result["overrun_risk_pct"] <= 100
    assert len(result["shap_top5"]) <= 5
    for item in result["shap_top5"]:
        assert set(item.keys()) == {"feature", "impact", "direction", "explanation"}
