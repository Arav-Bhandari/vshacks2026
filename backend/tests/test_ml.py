import json

import numpy as np
import pytest

pytest.importorskip("xgboost")

from app.ml.embeddings import build_text
from app.ml.features import (
    FEATURE_ORDER,
    STRUCT_FEATURES,
    _canonical_row,
    trial_row_features,
    usdm_features,
)
from app.ml.train import train
from app.ml.predictor import predict_duration_risk
from app.ml.stacking import (
    NeighborDurationBaseline,
    RidgeTextHead,
    apply_blend,
    fit_nonnegative_blend,
)

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
    "interventions": "DRUG: drug1; DRUG: drug2",
    "sponsor": "State University",
    "start_date": "2015-01-01",
}


def test_features_same_keys_and_order():
    f1 = usdm_features(USDM)
    f2 = trial_row_features(TRIAL_ROW)
    assert list(f1.keys()) == STRUCT_FEATURES
    assert list(f2.keys()) == STRUCT_FEATURES
    assert len(FEATURE_ORDER) == len(STRUCT_FEATURES) + 384
    assert not {
        "start_date_is_estimated",
        "n_visits",
        "visits_missing",
        "n_procedures",
        "procedures_missing",
        "planned_duration_months",
        "planned_duration_missing",
    } & set(STRUCT_FEATURES)


def test_db_row_and_usdm_have_identical_features_for_same_design():
    row = {
        "phase": "PHASE3",
        "enrollment": 240,
        "enrollment_type": "ANTICIPATED",
        "arms": 2,
        "primary_outcomes": "Overall response; Progression-free survival",
        "secondary_outcomes": "Quality of life",
        "primary_outcome_timeframes": "6 months; 9 months",
        "secondary_outcome_timeframes": "12 months",
        "eligibility": (
            "Inclusion Criteria:\n- age 18 or older\n- confirmed disease\n"
            "Exclusion Criteria:\n- prior therapy"
        ),
        "allocation": "RANDOMIZED",
        "masking": "DOUBLE",
        "conditions": "Condition A, Condition B",
        "interventions": "DRUG: Study drug; BIOLOGICAL: Comparator",
        "sponsor": "Example Pharma",
        "sponsor_class": "INDUSTRY",
        "start_date": "2026-01",
        "start_date_type": "ESTIMATED",
        "primary_completion_date": "2027-01",
        "primary_completion_date_type": "ESTIMATED",
        "study_type": "INTERVENTIONAL",
        "primary_purpose": "TREATMENT",
        "intervention_model": "PARALLEL",
        "sex": "ALL",
        "minimum_age": "18 Years",
        "maximum_age": "75 Years",
        "healthy_volunteers": "No",
        "site_count": 20,
        "country_count": 3,
        "collaborators_count": 2,
        "visit_count": 2,
        "procedure_count": 3,
    }
    usdm = {
        "study": {
            "phase": "Phase 3",
            "studyType": "INTERVENTIONAL",
            "primaryPurpose": "TREATMENT",
            "conditions": ["Condition A", "Condition B"],
            "interventions": [
                {"name": "Study drug", "type": "DRUG"},
                {"name": "Comparator", "type": "BIOLOGICAL"},
            ],
            "population": {
                "plannedEnrollment": 240,
                "criteria": {
                    "inclusion": ["age 18 or older", "confirmed disease"],
                    "exclusion": ["prior therapy"],
                },
                "minimumAge": "18 Years",
                "maximumAge": "75 Years",
                "sex": "ALL",
                "healthyVolunteers": False,
            },
            "arms": [{"name": "A"}, {"name": "B"}],
            "objectives": [
                {
                    "level": "primary",
                    "endpoints": [
                        {"name": "Overall response", "timeframe": "6 months"},
                        {"name": "Progression-free survival", "timeframe": "9 months"},
                    ],
                },
                {
                    "level": "secondary",
                    "endpoints": [{"name": "Quality of life", "timeframe": "12 months"}],
                },
            ],
            "design": {
                "allocation": "RANDOMIZED",
                "masking": "DOUBLE",
                "interventionModel": "PARALLEL",
            },
            "sponsor": {"name": "Example Pharma", "class": "INDUSTRY"},
            "startDate": "2026-01",
            "startDateType": "ESTIMATED",
            "sites": [{"name": f"site {i}"} for i in range(20)],
            "countries": ["US", "CA", "GB"],
            "collaborators": [{"name": "A"}, {"name": "B"}],
            "scheduleOfActivities": {
                "visits": [
                    {"name": "baseline", "procedures": ["blood draw"]},
                    {"name": "follow-up", "procedures": ["scan", "survey"]},
                ]
            },
            "estimatedDuration": "12 months",
        }
    }

    assert trial_row_features(row) == usdm_features(usdm)


def test_actual_completion_date_is_not_a_planned_duration_feature():
    row = {
        "start_date": "2020-01",
        "primary_completion_date": "2024-01",
        "primary_completion_date_type": "ACTUAL",
        "completion_date": "2025-01",
        "completion_date_type": "ACTUAL",
    }
    assert _canonical_row(row)["planned_duration"] is None
    assert "planned_duration_months" not in trial_row_features(row)


def test_actual_enrollment_is_not_used_as_a_design_feature():
    actual = trial_row_features({"enrollment": 10000, "enrollment_type": "ACTUAL"})
    planned = trial_row_features({"enrollment": 250, "enrollment_type": "ANTICIPATED"})
    assert actual["log_enrollment"] == 0
    assert actual["enrollment_missing"] == 1
    assert planned["log_enrollment"] > 0
    assert planned["enrollment_missing"] == 0


def test_embedding_text_extracts_intervention_dict_names():
    text = build_text(
        [{"name": "glioblastoma"}],
        [{"name": "temozolomide", "type": "DRUG"}, {"name": "radiation"}],
        "Combination protocol",
    )
    assert text == "glioblastoma. temozolomide, radiation. Combination protocol"
    assert "{'" not in text


def test_embedding_text_recipe_matches_db_and_usdm_values():
    row_text = build_text(
        "Condition A, Condition B",
        "DRUG: Study drug; BIOLOGICAL: Comparator",
        "Trial title",
        "Response; Survival",
        "Quality of life",
        "6 months; 9 months",
        "12 months",
    )
    usdm_text = build_text(
        ["Condition A", "Condition B"],
        [{"name": "Study drug", "type": "DRUG"}, {"name": "Comparator", "type": "BIOLOGICAL"}],
        "Trial title",
        [{"name": "Response"}, {"name": "Survival"}],
        [{"name": "Quality of life"}],
        ["6 months", "9 months"],
        ["12 months"],
    )
    assert row_text == usdm_text


def test_structured_db_arrays_preserve_boundaries_and_precomputed_counts():
    features = trial_row_features({
        "conditions": "legacy, value, would, overcount",
        "conditions_json": json.dumps(["Diabetes Mellitus, Type 2"]),
        "interventions": "legacy; values",
        "interventions_json": json.dumps([
            {"type": "DRUG", "name": "Drug; combination"}
        ]),
        "primary_outcomes": "legacy; outcome; would; overcount",
        "primary_outcomes_json": json.dumps(["Measure; with internal delimiter"]),
        "secondary_outcomes_json": "[]",
        "eligibility": "Inclusion Criteria:\n- legacy\nExclusion Criteria:\n- legacy",
        "inclusion_criteria_count": 7,
        "exclusion_criteria_count": 4,
    })

    assert features["n_conditions"] == 1
    assert features["n_interventions"] == 1
    assert features["n_endpoints_primary"] == 1
    assert features["n_endpoints_secondary"] == 0
    assert features["n_inclusion"] == 7
    assert features["n_exclusion"] == 4


def test_blank_usdm_template_records_are_missing_not_observations():
    features = usdm_features({
        "study": {
            "studyType": "INTERVENTIONAL|OBSERVATIONAL|",
            "phase": "",
            "arms": [{"name": "", "type": "", "description": ""}],
            "objectives": [{
                "level": "primary|secondary",
                "endpoints": [{"name": "", "description": "", "timeframe": ""}],
            }],
            "population": {
                "criteria": {"inclusion": [], "exclusion": []},
                "healthyVolunteers": False,
            },
            "sites": [{"name": "", "country": ""}],
            "countries": [],
            "collaborators": [{"name": "", "class": ""}],
            "scheduleOfActivities": {
                "visits": [{"name": "", "timing": "", "procedures": []}]
            },
        }
    })

    assert features["arms_missing"] == 1
    assert features["endpoints_missing"] == 1
    assert features["eligibility_missing"] == 1
    assert features["study_type_missing"] == 1
    assert features["sites_missing"] == 1
    assert features["countries_missing"] == 1
    assert features["collaborators_missing"] == 1
    assert features["healthy_volunteers"] == 0
    assert features["healthy_volunteers_missing"] == 0


@pytest.mark.parametrize(
    ("timeframe", "expected_months"),
    [
        ("Week 4 through Week 16", 16 * 7 / 30.4375),
        ("Baseline to Month 12", 12),
        ("Day 0 through Day 221", 221 / 30.4375),
        ("Year 2", 24),
        ("Weeks 4-16", 16 * 7 / 30.4375),
        ("6-month follow-up", 6),
    ],
)
def test_unit_before_number_timeframes(timeframe, expected_months):
    features = trial_row_features({
        "primary_outcome_timeframes_json": json.dumps([timeframe])
    })
    assert features["primary_outcome_timeframe_months"] == pytest.approx(expected_months)


def test_calibration_and_baseline_blending():
    import app.ml.predictor as predictor_mod

    assert predictor_mod._calibrate(10, {
        "calibration": {"method": "affine_raw", "slope": 2, "intercept": 3},
        "smearing_factor": 99,
    }) == 23
    assert predictor_mod._calibrate(10, {"smearing_factor": 1.2}) == pytest.approx(12.2)
    assert predictor_mod._calibrate(10, {
        "calibration": {"method": "smearing", "factor": 1.2}
    }) == pytest.approx(12.2)
    assert predictor_mod._blend_with_baseline(20, 10, {
        "baseline_blend": {"model_weight": 0.75, "baseline_weight": 0.25, "intercept": 1}
    }) == 18.5


def test_predictor_fails_closed_without_a_trained_model(tmp_path, monkeypatch):
    import app.ml.predictor as predictor_mod

    monkeypatch.setattr(predictor_mod, "MODELS_DIR", tmp_path)
    monkeypatch.delenv("ALLOW_DEMO_MODEL", raising=False)
    predictor_mod._cache.clear()
    with pytest.raises(FileNotFoundError, match="run the training command"):
        predictor_mod._load_model()


def test_train_and_predict(tmp_path, monkeypatch):
    import app.ml.train as train_mod
    import app.ml.predictor as predictor_mod

    monkeypatch.setattr(train_mod, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(predictor_mod, "MODELS_DIR", tmp_path)
    predictor_mod._cache.clear()

    meta = train(quick=True, allow_demo_save=True)
    assert meta["n_rows"] > 0
    assert meta["synthetic"] is True
    assert meta["selection_strategy"] == "rolling_origin"
    assert meta["evaluations"]["mature_temporal"]["selection_uses_test"] is False
    assert meta["evaluations"]["mature_temporal"]["test_year"] == 2018
    assert (tmp_path / "duration_text_ridge.npz").exists()
    assert meta["stacking"]["text_head"]["artifact_sha256"]
    assert meta["baseline_blend"]["model_weight"] >= 0
    assert meta["baseline_blend"]["baseline_weight"] >= 0
    specialist = meta["specialists"]["phase23_interventional"]
    if specialist["available"]:
        assert (tmp_path / specialist["artifact"]).exists()

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


def test_quick_model_save_requires_explicit_opt_in(tmp_path, monkeypatch):
    import app.ml.train as train_mod

    monkeypatch.setattr(train_mod, "MODELS_DIR", tmp_path)
    with pytest.raises(ValueError, match="Refusing to save a synthetic demo model"):
        train(quick=True)
    assert not (tmp_path / "duration_model.json").exists()


def test_ridge_artifact_and_nonnegative_blend_roundtrip(tmp_path):
    rng = np.random.default_rng(7)
    embeddings = rng.normal(size=(80, 12)).astype("float32")
    target = 2.0 + embeddings[:, 0] * 0.4 - embeddings[:, 1] * 0.2
    head = RidgeTextHead.fit(embeddings, target, alpha=1.0)
    before = head.predict(embeddings[:5])
    artifact = tmp_path / "ridge.npz"
    metadata = head.save(artifact)
    after = RidgeTextHead.load(artifact).predict(embeddings[:5])
    assert metadata["artifact_sha256"]
    assert after == pytest.approx(before, abs=1e-6)

    blend = fit_nonnegative_blend(
        [before, before + 0.1], before + 0.05, ["xgb_log", "text_log"]
    )
    assert all(weight >= 0 for weight in blend["weights"].values())
    prediction = apply_blend(blend, xgb_log=before, text_log=before + 0.1)
    assert prediction.shape == before.shape


def test_neighbor_baseline_uses_bounded_leakage_safe_reference(monkeypatch):
    monkeypatch.setenv("TRAIN_NEIGHBOR_SKLEARN_MAX_REFERENCE", "20")
    rng = np.random.default_rng(9)
    embeddings = rng.normal(size=(60, 8)).astype("float32")
    durations = np.linspace(5, 50, 60)
    baseline = NeighborDurationBaseline(n_neighbors=5, max_reference=40, seed=3)
    baseline.fit(embeddings[:50], durations[:50])
    prediction = baseline.predict(embeddings[50:])
    assert prediction.shape == (10,)
    assert np.isfinite(prediction).all()
    assert baseline.reference_size <= 40
