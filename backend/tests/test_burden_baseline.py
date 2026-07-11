from app.services.baseline import compute_baseline
from app.services.burden_analyzer import analyze_burden

SIMPLE_USDM = {
    "study": {
        "conditions": ["diabetes"],
        "arms": [{"name": "a"}, {"name": "b"}],
        "objectives": [{"endpoints": [{"description": "hba1c change"}]}],
        "design": {"interventionModel": "parallel"},
        "eligibility": {
            "inclusionCriteria": ["age 18+"],
            "exclusionCriteria": ["pregnant"],
            "minAge": 18,
            "maxAge": 75,
        },
        "plannedEnrollment": 300,
        "scheduleOfActivities": [
            {"name": "visit1", "procedures": ["blood draw"]},
            {"name": "visit2", "procedures": ["blood draw", "MRI"]},
        ],
        "durationMonths": 12,
    }
}

COMPLEX_USDM = {
    "study": {
        "conditions": ["rare mitochondrial syndrome"],
        "arms": [{"name": f"arm{i}"} for i in range(5)],
        "objectives": [{"endpoints": [{"description": f"ep{i}"} for i in range(4)]}],
        "design": {"interventionModel": "crossover"},
        "eligibility": {
            "inclusionCriteria": [f"c{i}" for i in range(8)],
            "exclusionCriteria": [f"e{i}" for i in range(8)],
            "minAge": 18,
            "maxAge": 30,
        },
        "plannedEnrollment": 20,
        "scheduleOfActivities": [
            {"name": f"v{i}", "procedures": ["biopsy", "lumbar puncture", "infusion"]}
            for i in range(10)
        ],
        "durationMonths": 24,
    }
}


def test_analyze_burden_missing_fields_defaults():
    result = analyze_burden({"study": {}})
    for key in ("complexity_score", "recruitment_difficulty", "patient_burden"):
        assert 0 <= result[key] <= 100
    assert isinstance(result["factors"], list)


def test_analyze_burden_complex_scores_higher():
    simple = analyze_burden(SIMPLE_USDM)
    complex_ = analyze_burden(COMPLEX_USDM)
    assert complex_["complexity_score"] > simple["complexity_score"]
    assert complex_["recruitment_difficulty"] > simple["recruitment_difficulty"]
    assert complex_["patient_burden"] > simple["patient_burden"]


def test_analyze_burden_clamped_and_factors_nonempty():
    result = analyze_burden(COMPLEX_USDM)
    assert all(f["score"] >= 0 for f in result["factors"])
    assert len(result["factors"]) > 0


def _trial(nct_id, score, duration, enrollment):
    return {
        "nct_id": nct_id, "duration_months": duration, "enrollment": enrollment,
        "similarity": {"total": score},
    }


def test_compute_baseline_empty():
    result = compute_baseline([])
    assert result["n_trials"] == 0
    assert result["expected_duration_months"] is None
    assert result["trials_used"] == []


def test_compute_baseline_small_n_uses_min_max():
    trials = [_trial("NCT1", 0.9, 10, 100), _trial("NCT2", 0.5, 20, 200)]
    result = compute_baseline(trials)
    assert result["n_trials"] == 2
    assert result["ci_low"] == 10
    assert result["ci_high"] == 20
    assert result["median_enrollment"] == 150


def test_compute_baseline_weighted_mean_and_topk():
    trials = [_trial(f"NCT{i}", 1.0 - i * 0.05, 12, 100) for i in range(15)]
    result = compute_baseline(trials, k=10)
    assert result["n_trials"] == 10
    assert result["expected_duration_months"] == 12.0
    assert len(result["trials_used"]) == 10


def test_compute_baseline_skips_none_values():
    trials = [
        _trial("NCT1", 0.9, None, 100),
        _trial("NCT2", 0.8, 15, None),
        _trial("NCT3", 0.7, 10, 200),
    ]
    result = compute_baseline(trials)
    assert result["expected_duration_months"] is not None
    assert result["median_enrollment"] == 150
