import pytest

from app.services import similarity_engine as se

USDM = {
    "study": {
        "conditions": ["type 2 diabetes"],
        "phase": "Phase 2",
        "objectives": [{"endpoints": [{"description": "change in HbA1c from baseline"}]}],
        "design": {"allocation": "randomized", "masking": "double", "interventionModel": "parallel"},
        "arms": [{"name": "drug"}, {"name": "placebo"}],
        "interventions": [{"name": "metformin", "type": "drug"}],
    }
}

TRIAL_CLOSE = {
    "nct_id": "NCT001", "title": "t", "status": "s", "phase": "Phase 2",
    "conditions": ["type 2 diabetes"], "interventions": ["metformin"],
    "primary_outcomes": "change in HbA1c from baseline",
    "secondary_outcomes": None, "enrollment": 100, "start_date": None,
    "completion_date": None, "duration_months": 12, "allocation": "randomized",
    "masking": "double", "arms": 2, "intervention_model": "parallel",
    "sponsor": None, "eligibility": None,
}

TRIAL_FAR = {
    **TRIAL_CLOSE,
    "nct_id": "NCT002", "phase": "Phase 4", "conditions": ["lung cancer"],
    "primary_outcomes": "overall survival", "allocation": "non-randomized",
    "masking": "none", "arms": 6, "intervention_model": "single group",
}


def _mock_search(monkeypatch, results):
    monkeypatch.setattr(se, "search_trials", lambda query, limit=50: results)


def test_no_candidates_returns_empty(monkeypatch):
    _mock_search(monkeypatch, [])
    assert se.find_similar(USDM) == []


def test_jaccard_fallback_ranks_close_trial_higher(monkeypatch):
    monkeypatch.setattr(se, "_get_model", lambda: None)
    _mock_search(monkeypatch, [TRIAL_CLOSE, TRIAL_FAR])
    results = se.find_similar(USDM)
    assert results[0]["nct_id"] == "NCT001"
    assert results[0]["similarity"]["total"] > results[1]["similarity"]["total"]
    assert 0 <= results[0]["similarity"]["total"] <= 1


def test_phase_score_exact_adjacent_none():
    assert se._phase_score("Phase 2", "Phase 2") == 1.0
    assert se._phase_score("Phase 2", "Phase 3") == 0.5
    assert se._phase_score("Phase 2", "Phase 4") == 0.0
    assert se._phase_score(None, "Phase 2") == 0.0


def test_design_score_full_match():
    trial = TRIAL_CLOSE
    assert se._design_score(USDM, trial) == 1.0


def test_semantic_path_smoke(monkeypatch):
    pytest.importorskip("sentence_transformers")
    _mock_search(monkeypatch, [TRIAL_CLOSE, TRIAL_FAR])
    results = se.find_similar(USDM)
    assert len(results) == 2
