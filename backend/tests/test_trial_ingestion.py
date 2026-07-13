import json

import app.database.db as db_module
from app.database.fetch_trials import import_snapshot_file, parse_study
from app.ml.features import trial_row_features


def test_parse_study_persists_lossless_arrays_and_criteria_counts():
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000001", "briefTitle": "Test"},
            "statusModule": {
                "overallStatus": "COMPLETED",
                "startDateStruct": {"date": "2020-01-15", "type": "ACTUAL"},
                "completionDateStruct": {"date": "2021-01-15", "type": "ACTUAL"},
            },
            "designModule": {
                "studyType": "INTERVENTIONAL",
                "enrollmentInfo": {"count": 20, "type": "ACTUAL"},
            },
            "conditionsModule": {"conditions": ["Diabetes Mellitus, Type 2"]},
            "armsInterventionsModule": {
                "interventions": [{"type": "DRUG", "name": "Drug; combination"}]
            },
            "outcomesModule": {
                "primaryOutcomes": [{
                    "measure": "Measure; with delimiter",
                    "timeFrame": "Week 4 through Week 16",
                }],
                "secondaryOutcomes": [{
                    "measure": "Secondary",
                    "timeFrame": "Day 90",
                }],
            },
            "eligibilityModule": {
                "eligibilityCriteria": (
                    "Inclusion Criteria:\n- first\n- second\n"
                    "Exclusion Criteria:\n1. third\n2. fourth"
                )
            },
        }
    }

    row = parse_study(study, fetched_at="2026-01-01T00:00:00+00:00")

    assert json.loads(row["conditions_json"]) == ["Diabetes Mellitus, Type 2"]
    assert json.loads(row["interventions_json"]) == [
        {"type": "DRUG", "name": "Drug; combination"}
    ]
    assert json.loads(row["primary_outcomes_json"]) == ["Measure; with delimiter"]
    assert json.loads(row["primary_outcome_timeframes_json"]) == [
        "Week 4 through Week 16"
    ]
    assert json.loads(row["secondary_outcome_timeframes_json"]) == ["Day 90"]
    assert row["inclusion_criteria_count"] == 2
    assert row["exclusion_criteria_count"] == 2
    assert row["feature_snapshot_kind"] == "LATEST"
    assert row["feature_snapshot_date"] == "2026-01-01T00:00:00+00:00"


def test_import_initial_snapshot_preserves_actual_label_endpoints(tmp_path, monkeypatch):
    monkeypatch.setattr(db_module, "DB_PATH", tmp_path / "trials.db")
    db_module.init_db()

    latest = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000002", "briefTitle": "Latest"},
            "statusModule": {
                "overallStatus": "COMPLETED",
                "startDateStruct": {"date": "2020-03-01", "type": "ACTUAL"},
                "completionDateStruct": {"date": "2022-03-01", "type": "ACTUAL"},
            },
            "designModule": {
                "studyType": "INTERVENTIONAL",
                "enrollmentInfo": {"count": 95, "type": "ACTUAL"},
            },
        }
    }
    latest_row = parse_study(latest, fetched_at="2026-01-01T00:00:00+00:00")
    db_module.upsert_trials([latest_row])

    initial = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT00000002", "briefTitle": "Initial"},
            "statusModule": {
                "overallStatus": "RECRUITING",
                "studyFirstSubmitDate": "2019-12-15",
                "startDateStruct": {"date": "2020-01-01", "type": "ESTIMATED"},
                "completionDateStruct": {"date": "2021-06-01", "type": "ESTIMATED"},
            },
            "designModule": {
                "studyType": "INTERVENTIONAL",
                "enrollmentInfo": {"count": 120, "type": "ESTIMATED"},
            },
            "conditionsModule": {"conditions": ["Condition, with comma"]},
        }
    }
    snapshot_path = tmp_path / "initial.json"
    snapshot_path.write_text(json.dumps(initial))

    assert import_snapshot_file(snapshot_path) == 1
    with db_module.get_db() as db:
        row = dict(db.execute(
            "SELECT * FROM trials WHERE nct_id='NCT00000002'"
        ).fetchone())

    assert row["feature_snapshot_kind"] == "INITIAL"
    assert row["feature_snapshot_date"] == "2019-12-15"
    assert row["status"] == "COMPLETED"
    assert row["duration_months"] == latest_row["duration_months"]
    assert row["start_date"] == "2020-03-01"
    assert row["start_date_type"] == "ACTUAL"
    assert row["feature_start_date"] == "2020-01-01"
    assert row["feature_start_date_type"] == "ESTIMATED"
    assert row["enrollment"] == 120
    assert row["enrollment_type"] == "ESTIMATED"
    assert trial_row_features(row)["start_year"] == 2020
    assert trial_row_features(row)["n_conditions"] == 1
