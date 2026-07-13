import hashlib

import pytest

pytest.importorskip("xgboost")

from app.ml import predictor


class _Booster:
    def num_features(self):
        return 100


class _Model:
    def get_booster(self):
        return _Booster()


def test_schema_validation_requires_complete_metadata():
    with pytest.raises(ValueError, match="metadata is incomplete"):
        predictor._validate_model_schema(_Model(), {"schema_version": 3})


def test_artifact_checksum_is_enforced(tmp_path):
    artifact = tmp_path / "artifact.bin"
    artifact.write_bytes(b"expected")
    expected = hashlib.sha256(b"different").hexdigest()
    with pytest.raises(ValueError, match="checksum mismatch"):
        predictor._require_artifact(artifact, expected)


def test_log_to_raw_rejects_nonfinite_and_caps_scope():
    with pytest.raises(ValueError, match="non-finite"):
        predictor._log_to_raw(float("nan"))
    assert predictor._log_to_raw(1000) == pytest.approx(240)
    assert predictor._log_to_raw(-1000) == 0


def test_phase_specialist_route_requires_both_phase_and_study_type():
    meta = {"specialists": {"phase23_interventional": {"available": True}}}
    assert predictor._use_phase23_specialist(
        {"phase_num": 2, "is_interventional": 1}, meta
    )
    assert not predictor._use_phase23_specialist(
        {"phase_num": 2, "is_interventional": 0}, meta
    )
    assert not predictor._use_phase23_specialist(
        {"phase_num": 4, "is_interventional": 1}, meta
    )
