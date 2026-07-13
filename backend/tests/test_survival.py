import datetime as dt
import math

import pytest

pytest.importorskip("xgboost")

from app.ml.survival import _label_for_row, _parse_date


def test_survival_completed_row_has_exact_interval():
    label = _label_for_row(
        {
            "status": "COMPLETED",
            "start_date": "2020-01-01",
            "completion_date": "2022-01-01",
            "completion_date_type": "ACTUAL",
        },
        dt.date(2026, 1, 1),
    )
    assert label is not None
    assert label[0] == label[1]
    assert label[0] == pytest.approx(24, rel=0.02)


def test_survival_active_row_is_right_censored_at_fetch_date():
    label = _label_for_row(
        {
            "status": "RECRUITING",
            "start_date": "2024-01-01",
            "fetched_at": "2025-01-01T12:00:00+00:00",
        },
        dt.date(2026, 1, 1),
    )
    assert label is not None
    assert label[0] == pytest.approx(12, rel=0.02)
    assert math.isinf(label[1])


def test_survival_rejects_estimated_completion_as_event():
    assert _label_for_row(
        {
            "status": "COMPLETED",
            "start_date": "2020-01-01",
            "completion_date": "2022-01-01",
            "completion_date_type": "ESTIMATED",
        },
        dt.date(2026, 1, 1),
    ) is None


def test_parse_date_accepts_month_precision():
    assert _parse_date("2024-07") == dt.date(2024, 7, 1)
