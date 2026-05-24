from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from app.main import calculate_coverage_days
from app.model_inference import (
    DATE_COL,
    INPUT_COLUMNS,
    TARGET_COL,
    USER_COL,
    records_to_dataframe,
)
from .conftest import valid_record


def test_calculate_coverage_days_for_empty_records():
    assert calculate_coverage_days([]) == 0


def test_calculate_coverage_days_counts_calendar_span():
    records = [
        {"date": "2024-01-01"},
        {"date": "2024-01-04"},
        {"date": "2024-01-14"},
    ]

    assert calculate_coverage_days(records) == 14


def test_records_to_dataframe_uses_model_columns():
    records = [
        valid_record(date(2024, 1, 1), fitness_level=0.04),
        valid_record(date(2024, 1, 2), fitness_level=0.06),
    ]

    df = records_to_dataframe(7, records)

    assert list(df.columns) == [USER_COL, DATE_COL] + INPUT_COLUMNS
    assert df[USER_COL].tolist() == [7, 7]
    assert df[DATE_COL].tolist() == ["2024-01-01", "2024-01-02"]
    assert df[TARGET_COL].tolist() == [0.04, 0.06]


def test_prepare_student_window_validates_required_columns(monkeypatch):
    from app.model_inference import FitnessInferenceEngine

    engine = object.__new__(FitnessInferenceEngine)

    with pytest.raises(ValueError, match="participant_id"):
        engine.prepare_student_window(pd.DataFrame({"date": ["2024-01-01"]}))


def test_prepare_student_window_rejects_multiple_students():
    from app.model_inference import FitnessInferenceEngine

    engine = object.__new__(FitnessInferenceEngine)
    engine.artifacts = None
    df = pd.DataFrame(
        {
            USER_COL: [1, 2],
            DATE_COL: ["2024-01-01", "2024-01-02"],
        }
    )

    with pytest.raises(ValueError, match="одного студента"):
        engine.prepare_student_window(df)


def test_prepare_student_window_rejects_invalid_dates():
    from app.model_inference import FitnessInferenceEngine

    engine = object.__new__(FitnessInferenceEngine)
    engine.artifacts = None
    df = pd.DataFrame(
        {
            USER_COL: [1],
            DATE_COL: ["bad-date"],
        }
    )

    with pytest.raises(ValueError, match="некорректные значения"):
        engine.prepare_student_window(df)


def test_prepare_student_window_rejects_short_calendar_coverage():
    from app.model_inference import FitnessInferenceEngine

    engine = object.__new__(FitnessInferenceEngine)
    engine.artifacts = None
    start = date(2024, 1, 1)
    df = pd.DataFrame(
        {
            USER_COL: [1 for _ in range(13)],
            DATE_COL: [(start + timedelta(days=offset)).isoformat() for offset in range(13)],
        }
    )

    with pytest.raises(ValueError, match="14"):
        engine.prepare_student_window(df)
