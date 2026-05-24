from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from app.model_inference import ARTIFACTS_PATH, MODEL_PATH, FitnessInferenceEngine, records_to_dataframe
from .conftest import valid_record


pytestmark = pytest.mark.integration


def test_real_model_loads_when_artifacts_exist():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file is missing: {MODEL_PATH}")
    if not ARTIFACTS_PATH.exists():
        pytest.skip(f"Preprocessing artifact is missing: {ARTIFACTS_PATH}")

    engine = FitnessInferenceEngine()

    assert engine.model is not None
    assert engine.artifacts.n_features_per_step > 0


def test_real_model_predicts_for_valid_14_day_window():
    if not MODEL_PATH.exists():
        pytest.skip(f"Model file is missing: {MODEL_PATH}")
    if not ARTIFACTS_PATH.exists():
        pytest.skip(f"Preprocessing artifact is missing: {ARTIFACTS_PATH}")

    start = date(2024, 1, 1)
    records = [
        valid_record(
            start + timedelta(days=offset),
            fitness_level=round(0.04 + offset * 0.02, 2),
            daily_steps=7128 + offset * 100,
        )
        for offset in range(14)
    ]
    df = records_to_dataframe(1, records)

    engine = FitnessInferenceEngine()
    result = engine.predict_next_14_days_growth(df)

    assert result["model_used"] == "LSTM"
    assert isinstance(result["current_fitness_level"], float)
    assert isinstance(result["predicted_delta_14_days"], float)
    assert isinstance(result["predicted_fitness_level_after_14_days"], float)
