from __future__ import annotations

from datetime import date, timedelta
from typing import Any

import pytest
from fastapi.testclient import TestClient

import app.main as main


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DB_PATH", tmp_path / "fitness_test.db")
    monkeypatch.setattr(main, "_engine", None)
    main.init_db()

    with TestClient(main.app) as test_client:
        yield test_client


def valid_record(day: date | str = date(2024, 1, 1), **overrides: Any) -> dict[str, Any]:
    if isinstance(day, date):
        day_value = day.isoformat()
    else:
        day_value = day

    payload: dict[str, Any] = {
        "date": day_value,
        "age": 56,
        "gender": "F",
        "height_cm": 165.3,
        "weight_kg": 53.7,
        "activity_type": "Dancing",
        "duration_minutes": 41,
        "intensity": "Low",
        "calories_burned": 3.3,
        "avg_heart_rate": 103,
        "hours_sleep": 6.6,
        "stress_level": 3,
        "daily_steps": 7128,
        "hydration_level": 1.5,
        "bmi": 19.6,
        "resting_heart_rate": 69.5,
        "blood_pressure_systolic": 110.7,
        "blood_pressure_diastolic": 72.9,
        "health_condition": None,
        "smoking_status": "Never",
        "fitness_level": 0.04,
    }
    payload.update(overrides)
    return payload


def create_student(client: TestClient, name: str = "Ivanova Anna") -> int:
    response = client.post(
        "/api/students",
        json={"name": name, "group_name": "FK-101"},
    )
    assert response.status_code == 200
    return int(response.json()["id"])


def add_records_for_days(client: TestClient, student_id: int, days_count: int) -> None:
    start = date(2024, 1, 1)
    for offset in range(days_count):
        day = start + timedelta(days=offset)
        response = client.post(
            f"/api/students/{student_id}/records",
            json=valid_record(
                day,
                fitness_level=round(0.04 + offset * 0.02, 2),
                daily_steps=7128 + offset * 100,
            ),
        )
        assert response.status_code == 200
