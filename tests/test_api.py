from __future__ import annotations

from datetime import date

from .conftest import add_records_for_days, create_student, valid_record


def test_create_and_list_student(client):
    response = client.post(
        "/api/students",
        json={"name": "  Ivanov Ivan  ", "group_name": "FK-101"},
    )

    assert response.status_code == 200
    assert response.json() == {"id": 1, "name": "Ivanov Ivan", "group_name": "FK-101"}

    response = client.get("/api/students")

    assert response.status_code == 200
    assert response.json() == [
        {"id": 1, "name": "Ivanov Ivan", "group_name": "FK-101", "records_count": 0}
    ]


def test_add_record_list_records_and_status(client):
    student_id = create_student(client)

    response = client.post(
        f"/api/students/{student_id}/records",
        json=valid_record(date(2024, 1, 1)),
    )
    assert response.status_code == 200
    assert response.json() == {
        "status": "saved",
        "student_id": student_id,
        "date": "2024-01-01",
    }

    response = client.get(f"/api/students/{student_id}/records")
    assert response.status_code == 200
    records = response.json()
    assert len(records) == 1
    assert records[0]["student_id"] == student_id
    assert records[0]["date"] == "2024-01-01"
    assert records[0]["activity_type"] == "Dancing"

    response = client.get(f"/api/students/{student_id}/status")
    assert response.status_code == 200
    assert response.json() == {
        "student_id": student_id,
        "records_count": 1,
        "coverage_days": 1,
        "required_days": 14,
        "latest_date": "2024-01-01",
        "ready": False,
    }


def test_upsert_record_by_student_and_date(client):
    student_id = create_student(client)

    first = client.post(
        f"/api/students/{student_id}/records",
        json=valid_record(date(2024, 1, 1), weight_kg=53.7),
    )
    second = client.post(
        f"/api/students/{student_id}/records",
        json=valid_record(date(2024, 1, 1), weight_kg=55.1),
    )

    assert first.status_code == 200
    assert second.status_code == 200

    response = client.get(f"/api/students/{student_id}/records")
    records = response.json()

    assert len(records) == 1
    assert records[0]["weight_kg"] == 55.1


def test_status_ready_after_14_calendar_days(client):
    student_id = create_student(client)
    add_records_for_days(client, student_id, 14)

    response = client.get(f"/api/students/{student_id}/status")

    assert response.status_code == 200
    assert response.json()["records_count"] == 14
    assert response.json()["coverage_days"] == 14
    assert response.json()["latest_date"] == "2024-01-14"
    assert response.json()["ready"] is True


def test_predict_requires_14_calendar_days(client):
    student_id = create_student(client)
    add_records_for_days(client, student_id, 13)

    response = client.post(f"/api/students/{student_id}/predict")

    assert response.status_code == 400


def test_predict_uses_inference_engine_when_student_is_ready(client, monkeypatch):
    student_id = create_student(client)
    add_records_for_days(client, student_id, 14)

    class FakeEngine:
        def predict_next_14_days_growth(self, df):
            assert len(df) == 14
            assert set(["participant_id", "date", "fitness_level"]).issubset(df.columns)
            assert df["participant_id"].nunique() == 1
            return {
                "model_used": "LSTM",
                "current_fitness_level": 0.3,
                "predicted_delta_14_days": 0.12,
                "predicted_fitness_level_after_14_days": 0.42,
            }

    import app.main as main

    monkeypatch.setattr(main, "get_engine", lambda: FakeEngine())

    response = client.post(f"/api/students/{student_id}/predict")

    assert response.status_code == 200
    assert response.json() == {
        "student_id": student_id,
        "records_count": 14,
        "coverage_days": 14,
        "model_used": "LSTM",
        "current_fitness_level": 0.3,
        "predicted_delta_14_days": 0.12,
        "predicted_fitness_level_after_14_days": 0.42,
    }


def test_unknown_student_returns_404(client):
    response = client.get("/api/students/999/records")

    assert response.status_code == 404
