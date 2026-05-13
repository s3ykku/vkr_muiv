from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .model_inference import INPUT_COLUMNS, L, FitnessInferenceEngine, records_to_dataframe


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_PATH = PROJECT_ROOT / "fitness_service.db"
STATIC_DIR = Path(__file__).resolve().parent / "static"

app = FastAPI(title="Fitness Level Prediction Service")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_engine: FitnessInferenceEngine | None = None


class StudentCreate(BaseModel):
    name: str = Field(min_length=1)
    group_name: str | None = None


class RecordCreate(BaseModel):
    date: date
    age: int
    gender: str
    height_cm: float
    weight_kg: float
    activity_type: str
    duration_minutes: int
    intensity: str
    calories_burned: float
    avg_heart_rate: int
    hours_sleep: float
    stress_level: int
    daily_steps: int
    hydration_level: float
    bmi: float
    resting_heart_rate: float
    blood_pressure_systolic: float
    blood_pressure_diastolic: float
    health_condition: str | None = None
    smoking_status: str
    fitness_level: float


def get_engine() -> FitnessInferenceEngine:
    global _engine
    if _engine is None:
        _engine = FitnessInferenceEngine()
    return _engine


def connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    with connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS students (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                group_name TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_id INTEGER NOT NULL,
                date TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                height_cm REAL NOT NULL,
                weight_kg REAL NOT NULL,
                activity_type TEXT NOT NULL,
                duration_minutes INTEGER NOT NULL,
                intensity TEXT NOT NULL,
                calories_burned REAL NOT NULL,
                avg_heart_rate INTEGER NOT NULL,
                hours_sleep REAL NOT NULL,
                stress_level INTEGER NOT NULL,
                daily_steps INTEGER NOT NULL,
                hydration_level REAL NOT NULL,
                bmi REAL NOT NULL,
                resting_heart_rate REAL NOT NULL,
                blood_pressure_systolic REAL NOT NULL,
                blood_pressure_diastolic REAL NOT NULL,
                health_condition TEXT,
                smoking_status TEXT NOT NULL,
                fitness_level REAL NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY(student_id) REFERENCES students(id) ON DELETE CASCADE,
                UNIQUE(student_id, date)
            )
            """
        )


@app.on_event("startup")
def on_startup() -> None:
    init_db()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/students")
def list_students() -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT s.id, s.name, s.group_name, COUNT(r.id) AS records_count
            FROM students s
            LEFT JOIN records r ON r.student_id = s.id
            GROUP BY s.id
            ORDER BY s.name
            """
        ).fetchall()
    return [dict(row) for row in rows]


@app.post("/api/students")
def create_student(payload: StudentCreate) -> dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    with connect() as conn:
        cursor = conn.execute(
            "INSERT INTO students (name, group_name, created_at) VALUES (?, ?, ?)",
            (payload.name.strip(), payload.group_name, now),
        )
        student_id = cursor.lastrowid
    return {"id": student_id, "name": payload.name.strip(), "group_name": payload.group_name}


@app.delete("/api/students/{student_id}")
def delete_student(student_id: int) -> dict[str, str]:
    with connect() as conn:
        cursor = conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Студент не найден")
    return {"status": "deleted"}


@app.get("/api/students/{student_id}/records")
def list_records(student_id: int) -> list[dict[str, Any]]:
    ensure_student_exists(student_id)
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM records WHERE student_id = ? ORDER BY date DESC",
            (student_id,),
        ).fetchall()
    return [record_row_to_dict(row) for row in rows]


@app.post("/api/students/{student_id}/records")
def add_record(student_id: int, payload: RecordCreate) -> dict[str, Any]:
    ensure_student_exists(student_id)
    data = payload.dict()
    data["date"] = payload.date.isoformat()
    data["health_condition"] = data.get("health_condition") or None
    now = datetime.now().isoformat(timespec="seconds")

    columns = ["student_id", "created_at", "date"] + INPUT_COLUMNS
    values = [student_id, now, data["date"]] + [data.get(column) for column in INPUT_COLUMNS]
    update_clause = ", ".join([f"{column} = excluded.{column}" for column in INPUT_COLUMNS])
    placeholders = ", ".join(["?"] * len(columns))

    with connect() as conn:
        conn.execute(
            f"""
            INSERT INTO records ({", ".join(columns)})
            VALUES ({placeholders})
            ON CONFLICT(student_id, date) DO UPDATE SET
                {update_clause},
                created_at = excluded.created_at
            """,
            values,
        )

    return {"status": "saved", "student_id": student_id, "date": data["date"]}


@app.delete("/api/students/{student_id}/records/{record_id}")
def delete_record(student_id: int, record_id: int) -> dict[str, str]:
    ensure_student_exists(student_id)
    with connect() as conn:
        cursor = conn.execute(
            "DELETE FROM records WHERE id = ? AND student_id = ?",
            (record_id, student_id),
        )
        if cursor.rowcount == 0:
            raise HTTPException(status_code=404, detail="Запись не найдена")
    return {"status": "deleted"}


@app.get("/api/students/{student_id}/status")
def get_student_status(student_id: int) -> dict[str, Any]:
    ensure_student_exists(student_id)
    records = get_student_records(student_id)
    coverage_days = calculate_coverage_days(records)
    latest_date = max((record["date"] for record in records), default=None)
    return {
        "student_id": student_id,
        "records_count": len(records),
        "coverage_days": coverage_days,
        "required_days": L,
        "latest_date": latest_date,
        "ready": coverage_days >= L,
    }


@app.post("/api/students/{student_id}/predict")
def predict(student_id: int) -> dict[str, Any]:
    ensure_student_exists(student_id)
    records = get_student_records(student_id)
    coverage_days = calculate_coverage_days(records)
    if coverage_days < L:
        raise HTTPException(
            status_code=400,
            detail=f"Для расчёта нужно покрытие минимум {L} календарных дней.",
        )

    df = records_to_dataframe(student_id, records)
    try:
        result = get_engine().predict_next_14_days_growth(df)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return {
        "student_id": student_id,
        "records_count": len(records),
        "coverage_days": coverage_days,
        **result,
    }


def ensure_student_exists(student_id: int) -> None:
    with connect() as conn:
        row = conn.execute("SELECT id FROM students WHERE id = ?", (student_id,)).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Студент не найден")


def get_student_records(student_id: int) -> list[dict[str, Any]]:
    with connect() as conn:
        rows = conn.execute(
            "SELECT * FROM records WHERE student_id = ? ORDER BY date",
            (student_id,),
        ).fetchall()
    return [record_row_to_dict(row) for row in rows]


def record_row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    return {column: data.get(column) for column in ["id", "student_id", "date"] + INPUT_COLUMNS}


def calculate_coverage_days(records: list[dict[str, Any]]) -> int:
    if not records:
        return 0

    dates = [datetime.fromisoformat(record["date"]).date() for record in records]
    return (max(dates) - min(dates)).days + 1
