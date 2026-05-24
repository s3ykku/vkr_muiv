from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "health_fitness_dataset.csv"
MODEL_PATH = PROJECT_ROOT / "model.pth"
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts" / "lstm_preprocessing.joblib"

USER_COL = "participant_id"
DATE_COL = "date"
TARGET_COL = "fitness_level"
OBSERVED_FLAG_COL = "has_observation"

HORIZON_DAYS = 14
L = 14
RANDOM_STATE = 42

CATEGORICAL_CANDIDATES = [
    "gender",
    "activity_type",
    "intensity",
    "smoking_status",
    "health_condition",
]

INPUT_COLUMNS = [
    "age",
    "gender",
    "height_cm",
    "weight_kg",
    "activity_type",
    "duration_minutes",
    "intensity",
    "calories_burned",
    "avg_heart_rate",
    "hours_sleep",
    "stress_level",
    "daily_steps",
    "hydration_level",
    "bmi",
    "resting_heart_rate",
    "blood_pressure_systolic",
    "blood_pressure_diastolic",
    "health_condition",
    "smoking_status",
    "fitness_level",
]


class LSTMRegressor(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0 if num_layers == 1 else dropout,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last_hidden = out[:, -1, :]
        return self.head(last_hidden)


@dataclass
class PreprocessingArtifacts:
    flat_scaler: StandardScaler
    row_feature_cols: list[str]
    num_cols: list[str]
    cat_cols: list[str]
    n_features_per_step: int


class FitnessInferenceEngine:
    def __init__(
        self,
        csv_path: Path = CSV_PATH,
        model_path: Path = MODEL_PATH,
        artifacts_path: Path = ARTIFACTS_PATH,
    ):
        self.csv_path = csv_path
        self.model_path = model_path
        self.artifacts_path = artifacts_path
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.artifacts = self._load_or_build_artifacts()
        self.model = self._load_lstm_model()

    def predict_next_14_days_growth(
        self,
        student_14_days_df: pd.DataFrame,
        current_fitness_level: float | None = None,
    ) -> dict[str, float | str]:
        if student_14_days_df.empty:
            raise ValueError("Для прогноза нужна хотя бы одна запись студента.")

        if current_fitness_level is None:
            current_fitness_level = float(
                student_14_days_df.sort_values(DATE_COL)[TARGET_COL].iloc[-1]
            )

        if pd.isna(current_fitness_level):
            raise ValueError("current_fitness_level должен быть числом, а не NaN.")

        x_flat = self.prepare_student_window(student_14_days_df)
        x_seq = x_flat.reshape(1, L, self.artifacts.n_features_per_step)
        x_tensor = torch.as_tensor(x_seq, dtype=torch.float32).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred_delta = float(self.model(x_tensor).cpu().numpy().ravel()[0])

        pred_future_fitness = float(current_fitness_level) + pred_delta

        return {
            "model_used": "LSTM",
            "current_fitness_level": float(current_fitness_level),
            "predicted_delta_14_days": pred_delta,
            "predicted_fitness_level_after_14_days": float(pred_future_fitness),
        }

    def prepare_student_window(self, student_14_days_df: pd.DataFrame) -> np.ndarray:
        required_cols = {USER_COL, DATE_COL}
        missing_cols = required_cols - set(student_14_days_df.columns)
        if missing_cols:
            raise ValueError(f"Во входных данных не хватает колонок: {sorted(missing_cols)}")

        student_df = student_14_days_df.copy()
        student_df[DATE_COL] = pd.to_datetime(student_df[DATE_COL], errors="coerce")
        student_df = self._normalize_categorical_missing(student_df)

        if student_df.empty:
            raise ValueError("Для прогноза нужен хотя бы один день наблюдений.")

        if student_df[USER_COL].nunique(dropna=True) > 1:
            raise ValueError("Для прогноза передайте данные только одного студента.")

        if student_df[DATE_COL].isna().any():
            raise ValueError("В колонке с датой есть некорректные значения.")

        student_df = (
            student_df.sort_values(DATE_COL)
            .drop_duplicates(subset=[DATE_COL], keep="last")
            .reset_index(drop=True)
        )

        last_date = student_df[DATE_COL].max()
        first_date = student_df[DATE_COL].min()
        if (last_date - first_date).days + 1 < L:
            raise ValueError(
                f"Для прогноза нужно покрытие минимум {L} календарных дней "
                "между первой и последней датой."
            )

        student_id = student_df[USER_COL].iloc[0]
        window_dates = pd.date_range(end=last_date, periods=L, freq="D")

        student_df[OBSERVED_FLAG_COL] = 1
        calendar = pd.DataFrame({USER_COL: student_id, DATE_COL: window_dates})
        tmp = calendar.merge(student_df, on=[USER_COL, DATE_COL], how="left", sort=True)
        tmp[OBSERVED_FLAG_COL] = tmp[OBSERVED_FLAG_COL].fillna(0).astype("int8")

        cols_for_features = [
            c for c in self.artifacts.num_cols if c in tmp.columns
        ] + [c for c in self.artifacts.cat_cols if c in tmp.columns]
        tmp = tmp[[USER_COL, DATE_COL] + cols_for_features].copy()

        if self.artifacts.num_cols:
            tmp[self.artifacts.num_cols] = tmp[self.artifacts.num_cols].fillna(0.0)

        if self.artifacts.cat_cols:
            tmp = pd.get_dummies(
                tmp,
                columns=[c for c in self.artifacts.cat_cols if c in tmp.columns],
                dummy_na=True,
            )

        for col in self.artifacts.row_feature_cols:
            if col not in tmp.columns:
                tmp[col] = 0

        tmp = tmp[[USER_COL, DATE_COL] + self.artifacts.row_feature_cols]
        flat = tmp[self.artifacts.row_feature_cols].to_numpy(dtype=np.float32).reshape(1, -1)
        flat = self.artifacts.flat_scaler.transform(flat).astype(np.float32)
        return flat

    def _load_lstm_model(self) -> LSTMRegressor:
        if not self.model_path.exists():
            raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")

        model = LSTMRegressor(input_size=self.artifacts.n_features_per_step).to(self.device)
        state_dict = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.eval()
        return model

    def _load_or_build_artifacts(self) -> PreprocessingArtifacts:
        if self.artifacts_path.exists():
            data = joblib.load(self.artifacts_path)
            return PreprocessingArtifacts(**data)

        artifacts = self._build_artifacts_from_csv()
        self.artifacts_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifacts.__dict__, self.artifacts_path)
        return artifacts

    def _build_artifacts_from_csv(self) -> PreprocessingArtifacts:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Файл датасета не найден: {self.csv_path}")

        df = pd.read_csv(self.csv_path)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
        df = df.dropna(subset=[USER_COL, DATE_COL, TARGET_COL]).copy()
        df = self._normalize_categorical_missing(df)

        df["future_date"] = df[DATE_COL] + pd.to_timedelta(HORIZON_DAYS, unit="D")
        future_tbl = (
            df[[USER_COL, DATE_COL, TARGET_COL]]
            .rename(columns={DATE_COL: "future_meas_date", TARGET_COL: "fitness_level_future"})
            .sort_values(["future_meas_date", USER_COL])
            .reset_index(drop=True)
        )
        left_sorted = df.sort_values(["future_date", USER_COL]).reset_index(drop=True)

        df = pd.merge_asof(
            left=left_sorted,
            right=future_tbl,
            left_on="future_date",
            right_on="future_meas_date",
            by=USER_COL,
            direction="forward",
            allow_exact_matches=True,
        )
        df = df.dropna(subset=["fitness_level_future"]).reset_index(drop=True)
        df["y_delta"] = df["fitness_level_future"] - df[TARGET_COL]
        df["y_grow"] = (df["y_delta"] > 0).astype("int64")

        target_df = df.copy()
        calendar_blocks = []
        for pid, group in target_df.groupby(USER_COL, sort=False):
            group = (
                group.sort_values(DATE_COL)
                .drop_duplicates(subset=[DATE_COL], keep="last")
                .reset_index(drop=True)
                .copy()
            )
            group[OBSERVED_FLAG_COL] = 1
            full_dates = pd.date_range(group[DATE_COL].min(), group[DATE_COL].max(), freq="D")
            calendar = pd.DataFrame({USER_COL: pid, DATE_COL: full_dates})
            block = calendar.merge(group, on=[USER_COL, DATE_COL], how="left", sort=True)
            block[OBSERVED_FLAG_COL] = block[OBSERVED_FLAG_COL].fillna(0).astype("int8")
            calendar_blocks.append(block)

        df = (
            pd.concat(calendar_blocks, ignore_index=True)
            .sort_values([USER_COL, DATE_COL])
            .reset_index(drop=True)
        )

        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        drop_num = {TARGET_COL, "fitness_level_future", "y_delta", "y_grow", USER_COL}
        num_cols = [c for c in num_cols if c not in drop_num]
        cat_cols = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]

        feat_df = df[[USER_COL, DATE_COL] + num_cols + cat_cols].copy()
        if num_cols:
            feat_df[num_cols] = feat_df[num_cols].fillna(0.0)
        if cat_cols:
            feat_df = pd.get_dummies(feat_df, columns=cat_cols, dummy_na=True)

        row_feature_cols = [c for c in feat_df.columns if c not in [USER_COL, DATE_COL]]

        Xs: list[np.ndarray] = []
        yds: list[np.ndarray] = []
        gids: list[np.ndarray] = []
        for pid, group in df.groupby(USER_COL, sort=False):
            out = self._build_windows_for_user(group, feat_df, row_feature_cols)
            if out is None:
                continue

            x_group, y_group = out
            Xs.append(x_group)
            yds.append(y_group)
            gids.append(np.full(len(y_group), pid, dtype=object))

        if not Xs:
            raise ValueError("Не удалось сформировать обучающие окна для scaler.")

        X = np.concatenate(Xs, axis=0)
        y_delta = np.concatenate(yds, axis=0)
        groups = np.concatenate(gids, axis=0)

        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=RANDOM_STATE)
        train_idx, _ = next(gss.split(X, y_delta, groups=groups))

        flat_scaler = StandardScaler()
        flat_scaler.fit(X[train_idx])

        return PreprocessingArtifacts(
            flat_scaler=flat_scaler,
            row_feature_cols=row_feature_cols,
            num_cols=num_cols,
            cat_cols=cat_cols,
            n_features_per_step=len(row_feature_cols),
        )

    @staticmethod
    def _build_windows_for_user(
        user_block: pd.DataFrame,
        feat_df: pd.DataFrame,
        row_feature_cols: list[str],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        user_block = user_block.sort_values(DATE_COL)
        ub_feat = feat_df.loc[user_block.index, row_feature_cols].to_numpy(dtype=np.float32)
        y_delta = user_block["y_delta"].to_numpy(dtype=np.float32)
        dates_count = len(user_block)
        observed_mask = user_block[OBSERVED_FLAG_COL].fillna(0).to_numpy(dtype=np.int8)
        valid_target_mask = user_block["y_delta"].notna().to_numpy()

        if dates_count < L:
            return None

        x_list = []
        y_list = []
        for i in range(L - 1, dates_count):
            if observed_mask[i] == 0 or not valid_target_mask[i]:
                continue

            window = ub_feat[i - L + 1 : i + 1]
            x_list.append(window.reshape(-1))
            y_list.append(y_delta[i])

        if not x_list:
            return None

        return np.stack(x_list), np.array(y_list, dtype=np.float32)

    @staticmethod
    def _normalize_categorical_missing(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        for col in CATEGORICAL_CANDIDATES:
            if col in df.columns:
                df[col] = df[col].replace({"": np.nan, "None": np.nan, "none": np.nan})
        return df


def records_to_dataframe(student_id: int, records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {USER_COL: student_id}
        for column in [DATE_COL] + INPUT_COLUMNS:
            row[column] = record.get(column)
        rows.append(row)
    return pd.DataFrame(rows)
