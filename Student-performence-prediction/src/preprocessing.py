"""Data loading, auditing, target creation, and reusable preprocessing."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
TARGET_LABELS = ["At Risk", "Average", "High Performance"]
TARGET_TO_CODE = {label: code for code, label in enumerate(TARGET_LABELS)}
LEAKAGE_COLUMNS = ["G1", "G2", "G3"]
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_DIR / "data" / "raw" / "student-mat.csv"


def load_dataset(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the supplied semicolon-separated student mathematics dataset."""
    return pd.read_csv(data_path, sep=";")


def create_target(dataframe: pd.DataFrame) -> pd.Series:
    """Map final grades to the agreed three-class performance target."""
    grade = dataframe["G3"]
    labels = pd.Series("Average", index=dataframe.index, dtype="object")
    labels.loc[grade <= 9] = "At Risk"
    labels.loc[grade >= 15] = "High Performance"
    return labels.map(TARGET_TO_CODE).astype(int)


def get_feature_groups(dataframe: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Return feature, numeric feature, and categorical feature names."""
    feature_columns = [column for column in dataframe.columns if column not in LEAKAGE_COLUMNS]
    numeric_columns = dataframe[feature_columns].select_dtypes(include="number").columns.tolist()
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    return feature_columns, numeric_columns, categorical_columns


def prepare_features(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Remove leakage columns and return raw features with encoded target."""
    target = create_target(dataframe)
    feature_columns, numeric_columns, categorical_columns = get_feature_groups(dataframe)
    features = dataframe.loc[:, feature_columns].copy()
    return features, target, numeric_columns, categorical_columns


def build_preprocessor(numeric_columns: list[str], categorical_columns: list[str], scale_numeric: bool) -> ColumnTransformer:
    """Build a training-only preprocessing transformer for sklearn models."""
    numeric_steps: list[tuple[str, Any]] = []
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_transformer: Any = Pipeline(numeric_steps) if numeric_steps else "passthrough"
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_columns),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ],
        remainder="drop",
    )


def audit_dataset(dataframe: pd.DataFrame) -> dict[str, Any]:
    """Produce a factual audit without altering legitimate observations."""
    feature_columns, numeric_columns, categorical_columns = get_feature_groups(dataframe)
    expected_ranges = {
        "age": (15, 22), "Medu": (0, 4), "Fedu": (0, 4), "traveltime": (1, 4),
        "studytime": (1, 4), "failures": (0, 4), "famrel": (1, 5), "freetime": (1, 5),
        "goout": (1, 5), "Dalc": (1, 5), "Walc": (1, 5), "health": (1, 5),
        "absences": (0, None), "G1": (0, 20), "G2": (0, 20), "G3": (0, 20),
    }
    invalid_values = {
        column: int(((dataframe[column] < lower) | ((dataframe[column] > upper) if upper is not None else False)).sum())
        for column, (lower, upper) in expected_ranges.items()
    }
    target = create_target(dataframe)
    return {
        "rows": int(dataframe.shape[0]),
        "columns": int(dataframe.shape[1]),
        "column_names": dataframe.columns.tolist(),
        "data_types": {column: str(dtype) for column, dtype in dataframe.dtypes.items()},
        "missing_values": {column: int(count) for column, count in dataframe.isna().sum().items()},
        "duplicate_rows": int(dataframe.duplicated().sum()),
        "categorical_unique_values": {column: sorted(dataframe[column].dropna().unique().tolist()) for column in categorical_columns},
        "numerical_ranges": {
            column: {"min": float(dataframe[column].min()), "max": float(dataframe[column].max())}
            for column in numeric_columns + LEAKAGE_COLUMNS
        },
        "invalid_value_counts": invalid_values,
        "target_distribution": {
            TARGET_LABELS[code]: int(count)
            for code, count in target.value_counts().sort_index().items()
        },
        "features_used": feature_columns,
        "excluded_leakage_columns": LEAKAGE_COLUMNS,
    }


def save_audit(audit: dict[str, Any], output_dir: Path) -> None:
    """Save machine-readable and concise human-readable audit reports."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "data_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    lines = [
        "# Data Audit",
        "",
        f"- Rows: {audit['rows']}",
        f"- Columns: {audit['columns']}",
        f"- Missing values: {sum(audit['missing_values'].values())}",
        f"- Duplicate rows: {audit['duplicate_rows']}",
        f"- Invalid values against documented ranges: {sum(audit['invalid_value_counts'].values())}",
        "- G1, G2, and G3 are excluded from all model features to prevent leakage.",
        "",
        "## Target Distribution",
    ]
    lines.extend(f"- {label}: {count}" for label, count in audit["target_distribution"].items())
    (output_dir / "data_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
