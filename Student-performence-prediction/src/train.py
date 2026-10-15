"""End-to-end reproducible training entry point for the project."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.evaluate import create_eda_figures, create_model_comparison, evaluate_model, save_feature_importance
from src.preprocessing import (
    RANDOM_STATE,
    ROOT_DIR,
    TARGET_LABELS,
    audit_dataset,
    build_preprocessor,
    load_dataset,
    prepare_features,
    save_audit,
)


MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"


def build_sklearn_models(numeric_columns: list[str], categorical_columns: list[str]) -> dict[str, Pipeline]:
    """Create the three sklearn-compatible model pipelines."""
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", build_preprocessor(numeric_columns, categorical_columns, scale_numeric=True)),
            ("model", LogisticRegression(max_iter=3000, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", build_preprocessor(numeric_columns, categorical_columns, scale_numeric=False)),
            ("model", RandomForestClassifier(n_estimators=400, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=-1)),
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", build_preprocessor(numeric_columns, categorical_columns, scale_numeric=False)),
            ("model", XGBClassifier(
                n_estimators=250, learning_rate=0.05, max_depth=3, subsample=0.9,
                colsample_bytree=0.9, objective="multi:softprob", num_class=3,
                eval_metric="mlogloss", random_state=RANDOM_STATE, n_jobs=-1,
            )),
        ]),
    }


def choose_model(metrics: pd.DataFrame) -> pd.Series:
    """Prioritize Macro F1, then At Risk recall and ROC-AUC for selection."""
    return metrics.sort_values(
        ["Macro F1", "At Risk Recall", "High Performance Recall", "ROC-AUC", "Weighted F1"],
        ascending=False,
    ).iloc[0]


def main() -> None:
    """Run audit, EDA, model comparison, persistence, and prediction validation."""
    for directory in [MODELS_DIR, RESULTS_DIR / "plots", RESULTS_DIR / "confusion_matrices", RESULTS_DIR / "reports"]:
        directory.mkdir(parents=True, exist_ok=True)
    dataframe = load_dataset()
    audit = audit_dataset(dataframe)
    save_audit(audit, RESULTS_DIR)
    features, target, numeric_columns, categorical_columns = prepare_features(dataframe)
    features_train, features_test, target_train, target_test = train_test_split(
        features, target, test_size=0.20, random_state=RANDOM_STATE, stratify=target
    )
    create_eda_figures(dataframe.drop(columns=["G1", "G2", "G3"]), target, RESULTS_DIR / "plots")

    trained_models: dict[str, Any] = {}
    for model_name, pipeline in build_sklearn_models(numeric_columns, categorical_columns).items():
        pipeline.fit(features_train, target_train)
        trained_models[model_name] = pipeline

    catboost_train = features_train.copy()
    catboost_test = features_test.copy()
    for column in categorical_columns:
        catboost_train[column] = catboost_train[column].astype(str)
        catboost_test[column] = catboost_test[column].astype(str)
    catboost_model = CatBoostClassifier(
        iterations=300, depth=5, learning_rate=0.05, loss_function="MultiClass",
        random_seed=RANDOM_STATE, verbose=False, allow_writing_files=False,
    )
    catboost_model.fit(catboost_train, target_train, cat_features=categorical_columns)
    trained_models["CatBoost"] = catboost_model

    metric_rows = []
    for model_name, estimator in trained_models.items():
        test_features = catboost_test if model_name == "CatBoost" else features_test
        metric_rows.append(evaluate_model(model_name, estimator, test_features, target_test, RESULTS_DIR))
    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(RESULTS_DIR / "metrics.csv", index=False, float_format="%.4f")
    create_model_comparison(metrics, RESULTS_DIR / "plots")

    selected = choose_model(metrics)
    selected_name = str(selected["Model"])
    selected_estimator = trained_models[selected_name]
    feature_importance = save_feature_importance(selected_name, selected_estimator, RESULTS_DIR)
    artifact = {
        "model_type": "catboost_native" if selected_name == "CatBoost" else "sklearn_pipeline",
        "model_name": selected_name,
        "estimator": selected_estimator,
        "feature_columns": features.columns.tolist(),
        "categorical_columns": categorical_columns,
        "target_labels": TARGET_LABELS,
    }
    joblib.dump(artifact, MODELS_DIR / "final_model.joblib")
    selection_summary = {
        "selected_model": selected_name,
        "selection_priority": "Macro F1, then At Risk recall, High Performance recall, ROC-AUC, and Weighted F1.",
        "selected_metrics": {column: float(selected[column]) for column in metrics.columns if column != "Model"},
        "top_model_factors": feature_importance.head(10).to_dict(orient="records"),
        "test_split": {"test_size": 0.2, "random_state": RANDOM_STATE, "stratified": True},
        "leakage_exclusion": ["G1", "G2", "G3"],
    }
    (RESULTS_DIR / "model_selection.json").write_text(json.dumps(selection_summary, indent=2), encoding="utf-8")

    from src.predict import make_prediction
    label, probabilities = make_prediction(artifact, features_test.iloc[[0]])
    print(f"Completed training. Selected model: {selected_name}")
    print(f"Prediction smoke test: {label} | {probabilities}")


if __name__ == "__main__":
    main()
