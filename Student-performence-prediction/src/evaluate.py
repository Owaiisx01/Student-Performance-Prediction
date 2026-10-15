"""Evaluation, EDA, ROC, confusion-matrix, and importance utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize

from src.preprocessing import TARGET_LABELS


sns.set_theme(style="whitegrid", context="notebook")


def _save_figure(path: Path) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def create_eda_figures(dataframe: pd.DataFrame, target: pd.Series, output_dir: Path) -> None:
    """Create a focused, report-ready set of EDA figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    eda = dataframe.copy()
    eda["Performance"] = target.map(dict(enumerate(TARGET_LABELS)))
    order = TARGET_LABELS

    counts = eda["Performance"].value_counts().reindex(order)
    axis = counts.plot(kind="bar", color=["#d95f02", "#7570b3", "#1b9e77"])
    axis.set(title="Performance Category Distribution", xlabel="Performance category", ylabel="Students")
    for index, value in enumerate(counts):
        axis.text(index, value + 3, str(value), ha="center")
    _save_figure(output_dir / "target_distribution.png")

    for column, title in [("studytime", "Study Time vs Performance"), ("failures", "Previous Failures vs Performance")]:
        axis = sns.countplot(data=eda, x=column, hue="Performance", hue_order=order, palette="Set2")
        axis.set(title=title, xlabel=column, ylabel="Students")
        _save_figure(output_dir / f"{column}_vs_performance.png")

    axis = sns.boxplot(data=eda, x="Performance", y="absences", order=order, hue="Performance", palette="Set2", legend=False)
    axis.set(title="Absences vs Performance", xlabel="Performance category", ylabel="Absences")
    _save_figure(output_dir / "absences_vs_performance.png")

    categorical_columns = ["school", "sex", "higher", "internet", "schoolsup", "famsup"]
    melted = pd.concat(
        [eda.assign(Feature=column, Value=eda[column]) for column in categorical_columns], ignore_index=True
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    for axis, column in zip(axes.flat, categorical_columns):
        subset = melted[melted["Feature"] == column]
        sns.countplot(data=subset, x="Value", hue="Performance", hue_order=order, palette="Set2", ax=axis)
        axis.set_title(f"{column} by Performance")
        axis.set_xlabel(column)
        axis.set_ylabel("Students")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    for axis in axes.flat:
        if axis.get_legend() is not None:
            axis.get_legend().remove()
    figure.legend(handles, labels, loc="upper center", ncol=3)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(output_dir / "selected_categorical_features.png", dpi=180, bbox_inches="tight")
    plt.close(figure)

    numeric_columns = ["age", "Medu", "Fedu", "traveltime", "studytime", "failures", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]
    eda["target_code"] = target
    correlation = eda[numeric_columns + ["target_code"]].corr(numeric_only=True)
    figure, axis = plt.subplots(figsize=(12, 9))
    sns.heatmap(correlation, cmap="vlag", center=0, annot=False, square=True, ax=axis)
    axis.set_title("Correlation Heatmap: Numeric Variables and Performance Category (EDA Only)")
    _save_figure(output_dir / "correlation_heatmap.png")


def evaluate_model(
    model_name: str,
    estimator: Any,
    features_test: pd.DataFrame,
    target_test: pd.Series,
    results_dir: Path,
) -> dict[str, float | str]:
    """Compute metrics and save class reports, matrices, and ROC curves."""
    predictions = estimator.predict(features_test)
    probabilities = estimator.predict_proba(features_test)
    precision, recall, f1_score, _ = precision_recall_fscore_support(
        target_test, predictions, labels=[0, 1, 2], average=None, zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        target_test, predictions, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        target_test, predictions, average="weighted", zero_division=0
    )
    encoded_target = label_binarize(target_test, classes=[0, 1, 2])
    roc_auc = roc_auc_score(encoded_target, probabilities, multi_class="ovr", average="macro")
    metrics: dict[str, float | str] = {
        "Model": model_name,
        "Accuracy": accuracy_score(target_test, predictions),
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1": macro_f1,
        "Weighted F1": weighted_f1,
        "ROC-AUC": roc_auc,
        "At Risk Precision": precision[0],
        "At Risk Recall": recall[0],
        "At Risk F1": f1_score[0],
        "Average Precision": precision[1],
        "Average Recall": recall[1],
        "Average F1": f1_score[1],
        "High Performance Precision": precision[2],
        "High Performance Recall": recall[2],
        "High Performance F1": f1_score[2],
    }
    report = classification_report(target_test, predictions, target_names=TARGET_LABELS, output_dict=True, zero_division=0)
    report_path = results_dir / "reports" / f"{model_name.lower().replace(' ', '_')}_classification_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _create_confusion_matrix(model_name, target_test, predictions, results_dir / "confusion_matrices")
    _create_roc_curve(model_name, encoded_target, probabilities, results_dir / "plots")
    return metrics


def _create_confusion_matrix(model_name: str, actual: pd.Series, predicted: np.ndarray, output_dir: Path) -> None:
    matrix = confusion_matrix(actual, predicted, labels=[0, 1, 2])
    figure, axis = plt.subplots(figsize=(7, 5.5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=TARGET_LABELS, yticklabels=TARGET_LABELS, ax=axis)
    axis.set(title=f"{model_name} Confusion Matrix", xlabel="Predicted", ylabel="Actual")
    _save_figure(output_dir / f"{model_name.lower().replace(' ', '_')}.png")


def _create_roc_curve(model_name: str, actual_binary: np.ndarray, probabilities: np.ndarray, output_dir: Path) -> None:
    figure, axis = plt.subplots(figsize=(7, 5.5))
    for index, label in enumerate(TARGET_LABELS):
        false_positive_rate, true_positive_rate, _ = roc_curve(actual_binary[:, index], probabilities[:, index])
        axis.plot(false_positive_rate, true_positive_rate, label=label)
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Chance")
    axis.set(title=f"{model_name} One-vs-Rest ROC Curves", xlabel="False Positive Rate", ylabel="True Positive Rate")
    axis.legend(loc="lower right")
    _save_figure(output_dir / f"{model_name.lower().replace(' ', '_')}_roc.png")


def create_model_comparison(metrics: pd.DataFrame, output_dir: Path) -> None:
    """Plot actual comparative accuracy and F1/AUC metrics."""
    for metric in ["Accuracy", "Macro F1", "Weighted F1", "ROC-AUC"]:
        axis = sns.barplot(data=metrics, x="Model", y=metric, hue="Model", palette="Set2", legend=False)
        axis.set(title=f"Model Comparison: {metric}", xlabel="Model", ylabel=metric, ylim=(0, 1))
        axis.tick_params(axis="x", rotation=15)
        _save_figure(output_dir / f"comparison_{metric.lower().replace(' ', '_').replace('-', '_')}.png")


def save_feature_importance(model_name: str, estimator: Any, output_dir: Path) -> pd.DataFrame:
    """Extract and visualize selected-model feature importance."""
    if model_name == "CatBoost":
        values = estimator.get_feature_importance()
        names = estimator.feature_names_
    else:
        pipeline = estimator
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            values = model.feature_importances_
        else:
            values = np.abs(model.coef_).mean(axis=0)
        names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importance = pd.DataFrame({"Feature": names, "Importance": values}).sort_values("Importance", ascending=False)
    importance.to_csv(output_dir / "feature_importance.csv", index=False)
    plot_data = importance.head(15).sort_values("Importance")
    axis = sns.barplot(data=plot_data, x="Importance", y="Feature", hue="Feature", palette="viridis", legend=False)
    axis.set(title=f"Top Model Factors: {model_name}", xlabel="Model importance", ylabel="Feature")
    _save_figure(output_dir / "plots" / "selected_model_feature_importance.png")
    return importance
