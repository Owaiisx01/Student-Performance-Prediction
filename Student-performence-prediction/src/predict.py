"""Prediction helpers shared by the Streamlit application and tests."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.preprocessing import TARGET_LABELS


def make_prediction(artifact: dict[str, Any], input_dataframe: pd.DataFrame) -> tuple[str, dict[str, float]]:
    """Return a human-readable prediction and class probabilities."""
    expected_columns = artifact["feature_columns"]
    features = input_dataframe.loc[:, expected_columns].copy()
    if artifact["model_type"] == "catboost_native":
        for column in artifact["categorical_columns"]:
            features[column] = features[column].astype(str)
    estimator = artifact["estimator"]
    predicted_code = int(np.asarray(estimator.predict(features)).reshape(-1)[0])
    probabilities = estimator.predict_proba(features)[0]
    probability_map = {label: float(probability) for label, probability in zip(TARGET_LABELS, probabilities)}
    return TARGET_LABELS[predicted_code], probability_map
