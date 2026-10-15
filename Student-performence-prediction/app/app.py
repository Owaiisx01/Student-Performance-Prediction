"""Streamlit interface for the saved student performance model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR))
from src.predict import make_prediction


MODEL_PATH = PROJECT_DIR / "models" / "final_model.joblib"
RESULTS_PATH = PROJECT_DIR / "results"


@st.cache_resource
def load_artifact():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_importance() -> pd.DataFrame:
    return pd.read_csv(RESULTS_PATH / "feature_importance.csv").head(10)


def yes_no(label: str, default: bool = True) -> str:
    return st.selectbox(label, ["yes", "no"], index=0 if default else 1)


def collect_inputs() -> dict[str, object]:
    """Collect only early-prediction inputs; grades G1/G2/G3 are never requested."""
    values: dict[str, object] = {}
    with st.expander("Student Information", expanded=True):
        first, second, third = st.columns(3)
        values["school"] = first.selectbox("School", ["GP", "MS"])
        values["sex"] = second.selectbox("Sex", ["F", "M"])
        values["age"] = third.slider("Age", 15, 22, 17)
        values["address"] = first.selectbox("Address", ["U", "R"])
        values["famsize"] = second.selectbox("Family size", ["LE3", "GT3"])
        values["Pstatus"] = third.selectbox("Parent cohabitation status", ["T", "A"])
    with st.expander("Family and Education", expanded=True):
        first, second, third = st.columns(3)
        values["Medu"] = first.slider("Mother's education", 0, 4, 2)
        values["Fedu"] = second.slider("Father's education", 0, 4, 2)
        values["Mjob"] = third.selectbox("Mother's occupation", ["at_home", "health", "services", "teacher", "other"])
        values["Fjob"] = first.selectbox("Father's occupation", ["at_home", "health", "services", "teacher", "other"])
        values["reason"] = second.selectbox("School choice reason", ["course", "home", "reputation", "other"])
        values["guardian"] = third.selectbox("Guardian", ["mother", "father", "other"])
        values["schoolsup"] = yes_no("School educational support", False)
        values["famsup"] = yes_no("Family educational support")
        values["nursery"] = yes_no("Attended nursery school")
        values["higher"] = yes_no("Desire for higher education")
    with st.expander("Study Information", expanded=True):
        first, second, third = st.columns(3)
        values["traveltime"] = first.slider("Travel time category", 1, 4, 2)
        values["studytime"] = second.slider("Weekly study time category", 1, 4, 2)
        values["failures"] = third.slider("Previous failures", 0, 4, 0)
        values["paid"] = yes_no("Paid classes", False)
        values["activities"] = yes_no("Extra-curricular activities")
        values["absences"] = st.slider("Absences", 0, 93, 4)
    with st.expander("Social and Lifestyle Information", expanded=True):
        first, second, third = st.columns(3)
        values["famrel"] = first.slider("Family relationship quality", 1, 5, 4)
        values["freetime"] = second.slider("Free time", 1, 5, 3)
        values["goout"] = third.slider("Going out", 1, 5, 3)
        values["Dalc"] = first.slider("Workday alcohol consumption", 1, 5, 1)
        values["Walc"] = second.slider("Weekend alcohol consumption", 1, 5, 1)
        values["health"] = third.slider("Current health", 1, 5, 3)
        values["internet"] = yes_no("Internet access")
        values["romantic"] = yes_no("Romantic relationship", False)
    return values


def main() -> None:
    st.set_page_config(page_title="Student Academic Performance Predictor", page_icon="🎓", layout="wide")
    st.title("Student Academic Performance Predictor")
    st.write("Predict a student's academic performance category using demographic, family, school, social and study-related information.")
    artifact = load_artifact()
    st.caption(f"Saved model: {artifact['model_name']}. G1, G2, and G3 are intentionally excluded for early prediction.")
    values = collect_inputs()
    if st.button("Predict Performance", type="primary"):
        prediction, probabilities = make_prediction(artifact, pd.DataFrame([values]))
        st.subheader(f"Performance: {prediction.upper()}")
        st.write(f"The model estimates the highest probability for the {prediction} category. This is a statistical estimate, not a guaranteed future result.")
        for label, probability in probabilities.items():
            st.write(f"**{label}:** {probability:.1%}")
            st.progress(float(probability))
    st.divider()
    st.subheader("Important Model Factors")
    st.caption("These are factors the selected model assigns more importance when making predictions; they do not establish causation.")
    st.dataframe(load_importance(), use_container_width=True, hide_index=True)
    if (RESULTS_PATH / "model_selection.json").exists():
        selection = json.loads((RESULTS_PATH / "model_selection.json").read_text(encoding="utf-8"))
        st.caption(selection["selection_priority"])


if __name__ == "__main__":
    main()
