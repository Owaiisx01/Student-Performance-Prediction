# Student Academic Performance Prediction

A reproducible machine-learning mini-project that predicts a student's **At Risk**, **Average**, or **High Performance** category from demographic, family, school, social, study, and lifestyle information.

## Problem and objective

The project demonstrates a leakage-safe early-prediction workflow. It compares Logistic Regression, Random Forest, XGBoost, and CatBoost, then selects the saved Streamlit model using actual held-out test results. Macro F1 is prioritized because the classes are not equally represented.

## Dataset and target

- Source: supplied `student-mat.csv` (Portuguese secondary-school mathematics course dataset)
- Observations: 395 students; original columns: 33
- Target from final grade `G3`: `0–9 = At Risk`, `10–14 = Average`, and `15–20 = High Performance`
- Early-prediction rule: `G1`, `G2`, and `G3` are never included in model inputs.

## Data quality

The executed audit is saved in `results/data_audit.md` and `results/data_audit.json`. It programmatically reports dimensions, data types, missing values, duplicates, categories, numerical ranges, range validation, and target distribution. Legitimate observations, including high absence counts, are retained.

## Method

1. Create the target and remove `G1`, `G2`, and `G3` before splitting.
2. Create an 80/20 stratified train/test split using `random_state=42`.
3. Fit preprocessing only on the training data. Logistic Regression receives scaled numeric columns and one-hot categorical columns; tree pipelines use one-hot categorical columns without scaling. CatBoost receives native categorical columns.
4. Evaluate all four models only on the untouched test set with accuracy, precision, recall, F1, macro metrics, weighted F1, multiclass one-vs-rest ROC-AUC, confusion matrices, ROC curves, and class reports.
5. Select and serialize the final model using Macro F1, At Risk recall, High Performance recall, ROC-AUC, and weighted F1 in that priority order.

## Results and explainability

The following results come from the executed stratified 20% test split (`random_state=42`), not from training data:

| Model | Accuracy | Macro Precision | Macro Recall | Macro F1 | Weighted F1 | ROC-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Logistic Regression | 0.4430 | 0.3965 | 0.3838 | 0.3849 | 0.4313 | 0.6233 |
| Random Forest | 0.4937 | 0.4974 | 0.3892 | 0.3699 | 0.4303 | 0.6144 |
| XGBoost | 0.4810 | 0.4577 | 0.4114 | 0.4097 | 0.4486 | 0.6043 |
| CatBoost | 0.5696 | 0.5441 | 0.5146 | 0.5230 | 0.5605 | 0.6433 |

CatBoost is selected because it has the highest Macro F1, as well as the strongest overall weighted F1 and ROC-AUC in this experiment. Its At Risk recall is 0.5000 and High Performance recall is 0.3333; these weaker class-specific results remain visible in `results/metrics.csv` and the saved classification reports.

The generated `results/` directory contains metrics, confusion matrices, ROC curves, EDA figures, class-wise reports, selection rationale, and selected-model feature importance. Importance shows which factors the model relies on; it does not demonstrate causation.

## Installation and use

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
pip install -r requirements.txt
python -m src.train
streamlit run app/app.py
```

The Streamlit app loads `models/final_model.joblib`; it never retrains at startup and never requests `G1`, `G2`, or `G3`.

## Project structure

```text
app/                 Streamlit UI
data/raw/            Supplied source dataset
models/              Saved final artifact
notebooks/           Guided notebook entry points
results/             Audit, metrics, reports, figures, and matrices
src/                 Loading, preprocessing, training, evaluation, prediction
```

## Deployment

Push the repository to GitHub and deploy with Streamlit Community Cloud by setting the app entry point to `app/app.py`. No credentials or private services are required.

## Limitations and future work

- The dataset is small and comes from its original educational context, so generalization to other institutions is not guaranteed.
- Excluding `G1` and `G2` is intentional for early prediction but limits predictive signal.
- Predictions are statistical category estimates, not definitive assessments of a student's ability.
- Feature importance and correlations do not establish causation.
- Future work could include external validation, fairness analysis, educator review, and carefully governed data updates.

## Literature review

No literature citations are included here because the brief requires verified, non-fabricated recent papers. Add a separate, source-verified literature review before submitting an academic report.
