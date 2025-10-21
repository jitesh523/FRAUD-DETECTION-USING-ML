# Fraud Detection Using ML

This repo contains a notebook `Untitled-7.ipynb` that:
- Builds a robust scikit-learn Pipeline with preprocessing (scaling + one-hot for low-cardinality)
- Trains Logistic Regression with class weights
- Tunes hyperparameters, calibrates probabilities, and evaluates with ROC/PR curves
- Adds business-driven threshold selection (precision ≥ 90%)
- Provides alternative model pipeline (XGBoost if installed, else RandomForest)
- Saves artifacts for inference

## Quick Start

1) Install dependencies:
```bash
pip install -r requirements.txt
```

2) Run the notebook `Untitled-7.ipynb` to train and produce artifacts in `artifacts/`.
- Key cells:
  - Preprocessing + pipeline build
  - Evaluation (ROC/PR + threshold tuning)
  - Calibration and artifact saving

3) Score a new CSV with the saved model using `predict.py`:
```bash
python predict.py --input path/to/new_data.csv --output scored_output.csv --threshold-type business
```
- `--threshold-type` can be `business`, `f1`, or `custom` (use with `--threshold`).

## Artifacts

- `artifacts/calibrated_pipeline.joblib`: Calibrated sklearn Pipeline
- `artifacts/model_info.json`: Saved thresholds and metrics

## Notes

- If your data has a temporal field `step`, the notebook supports a time-based split to reduce leakage.
- For large datasets, consider switching OneHotEncoder to `sparse=True`.
- Alternative model (XGBoost/RandomForest) is included for comparison and may improve recall at high precision.
