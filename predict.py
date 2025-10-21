#!/usr/bin/env python3
import argparse
import json
import os
import sys
import pandas as pd
import joblib

ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), 'artifacts')
MODEL_PATH = os.path.join(ARTIFACTS_DIR, 'calibrated_pipeline.joblib')
INFO_PATH = os.path.join(ARTIFACTS_DIR, 'model_info.json')


def load_artifacts():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: model artifact not found at {MODEL_PATH}", file=sys.stderr)
        sys.exit(1)
    model = joblib.load(MODEL_PATH)

    thresholds = {}
    if os.path.exists(INFO_PATH):
        with open(INFO_PATH, 'r') as f:
            info = json.load(f)
        thresholds = {
            'f1': info.get('best_threshold_f1_calibrated'),
            'business': info.get('business_threshold_precision_90_calibrated')
        }
    else:
        print(f"Warning: {INFO_PATH} not found. Thresholds will default to 0.5", file=sys.stderr)
        thresholds = {'f1': 0.5, 'business': 0.5}

    # Fallback if keys are missing
    for k in ['f1', 'business']:
        if thresholds.get(k) is None:
            thresholds[k] = 0.5
    return model, thresholds


def main():
    parser = argparse.ArgumentParser(description='Score transactions CSV using saved calibrated pipeline.')
    parser.add_argument('--input', required=True, help='Path to input CSV file')
    parser.add_argument('--output', required=False, default='scored_output.csv', help='Path to write scored CSV')
    parser.add_argument('--threshold-type', choices=['business', 'f1', 'custom'], default='business', help='Which threshold to use for the binary flag')
    parser.add_argument('--threshold', type=float, default=0.5, help='Custom threshold if --threshold-type=custom')
    args = parser.parse_args()

    model, thresholds = load_artifacts()

    df = pd.read_csv(args.input)

    # Predict probabilities
    try:
        proba = model.predict_proba(df)[:, 1]
    except Exception as e:
        print('Error while predicting. Ensure input columns match training expectations.', file=sys.stderr)
        raise

    if args.threshold_type == 'custom':
        thr = args.threshold
    else:
        thr = thresholds[args.threshold_type]

    pred = (proba >= thr).astype(int)

    out = df.copy()
    out['fraud_proba'] = proba
    out['fraud_flag'] = pred

    out.to_csv(args.output, index=False)
    print(f"Wrote scored output to {args.output} using threshold={thr} ({args.threshold_type})")


if __name__ == '__main__':
    main()
