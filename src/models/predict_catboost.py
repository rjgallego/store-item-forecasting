from __future__ import annotations
from pathlib import Path

import argparse
import joblib
import pandas as pd 
import numpy as np

from src.features.config import (
    TEST_FEATURES_PATH, TARGET_COL,
    DATE_COL,SERIES_COL
)

def load_features(input_path: Path) -> pd.DataFrame:
    df = pd.read_parquet(input_path)
    return df

def load_model(model_path: Path):
    model = joblib.load(model_path)
    return model

def make_predictions(df: pd.DataFrame, model, log_target: bool = True) -> pd.DataFrame:
    exclude = [TARGET_COL, DATE_COL, SERIES_COL]
    X = df.drop(columns=exclude)

    if log_target:
        y_pred_log = model.predict(X)
        y_pred = np.expm1(y_pred_log)
    else:
        y_pred = model.predict(X)
    
    out = df[[SERIES_COL, DATE_COL]].copy()
    out['forecast'] = y_pred

    return out

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Catboost forecasts on a feature file and write predictions to CSV"
    )

    parser.add_argument(
        "--model-path",
        type=str,
        default="models/baseline_catboost.pkl",
        help="Path to the trained CatBoost model file (joblib pickle)"
    )

    parser.add_argument(
        "--input-path",
        type=str,
        default=str(TEST_FEATURES_PATH),
        help="Path to input features parquet file"
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default="reports/catboost_predictions.csv",
        help="Path to write predictions CSV"
    )

    parser.add_argument(
        "--no-log-target",
        action="store_true",
        help="Use flag if model was trained on raw target (no log1p)"
    )

    return parser.parse_args()

def main():
    args=parse_args()

    model_path = Path(args.model_path)
    input_path = Path(args.input_path)
    output_path = Path(args.output_path)

    print(input_path)

    log_target = not args.no_log_target

    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    print(f"Loading features from {input_path}...")
    df = load_features(input_path)
    
    print(f"Running predictions (log_target={log_target})...")
    preds_df = make_predictions(df, model, log_target=log_target)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    preds_df.to_csv(output_path, index=False)
    print(f"Saved predictions to {output_path}")
    print(f"Predictions shape: {preds_df.shape}")

if __name__ == "__main__":
    main()