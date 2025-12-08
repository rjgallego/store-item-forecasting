import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, Dict
import json

from src.features.config import (
    TRAIN_FEATURES_PATH, TEST_FEATURES_PATH, 
    TARGET_COL, SERIES_COL, DATE_COL,
    DEFAULT_CUTOFFS, HORIZON
)

def load_backtest_frame(path) -> pd.DataFrame: 
    path = Path(path)
    df = pd.read_parquet(path)
    df.sort_values([SERIES_COL, DATE_COL]).reset_index(drop=True)

    return df

def make_backtest_windows(df: pd.DataFrame, cutoffs: list[str] = DEFAULT_CUTOFFS, horizon: int = HORIZON) -> list[dict]:
    windows = []
    for c in cutoffs:
        cutoff = pd.Timestamp(c)
        train_df = df[df[DATE_COL] < cutoff].copy()
        test_df = df[(df[DATE_COL] >= cutoff) & (df[DATE_COL] < cutoff + pd.Timedelta(days=horizon))].copy()

        windows.append({
            "cutoff": cutoff,
            "train_df": train_df,
            "test_df": test_df
        })

    return windows

def extract_X_Y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]: 
    exclude = [TARGET_COL, DATE_COL, SERIES_COL]
    X = df.drop(columns=exclude)
    y_raw = df[TARGET_COL].values

    return X, y_raw

def evaluate_window(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> dict:

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    return {
        "mae": mae,
        "rmse": rmse
    }

def naive_baseline(test_df: pd.DataFrame, y_test: pd.DataFrame) -> dict:
    naive_pred = test_df
    mae_naive = mean_absolute_error(y_test, naive_pred)
    rmse_naive = np.sqrt(mean_squared_error(y_test, naive_pred))
    
    return {
        "mae_naive": mae_naive,
        "rmse_naive": rmse_naive
    }

def save_backtest_results(results_df: pd.DataFrame, summary_metrics: dict, path_results: str, path_summary: str) -> None:
    results_path = Path(path_results)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(results_path)

    summary_path = Path(path_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary_metrics, f, indent=4)

    return