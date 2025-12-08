import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from pathlib import Path
import joblib
from catboost import CatBoostRegressor
import json

from src.features.config import (
    TRAIN_FEATURES_PATH, TEST_FEATURES_PATH, 
    TARGET_COL, SERIES_COL, DATE_COL,
    DEFAULT_CUTOFFS, DEFAULT_PARAMETERS_CATBOOST, HORIZON
)

from src.eval.backtest_helpers import (
    load_backtest_frame, make_backtest_windows,
    extract_X_Y, evaluate_window, naive_baseline,
    save_backtest_results
)


def fit_model_for_windows(X_train: pd.DataFrame, y_train: pd.DataFrame, model_params: dict = DEFAULT_PARAMETERS_CATBOOST) -> CatBoostRegressor:

    model = CatBoostRegressor(
        loss_function=model_params["loss_function"],
        depth=model_params["depth"],
        learning_rate=model_params["learning_rate"],
        iterations=model_params["iterations"],
        random_seed=model_params["random_seed"],
        l2_leaf_reg=model_params["l2_leaf_reg"],
        verbose=False
    )

    model.fit(X_train, y_train)

    return model

def run_rolling_backtest(cutoffs: list[str] = DEFAULT_CUTOFFS, horizon: int = HORIZON, model_params: dict = DEFAULT_PARAMETERS_CATBOOST) -> tuple[pd.DataFrame, dict]:
    df = load_backtest_frame(TRAIN_FEATURES_PATH)
    windows = make_backtest_windows(df, cutoffs, horizon)
    results = []
    for w in windows:
        train_df = w["train_df"]
        test_df = w["test_df"]

        X_train, y_train_raw = extract_X_Y(train_df)
        X_test, y_test_raw = extract_X_Y(test_df)
        
        y_train = np.log1p(y_train_raw)

        model = fit_model_for_windows(X_train, y_train, model_params)

        y_pred_log = model.predict(X_test)
        y_pred = np.expm1(y_pred_log)

        metrics = evaluate_window(y_test_raw, y_pred)
        naive_1 = test_df["lag_1"].to_numpy()
        naive_7 = test_df["lag_7"].to_numpy()
        naive_28 = test_df["lag_28"].to_numpy()

        naive1_metrics = naive_baseline(naive_1, y_test_raw)
        naive7_metrics = naive_baseline(naive_7, y_test_raw)
        naive28_metrics = naive_baseline(naive_28, y_test_raw)

        mean_sales = test_df["sales"].mean()
        mae = metrics["mae"]
        error = (mae / mean_sales) * 100

        results.append({
            "cutoff": w["cutoff"],
            "train_end": train_df[DATE_COL].max(),
            "test_start": test_df[DATE_COL].min(),
            "test_end": test_df[DATE_COL].max(),
            "n_train_rows": len(train_df),
            "n_test_rows": len(test_df),
            "mae": metrics["mae"],
            "rmse": metrics["rmse"],
            "error": error,
            "mae_naive1": naive1_metrics["mae_naive"],
            "rmse_naive1": naive1_metrics["rmse_naive"],
            "mae_naive7": naive7_metrics["mae_naive"],
            "rmse_naive7": naive7_metrics["rmse_naive"],
            "mae_naive28": naive28_metrics["mae_naive"],
            "rmse_naive28": naive28_metrics["rmse_naive"]
        })
    results_df = pd.DataFrame(results).sort_values("cutoff").reset_index(drop=True)
    summary = {
        "n_windows": int(len(results_df)),
        "mae_mean": float(results_df["mae"].mean()),
        "mae_std": float(results_df["mae"].std()),
        "rmse_mean": float(results_df["rmse"].mean()),
        "rmse_std": float(results_df["rmse"].std()),
        "mae_naive1_mean": float(results_df["mae_naive1"].mean()),
        "rmse_naive1_mean": float(results_df["rmse_naive1"].mean()),
        "mae_naive7_mean": float(results_df["mae_naive7"].mean()),
        "rmse_naive7_mean": float(results_df["rmse_naive7"].mean()),
        "mae_naive28_mean": float(results_df["mae_naive28"].mean()),
        "rmse_naive28_mean": float(results_df["rmse_naive28"].mean())
    }

    return results_df, summary

def main():
    RESULTS_PATH = 'reports/backtest_rolling_windows_cb.csv'
    SUMMARY_PATH = 'reports/backtest_summary_catboost.json'

    results_df, summary = run_rolling_backtest()
    print("Backtest summary: ", summary)
    print(results_df.head())
    save_backtest_results(results_df, summary, RESULTS_PATH, SUMMARY_PATH)

if __name__ == "__main__":
    main()