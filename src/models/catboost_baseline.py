from __future__ import annotations
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from catboost import CatBoostRegressor


from src.features.config import (
    TRAIN_FEATURES_PATH, TEST_FEATURES_PATH, 
    TARGET_COL, SERIES_COL, DATE_COL
)

from src.models.baseline_helpers import (
    load_feature_data, get_X_y,
    evaluate_predictions
)


def train_model(X_train: pd.DataFrame, y_train: pd.DataFrame, params: Dict[str, Any] | None = None) -> CatBoostRegressor:

    # create the model 
    model = CatBoostRegressor(
        loss_function="RMSE",
        depth=8,
        learning_rate=0.05,
        iterations=2000,
        random_seed=42,
        l2_leaf_reg=5,
        verbose=False
    )

    # train the model
    model.fit(X_train, y_train)

    return model

def run_catboost_experiment():

    train, test = load_feature_data()
    X_train, y_train_raw, X_test, y_test_raw = get_X_y(train, test)

    y_train_log = np.log1p(y_train_raw)

    model = train_model(X_train, y_train_log)

    y_pred_log = model.predict(X_test)
    y_pred_raw = np.expm1(y_pred_log)

    metrics = evaluate_predictions(y_test_raw, y_pred_raw)

    if "lag_1" in test.columns:
        naive_pred = test["lag_1"].to_numpy()
        naive_metrics = evaluate_predictions(y_test_raw, naive_pred)
        metrics["naive_mae"] = naive_metrics["mae"]
        metrics["naive_rmse"] = naive_metrics["rmse"]

    results = test[[SERIES_COL, TARGET_COL]].copy()
    results["pred"] = y_pred_raw
    series_mae = (
        results.groupby(SERIES_COL).apply(lambda g: mean_absolute_error(g[TARGET_COL], g["pred"]))
    )

    metrics["mean_series_mae"] = float(series_mae.mean())
    metrics["median_series_mae"] = float(series_mae.median())

    return model, metrics, series_mae
