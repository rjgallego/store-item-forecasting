import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib

from src.features.config import (
    TRAIN_FEATURES_PATH, TEST_FEATURES_PATH, 
    TARGET_COL, SERIES_COL, DATE_COL,
    DEFAULT_PARAMETERS
)

def load_feature_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(TRAIN_FEATURES_PATH)
    test = pd.read_parquet(TEST_FEATURES_PATH)

    return train, test

def get_train_test(train, test) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exclude = [TARGET_COL, SERIES_COL, DATE_COL]
    X_train = train.drop(columns=exclude)
    X_test = test.drop(columns=exclude)
    
    y_train_raw = train[TARGET_COL].values
    y_test = test[TARGET_COL]

    y_train = np.log1p(y_train_raw)

    return X_train, y_train, X_test, y_test

def train_lgbm(X_train, y_train, parameters = DEFAULT_PARAMETERS) -> lgb.LGBMRegressor:
    
    # create the model 
    model = lgb.LGBMRegressor(
        force_col_wise=True,
        n_estimators=parameters["n_estimators"],
        learning_rate=parameters["learning_rate"],
        random_state=parameters["random_state"],
        min_child_samples=parameters["min_child_samples"],
        max_depth=parameters["max_depth"]
    )

    # train the model
    model.fit(X_train, y_train)

    return model

def predict_lgbm(model, X_test):
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    return y_pred


def evaluate_predictions(y_true, y_pred) -> dict:

    # evaluate the model
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}") 

    return {
        "mae": mae,
        "rmse": rmse,
    }

def evaluate_series(test, y_pred):
    
    results = test[[SERIES_COL, TARGET_COL]].copy()
    results["pred"] = y_pred
    series_mae = results.groupby(SERIES_COL).apply(lambda g: mean_absolute_error(g[TARGET_COL], g["pred"]), include_groups=False)
    print(series_mae.describe()) 

    return series_mae

def naive_baseline(test_df, y_test) -> dict:
    naive_pred = test_df
    mae_naive = mean_absolute_error(y_test, naive_pred)
    rmse_naive = np.sqrt(mean_squared_error(y_test, naive_pred))
    print(f"Naive (lag_1) - MAE: {mae_naive:.3f}, RMSE: {rmse_naive:.3f}")
    return {
        "mae_naive": mae_naive,
        "rmse_naive": rmse_naive
    }

def run_baseline_experiment() -> dict:
    train, test = load_feature_data()
    X_train, y_train, X_test, y_test = get_train_test(train, test)
    model = train_lgbm(X_train, y_train)
    y_pred = predict_lgbm(model, X_test)
    metrics_dict = evaluate_predictions(y_test, y_pred)
    series_mae = evaluate_series(test, y_pred)
    naive_metrics_dict = naive_baseline(test["lag_1"], y_test)

    metrics = {
         "mae": float(metrics_dict["mae"]),
        "rmse": float(metrics_dict["rmse"]),
        "naive_mae": float(naive_metrics_dict["mae_naive"]),
        "naive_rmse": float(naive_metrics_dict["rmse_naive"]),
        "num_series": int(len(series_mae)),
        "mean_series_mae": float(series_mae.mean()),
        "median_series_mae": float(series_mae.median()),
    }

    return model, metrics, series_mae
