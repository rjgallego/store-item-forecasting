import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple, Dict

from src.features.config import (
    TRAIN_FEATURES_PATH, TEST_FEATURES_PATH, 
    TARGET_COL, SERIES_COL, DATE_COL
)

def load_feature_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_parquet(TRAIN_FEATURES_PATH)
    test = pd.read_parquet(TEST_FEATURES_PATH)

    return train, test

def get_X_y(train: pd.DataFrame, test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    exclude = [TARGET_COL, SERIES_COL, DATE_COL]
    X_train = train.drop(columns=exclude)
    X_test = test.drop(columns=exclude)
    
    y_train_raw = train[TARGET_COL].to_numpy()
    y_test_raw = test[TARGET_COL].to_numpy()

    return X_train, y_train_raw, X_test, y_test_raw

def evaluate_predictions(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> Dict[float, float]:

    # evaluate the model
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"MAE: {mae:.3f}")
    print(f"RMSE: {rmse:.3f}") 

    return {
        "mae": float(mae),
        "rmse": float(rmse),
    }