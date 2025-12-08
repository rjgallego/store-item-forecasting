from pathlib import Path
import pandas as pd

# --- Paths (relative to project root) ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS = PROJECT_ROOT / "reports"

# --- Time settings ---
FREQ = "D"                              # daily data
CUTOFF = pd.Timestamp("2017-01-01")     # last date included in train
HORIZON = 28                            # forecast horizon (days) for backtests

# --- Feature toggles ---
USE_CALENDAR_FEATURES = True    # dow, month, weekend
USE_HOLIDAYS = True            # enable later if you add holiday libs
COUNTRY_HOLIDAYS = "US"

# --- Target/columns ---
DATE_COL = "date"
SERIES_COL = "series_id"
TARGET_COL = "sales"

# --- Lag/rolling windows (daily) ---
LAG_WINDOWS = [1, 7, 14, 28]
ROLLING_MEANS = [7, 14, 28, 60, 180]
ROLLING_STDS = [7, 28]

# --- Data quality ---
ALLOW_GAPS = False              # if False, assert full daily coverage per series

# --- Modeling / evaluation ---
METRICS = ["MAE", "RMSE", "sMAPE"]  # baseline metrics to compute
RANDOM_STATE = 42

# --- Output file names ---
TRAIN_FEATURES_PATH = DATA_PROCESSED / "train_features.parquet"
TEST_FEATURES_PATH  = DATA_PROCESSED / "test_features.parquet"
FEATURES_META_PATH  = DATA_PROCESSED / "features_metadata.json"

# --- Default cutoff dates for backtesting --
DEFAULT_CUTOFFS = [
    "2015-01-01",
    "2015-04-01",
    "2015-07-01",
    "2015-10-01",
    "2016-01-01",
    "2016-04-01",
    "2016-07-01",
    "2016-10-01",
]

# --- Default params for LightGBM model --
DEFAULT_PARAMETERS = {
    "force_col_wise": True,
    "min_child_samples": 30,
    "n_estimators": 300,
    "learning_rate": 0.1,
    "random_state": 42,
    "max_depth": 6,
    "num_leaves": 31
}

# --- Default params for CatBoost model --
DEFAULT_PARAMETERS_CATBOOST = {
    "loss_function": "RMSE",
    "depth": 8,
    "learning_rate": 0.05,
    "iterations": 2000,
    "random_seed": 42,
    "l2_leaf_reg": 5
}