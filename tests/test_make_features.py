import json
import re
import pandas as pd
import pytest

from features import make_features
from features.config import (
    DATA_PROCESSED, DATE_COL, SERIES_COL, TARGET_COL,
    TRAIN_FEATURES_PATH, TEST_FEATURES_PATH, FEATURES_META_PATH,
    CUTOFF, HORIZON, USE_CALENDAR_FEATURES
)

@pytest.mark.slow
def test_make_features_end_to_end():
    '''
    Integration test: runs pipeline and validates key properties of output
    requires data/process/store_item_demand_clean.csv exists beforehand
    '''

    make_features.main()

    assert TRAIN_FEATURES_PATH.exists()
    assert TEST_FEATURES_PATH.exists()
    assert json.loads(FEATURES_META_PATH.read_text())

    train = pd.read_parquet(TRAIN_FEATURES_PATH)
    test = pd.read_parquet(TEST_FEATURES_PATH)
    meta = json.loads(FEATURES_META_PATH.read_text())

    for df in (train, test):
        for col in (DATE_COL, SERIES_COL, TARGET_COL):
            assert col in df.columns, f"Missing required column {col}"
        assert pd.api.types.is_datetime64_any_dtype(df[DATE_COL]), f"{DATE_COL} not datetime"
        assert not df.duplicated(subset=[SERIES_COL, DATE_COL]).any(), "Duplicate (series,date) rows found"

    feature_cols = [c for c in train.columns if c not in (DATE_COL, SERIES_COL, TARGET_COL)]
    assert feature_cols, "No feature columns produced"
    assert any(c.startswith("lag_") for c in feature_cols), "No lag_* features produced"
    assert any(c.startswith("roll_mean_") for c in feature_cols), "No roll_mean_* features produced"

    if USE_CALENDAR_FEATURES:
        for c in ("dow", "weekofyear", "is_weekend"):
            assert c in feature_cols, f"Calendar feature {c} missing despite USE_CALENDAR_FEATURES=True"

    for df, name in ((train, "train"), (test, "test")):
        na_count = df[feature_cols].isna().sum().sum()
        assert na_count == 0, f"{name} has NA values in features: {na_count}"

    assert train[DATE_COL].max() <  CUTOFF, "Train contains dates on/after cutoff"
    assert test[DATE_COL].min()  >= CUTOFF, "Test contains dates before cutoff"

    if HORIZON and HORIZON > 0:
        per_series_counts = test.groupby(SERIES_COL).size().unique()
        assert len(per_series_counts) == 1, f"Test rows per series not uniform: {per_series_counts}"
        assert per_series_counts[0] == HORIZON, f"Expected {HORIZON} rows per series in test, got {per_series_counts[0]}"

    assert meta.get("cutoff") == str(pd.Timestamp(CUTOFF).date()), "Meta cutoff mismatch"
    assert meta.get("horizon") == HORIZON, "Meta horizon mismatch"
    assert isinstance(meta.get("feature_cols"), list) and meta["feature_cols"], "Meta feature_cols missing/empty"
    assert meta.get("rows_train") == len(train), "Meta rows_train mismatch"
    assert meta.get("rows_test")  == len(test),  "Meta rows_test mismatch"