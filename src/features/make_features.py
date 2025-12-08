from __future__ import annotations
import pandera.pandas as pa
from pandera.pandas import DataFrameSchema, Column, Check
import pandas as pd
import json
from pathlib import Path
from pandas.tseries.frequencies import to_offset
from pandas.tseries.holiday import USFederalHolidayCalendar
import numpy as np

from src.features.config import (
    DATA_PROCESSED, DATE_COL, SERIES_COL, TARGET_COL,
    FREQ, CUTOFF, HORIZON, LAG_WINDOWS, ROLLING_MEANS, ROLLING_STDS,
    USE_CALENDAR_FEATURES, USE_HOLIDAYS, COUNTRY_HOLIDAYS,
    TRAIN_FEATURES_PATH, TEST_FEATURES_PATH, FEATURES_META_PATH,
    METRICS, RANDOM_STATE, ALLOW_GAPS
)

STORE_ITEM_SCHEMA_PROCESSED = DataFrameSchema(
    {
        DATE_COL: Column(pa.DateTime, Check(lambda s: s.notnull())),
        SERIES_COL: Column(str, Check(lambda s: s.notnull())),
        "store": Column(int, Check(lambda s: s.between(1, 10))),
        "item": Column(int, Check(lambda s: s.between(1, 50))),
        TARGET_COL: Column(int, Check(lambda s: s>=0)),
    },
    coerce=True,
    strict=True,
    name="store_item_schema"
)

def load_processed_data(path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path, parse_dates = ["date"])
    df.sort_values(["series_id", "date"]).reset_index(drop=True)

    df = STORE_ITEM_SCHEMA_PROCESSED.validate(df)
    return df


def assert_daily_continuity(df: pd.DataFrame, freq: str = FREQ) -> None:

    df = df.sort_values([SERIES_COL, DATE_COL]).copy()

    # duplicate timestamp check per series
    dup_mask = df.duplicated(subset=[SERIES_COL, DATE_COL])
    if dup_mask.any():
        dup = df.loc[dup_mask, [SERIES_COL, DATE_COL]].head(10)
        raise AssertionError(
            f"Duplicate timestamps within series detected (showing up to 10):\n{dup}"
        )
    
    # Check time is only going forward (not backward)
    
    bad_order = (
        df.groupby(SERIES_COL, sort=False)[DATE_COL]
        .apply(lambda s: (s.diff() < pd.Timedelta(0)).any())
    )
    if bad_order.any():
        offenders = bad_order[bad_order].index.tolist()[:10]
        raise AssertionError(
            f"Non-monotonic dates (time going backwards) in series: {offenders} (showing up to 10)"
        )
    
    # Check for gaps against expected frequency
    offset = to_offset(freq)
    expected_delta = pd.to_timedelta(offset)

    diff = df.groupby(SERIES_COL, sort=False)[DATE_COL].diff()
    gap_mask = (~diff.isna()) & (diff != expected_delta)
    if gap_mask.any():
        gap_rows = df.loc[gap_mask, [SERIES_COL, DATE_COL]].copy()
        gap_rows['prev_date'] = (
            df.groupby(SERIES_COL, sort=False)[DATE_COL].shift(1)
        )[gap_mask]
        gap_rows["missing_steps"] = (
            (gap_rows[DATE_COL] - gap_rows['prev_date']) / expected_delta
        ).astype(int) - 1

        total_gaps = len(gap_rows)
        total_missing = int(gap_rows['missing_steps'], ascending=False).head(5).to_string(index=False)
        raise AssertionError(
            "Detected gaps in the time index within series.\n"
            f"- Total gaps: {total_gaps}\n"
            f"- Estimated missing steps: {total_missing}\n"
            f"- Examples (series_id, date, prev_date, missing_steps):\n{sample}\n"
            "If gaps are expected, set ALLOW_GAPS=True in config or fill them before feature generation."
        )

    return None

def add_calendar_features(df) -> pd.DataFrame:
    df["dow"] = df[DATE_COL].dt.dayofweek
    df["is_weekend"] = (df[DATE_COL].dt.dayofweek > 4).astype(int)
    df["weekofyear"] = df[DATE_COL].dt.isocalendar().week.astype(int)
    df["month"] = df[DATE_COL].dt.month
    df["quarter"] = df[DATE_COL].dt.quarter
    df["dom"] = df[DATE_COL].dt.day

    df["is_month_start"] = df[DATE_COL].dt.is_month_start.astype(int)
    df["is_month_end"] = df[DATE_COL].dt.is_month_end.astype(int)

    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)
        
    return df.sort_values([SERIES_COL, DATE_COL]).reset_index(drop=True)


def add_cycle_encoding(df) -> pd.DataFrame:
    df["week_sin"] = np.sin(2 * np.pi * df["weekofyear"] / 52)
    df["week_cos"] = np.cos(2 * np.pi * df["weekofyear"] / 52)

    df["doy"] = df["date"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * df["doy"] / 365)
    df["doy_cos"] = np.cos(2 * np.pi * df["doy"] / 365)
        
    return df.sort_values([SERIES_COL, DATE_COL]).reset_index(drop=True)


def add_holiday_features(df) -> pd.DataFrame:
    cal = USFederalHolidayCalendar()
    holidays = cal.holidays(start=df[DATE_COL].min(), end=df[DATE_COL].max())

    df["is_holiday"] = df[DATE_COL].isin(holidays).astype(int)
    df["days_to_holiday"] = df[DATE_COL].apply(lambda d: (holidays - d).days.min())

    return df.sort_values([SERIES_COL, DATE_COL]).reset_index(drop=True)


def add_lag_features(df, lags: list[int] = LAG_WINDOWS) -> pd.DataFrame:
    # ensure the dataframe is sorted to for correct lagging
    df = df.sort_values([SERIES_COL, DATE_COL]).reset_index(drop=True)

    #validation that date is sorted
    assert not(
        df.groupby(SERIES_COL)[DATE_COL].apply(lambda x: (x.diff().dt.days < 0).any()).any()
    ), "Error: at least one series has unsorted or duplicate dates"

    g = df.groupby(SERIES_COL, sort=False)[TARGET_COL]
    for w in lags:
        df[f"lag_{w}"] = g.shift(w)
    return df


def add_rolling_features(df, means: list[int] = ROLLING_MEANS, stds: list[int] = ROLLING_STDS) -> pd.DataFrame:
    df = df.sort_values([SERIES_COL, DATE_COL]).reset_index(drop=True)

    # validation that date is sorted
    assert not(
        df.groupby(SERIES_COL)[DATE_COL].apply(lambda x: (x.diff().dt.days < 0).any()).any()
    ), "Error: at least one series has unsorted or duplicate dates"

    g = df.groupby(SERIES_COL, sort=False)[TARGET_COL]
    for w in means:
        df[f"roll_mean_{w}"] = g.rolling(window=w, min_periods=w).mean().reset_index(level=0, drop=True)
    for w in stds:
        df[f"roll_std_{w}"] = g.rolling(window=w, min_periods=w).std().reset_index(level=0, drop=True)
    return df

def finalize_and_split(df) -> tuple[pd.DataFrame, pd.DataFrame, dic]:
    df = df.sort_values([SERIES_COL, DATE_COL]).reset_index(drop=True)

    # Create list of feature columns
    protected = [DATE_COL, SERIES_COL, TARGET_COL]
    feature_cols = [c for c in df.columns if c not in protected]

    # drop rows with NA in any of the feature columns
    na_before = int(df[feature_cols].isna().sum().sum())
    if na_before:
        df = df.dropna(subset=feature_cols).copy()

    # ensure target column is numeric data type
    df[TARGET_COL] = pd.to_numeric(df[TARGET_COL], errors="raise")

    # split into training and test sets
    train_df = df[df[DATE_COL] < CUTOFF].copy()
    test_df = df[df[DATE_COL] >= CUTOFF].copy()

    # Limit test to a single evaluation horizon
    # If you want the *first* HORIZON days per series after cutoff:
    if HORIZON is not None and HORIZON > 0:
        test_df = (
            test_df
            .assign(_rank=test_df.groupby(SERIES_COL)[DATE_COL].rank(method="first"))
            .query("_rank <= @HORIZON")
            .drop(columns="_rank")
        )

    meta = {
        "freq": FREQ,
        "cutoff": str(pd.Timestamp(CUTOFF).date()),
        "horizon": HORIZON,
        "n_series_total": int(df[SERIES_COL].nunique()),
        "rows_total": int(len(df)),
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "lag_windows": LAG_WINDOWS,
        "rolling_means": ROLLING_MEANS,
        "rolling_stds": ROLLING_STDS,
        "use_calendar": USE_CALENDAR_FEATURES,
        "use_holidays": USE_HOLIDAYS,
        "target": TARGET_COL,
        "feature_cols": feature_cols,
        "columns_out": df.columns.tolist(),
        "date_min": str(df[DATE_COL].min().date()),
        "date_max": str(df[DATE_COL].max().date()),
    }

    return train_df, test_df, meta


def save_artifacts(train_df, test_df, meta: dict) -> None:
    TRAIN_FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        train_df.to_parquet(TRAIN_FEATURES_PATH, index=False)
        test_df.to_parquet(TEST_FEATURES_PATH, index=False)
    except ImportError:
        print("⚠️ Parquet engine missing — saving as CSV instead.")
        train_df.to_csv(TRAIN_FEATURES_PATH.with_suffix(".csv"), index=False)
        test_df.to_csv(TEST_FEATURES_PATH.with_suffix(".csv"), index=False)

    with open(FEATURES_META_PATH, "w") as f:
        json.dump(meta, f)

    print(
        f"Saved:\n"
        f"- {TRAIN_FEATURES_PATH.name}: {train_df.shape}\n"
        f"- {TEST_FEATURES_PATH.name}:  {test_df.shape}\n"
        f"- {FEATURES_META_PATH.name}"
    )

def main():
    df = load_processed_data(DATA_PROCESSED / "store_item_demand_clean.csv")
    assert_daily_continuity(df)

    if(USE_CALENDAR_FEATURES):
        df = add_calendar_features(df)
        df = add_cycle_encoding(df)
    
    if(USE_HOLIDAYS):
        df = add_holiday_features(df)
    
    df = add_lag_features(df)
    df = add_rolling_features(df)
    df = pd.get_dummies(df, columns=["store", "item"])
    train_df, test_df, meta = finalize_and_split(df)
    save_artifacts(train_df, test_df, meta)

if __name__ == "__main__":
    main()