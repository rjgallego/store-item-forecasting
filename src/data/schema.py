from __future__ import annotations
from pathlib import Path
import pandas as pd
import pandera.pandas as pa
from pandera.pandas import Column, DataFrameSchema, Check


#Public schema: import into notebooks/scripts
STORE_ITEM_SCHEMA = DataFrameSchema(
    {
        "date": Column(pa.DateTime, Check(lambda s: s.notnull())),
        "store": Column(int, Check(lambda s: s.between(1, 10))),
        "item": Column(int, Check(lambda s: s.between(1, 50))),
        "sales": Column(int, Check(lambda s: s>=0))
    },
    coerce=True,
    strict=True,
    name="store_item_demand"
)

def load_and_validate(path: str | Path) -> pd.DataFrame:
    """
    Load the CSV file and validate against STORE_ITEM_SCHEMA
    Returns a DataFrame with coerced dtypes if validation passes.
    Raises pandera.errors.SchemaError if validation fails
    """

    path = Path(path)
    df = pd.read_csv(path, parse_dates = ["date"])
    df = df.sort_values(["store", "item", "date"]).reset_index(drop=True)
    df = STORE_ITEM_SCHEMA.validate(df)
    return df

def add_series_id(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a 'series_id' (store_item) and returns a new DataFrame with canonial column order
    """
    out = df.copy()
    out["series_id"] = out["store"].astype(str) + "_" + out["item"].astype(str)
    cols = ["date", "series_id", "store", "item", "sales"]
    return out[cols].sort_values(["series_id", "date"]).reset_index(drop=True)