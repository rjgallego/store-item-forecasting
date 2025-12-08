import pandas as pd
import pytest
from pandera.errors import SchemaError
from data.schema import STORE_ITEM_SCHEMA

def test_schema_accepts_valid():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2017-01-01", "2017-01-02"]),
        "store": [1, 1],
        "item": [1, 1],
        "sales": [10, 12]
    })
    validated = STORE_ITEM_SCHEMA.validate(df)
    assert len(validated) == 2

def test_schema_rejects_negative_sales():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2017-01-01"]),
        "store": [1],
        "item": [1],
        "sales": [-5],
    })
    with pytest.raises(SchemaError):
        STORE_ITEM_SCHEMA.validate(df)

def test_schema_rejects_extra_column():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2017-01-01"]),
        "store": [1],
        "item": [1],
        "sales": [5],
        "extra": [123],
    })
    import pandera as pa
    with pytest.raises(SchemaError):
        STORE_ITEM_SCHEMA.validate(df)