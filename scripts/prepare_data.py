from pathlib import Path
from src.data.schema import load_and_validate, add_series_id

RAW = Path("data/raw/train.csv")
PROCESSED = Path("data/processed/store_item_demand_clean.csv")

def main():
    df = load_and_validate(RAW)
    df = add_series_id(df)
    PROCESSED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED, index=False)
    print(f"Saved {len(df):,} rows to {PROCESSED}")

if __name__ == "__main__":
    main()