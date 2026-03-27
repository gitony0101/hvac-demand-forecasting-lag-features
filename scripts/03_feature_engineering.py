from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils import TIME_COL, add_calendar_and_bucket_features


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
DAILY_PATH = DATA_DIR / "df_daily.csv"
FEATURE_PATH = DATA_DIR / "df_daily_feature_creation.csv"


def main() -> None:
    df_daily = pd.read_csv(DAILY_PATH)
    df_daily[TIME_COL] = pd.to_datetime(df_daily[TIME_COL])

    feature_df = add_calendar_and_bucket_features(df_daily)
    feature_df = feature_df.sort_values(TIME_COL)
    feature_df.to_csv(FEATURE_PATH, index=False)

    print(f"Saved feature dataset to: {FEATURE_PATH}")
    print("Shape:", feature_df.shape)
    print("Columns:")
    print(feature_df.columns.tolist())


if __name__ == "__main__":
    main()
