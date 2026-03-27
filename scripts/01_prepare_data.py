from __future__ import annotations

from pathlib import Path

from src.utils import aggregate_to_daily, load_raw_data, preprocess_raw_data


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "Load_data_new.csv"
PROCESSED_HOURLY_PATH = DATA_DIR / "processed_data.csv"
DAILY_PATH = DATA_DIR / "df_daily.csv"


def main() -> None:
    raw_df = load_raw_data(str(RAW_PATH))
    processed_df = preprocess_raw_data(raw_df)
    processed_df.to_csv(PROCESSED_HOURLY_PATH, index=False)

    df_daily = aggregate_to_daily(processed_df)
    df_daily.to_csv(DAILY_PATH, index=False)

    print("Saved:")
    print(f"- {PROCESSED_HOURLY_PATH}")
    print(f"- {DAILY_PATH}")
    print("Daily shape:", df_daily.shape)


if __name__ == "__main__":
    main()
