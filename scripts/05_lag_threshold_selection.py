from __future__ import annotations

import os
from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils import (
    TARGET_COL,
    TIME_COL,
    add_lag_features,
    time_series_cv_summary_with_train_only_feature_selection,
)


DATA_DIR = BASE_DIR / "data"
FEATURE_PATH = DATA_DIR / "df_daily_feature_creation.csv"
RESULT_PATH = DATA_DIR / "lag_threshold_selection_summary.csv"


def build_lag_dataset(df: pd.DataFrame, max_lag_days: int = 60) -> pd.DataFrame:
    out = df.copy().sort_values(TIME_COL)
    out = add_lag_features(
        out,
        target_col=TARGET_COL,
        lag_days=range(1, max_lag_days + 1),
        prefix="electricity_demand_lag",
    )
    out = out.dropna().reset_index(drop=True)
    return out


def main() -> None:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {FEATURE_PATH}")

    # Defaults preserve current project behavior; env vars allow lighter smoke runs.
    max_lag_days = int(os.getenv("HVAC_MAX_LAG_DAYS", "60"))
    n_estimators = int(os.getenv("HVAC_N_ESTIMATORS", "200"))
    n_splits = int(os.getenv("HVAC_N_SPLITS", "5"))
    thresholds = [
        float(x.strip())
        for x in os.getenv("HVAC_THRESHOLDS", "0.0,0.1,0.3,0.5,0.6,0.7,0.8").split(",")
        if x.strip()
    ]

    df = pd.read_csv(FEATURE_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    lag_df = build_lag_dataset(df, max_lag_days=max_lag_days)

    model = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=10, random_state=42),
        n_estimators=n_estimators,
        random_state=42,
    )

    rows = []
    for threshold in thresholds:
        try:
            summary = time_series_cv_summary_with_train_only_feature_selection(
                model,
                lag_df,
                target_col=TARGET_COL,
                threshold=threshold,
                n_splits=n_splits,
                time_col=TIME_COL,
            )
        except ValueError:
            continue
        rows.append(
            {
                "threshold": threshold,
                "n_features": int(summary.detailed_results["n_features"].mean()),
                "avg_rmse": summary.avg_rmse,
                "avg_mae": summary.avg_mae,
                "avg_mape": summary.avg_mape,
                "avg_r2": summary.avg_r2,
            }
        )

    result_df = pd.DataFrame(rows).sort_values(["avg_rmse", "avg_mae"])
    result_df.to_csv(RESULT_PATH, index=False)
    print(result_df)


if __name__ == "__main__":
    main()
