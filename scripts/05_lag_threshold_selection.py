from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostRegressor
from sklearn.tree import DecisionTreeRegressor

from src.utils import TARGET_COL, TIME_COL, add_lag_features, time_series_cv_summary


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
FEATURE_PATH = DATA_DIR / "df_daily_feature_creation.csv"
RESULT_PATH = DATA_DIR / "lag_threshold_selection_summary.csv"


def build_60_day_lag_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(TIME_COL)
    out = add_lag_features(
        out,
        target_col=TARGET_COL,
        lag_days=range(1, 61),
        prefix="electricity_demand_lag",
    )
    out = out.dropna().reset_index(drop=True)
    return out


def selected_columns_by_threshold(df: pd.DataFrame, threshold: float) -> list[str]:
    corr = df.drop(columns=[TIME_COL]).corr(numeric_only=True)[TARGET_COL].dropna()
    cols = corr[abs(corr) > threshold].index.tolist()
    if TARGET_COL in cols:
        cols.remove(TARGET_COL)
    return cols


def main() -> None:
    df = pd.read_csv(FEATURE_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    lag_df = build_60_day_lag_dataset(df)

    model = AdaBoostRegressor(
        estimator=DecisionTreeRegressor(max_depth=10, random_state=42),
        n_estimators=200,
        random_state=42,
    )

    rows = []
    thresholds = [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.8]
    for threshold in thresholds:
        selected = selected_columns_by_threshold(lag_df, threshold)
        if not selected:
            continue
        X = lag_df[selected]
        y = lag_df[TARGET_COL]
        summary = time_series_cv_summary(model, X, y, n_splits=5)
        rows.append(
            {
                "threshold": threshold,
                "n_features": len(selected),
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
