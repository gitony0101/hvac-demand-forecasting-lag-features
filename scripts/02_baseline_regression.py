from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from src.utils import (
    TARGET_COL,
    TIME_COL,
    evaluate_holdout,
    split_dataframe_chronologically,
)


DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "figures"
DAILY_PATH = DATA_DIR / "df_daily.csv"
SPLIT_TIME = "2018-06-01"


def main() -> None:
    if not DAILY_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {DAILY_PATH}")

    df = pd.read_csv(DAILY_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.dropna().sort_values(TIME_COL)

    feature_cols = [
        c for c in df.columns if c not in [TIME_COL, TARGET_COL, "heat_demand_values"]
    ]
    model_df = df[[TIME_COL, TARGET_COL] + feature_cols].dropna().copy()

    X_train, X_test, y_train, y_test = split_dataframe_chronologically(
        model_df, split_time=SPLIT_TIME, target_col=TARGET_COL
    )
    test_times = model_df.loc[model_df[TIME_COL] > SPLIT_TIME, TIME_COL]

    models = {
        "decision_tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=10, random_state=42, n_jobs=-1
        ),
        "adaboost": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=6, random_state=42),
            n_estimators=200,
            random_state=42,
        ),
    }

    rows = []
    for name, model in models.items():
        result = evaluate_holdout(model, X_train, y_train, X_test, y_test, test_times)
        rows.append(
            {
                "model": name,
                "rmse": result.rmse,
                "mae": result.mae,
                "mape": result.mape,
                "r2": result.r2,
            }
        )
        result.predictions.to_csv(
            DATA_DIR / f"baseline_predictions_{name}.csv", index=False
        )

    summary = pd.DataFrame(rows).sort_values("rmse")
    summary.to_csv(DATA_DIR / "baseline_regression_summary.csv", index=False)
    print(summary)


if __name__ == "__main__":
    main()
