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
    add_lag_features,
    add_shifted_rolling_features,
    evaluate_holdout,
    select_correlated_features,
    time_series_cv_summary_with_train_only_feature_selection,
)


DATA_DIR = BASE_DIR / "data"
FEATURE_PATH = DATA_DIR / "df_daily_feature_creation.csv"
LAG_PATH = DATA_DIR / "df_daily_feature_lags.csv"
SPLIT_TIME = "2018-06-01"


def build_lag_dataset(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy().sort_values(TIME_COL)
    out = add_shifted_rolling_features(
        out,
        target_columns=["electricity_demand_values", "heat_demand_values"],
        window_sizes=[7, 14],
        shift=1,
    )
    out = add_lag_features(
        out,
        target_col="electricity_demand_values",
        lag_days=[1, 6, 13],
        prefix="electricity_demand_lag",
    )
    out = out.dropna().reset_index(drop=True)
    return out


def main() -> None:
    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {FEATURE_PATH}")

    df = pd.read_csv(FEATURE_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    lag_df = build_lag_dataset(df)
    lag_df.to_csv(LAG_PATH, index=False)

    # Fit feature selection only on historical training split to avoid holdout leakage.
    model_df = lag_df.sort_values(TIME_COL).dropna().copy()
    train_df = model_df[model_df[TIME_COL] <= pd.Timestamp(SPLIT_TIME)].copy()
    test_df = model_df[model_df[TIME_COL] > pd.Timestamp(SPLIT_TIME)].copy()
    selected = select_correlated_features(
        train_df.drop(columns=[TIME_COL]), threshold=0.3
    )
    if not selected:
        raise ValueError("No features selected from training split at threshold=0.3.")

    X_train = train_df[selected]
    y_train = train_df[TARGET_COL]
    X_test = test_df[selected]
    y_test = test_df[TARGET_COL]
    test_times = test_df[TIME_COL]

    models = {
        "decision_tree": DecisionTreeRegressor(max_depth=8, random_state=42),
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=10, random_state=42, n_jobs=-1
        ),
        "adaboost": AdaBoostRegressor(
            estimator=DecisionTreeRegressor(max_depth=8, random_state=42),
            n_estimators=200,
            random_state=42,
        ),
    }

    holdout_rows = []
    cv_rows = []

    for name, model in models.items():
        holdout = evaluate_holdout(model, X_train, y_train, X_test, y_test, test_times)
        holdout_rows.append(
            {
                "model": name,
                "rmse": holdout.rmse,
                "mae": holdout.mae,
                "mape": holdout.mape,
                "r2": holdout.r2,
            }
        )
        holdout.predictions.to_csv(
            DATA_DIR / f"lag_model_predictions_{name}.csv", index=False
        )

        cv = time_series_cv_summary_with_train_only_feature_selection(
            model,
            model_df,
            target_col=TARGET_COL,
            threshold=0.3,
            n_splits=5,
            time_col=TIME_COL,
        )
        cv_rows.append(
            {
                "model": name,
                "avg_rmse": cv.avg_rmse,
                "avg_mae": cv.avg_mae,
                "avg_mape": cv.avg_mape,
                "avg_r2": cv.avg_r2,
            }
        )
        cv.detailed_results.to_csv(
            DATA_DIR / f"lag_model_cv_details_{name}.csv", index=False
        )

    pd.DataFrame(holdout_rows).sort_values("rmse").to_csv(
        DATA_DIR / "lag_model_holdout_summary.csv", index=False
    )
    pd.DataFrame(cv_rows).sort_values("avg_rmse").to_csv(
        DATA_DIR / "lag_model_cv_summary.csv", index=False
    )

    print("Saved lag feature datasets and evaluation summaries.")


if __name__ == "__main__":
    main()
