from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor

from src.utils import (
    TARGET_COL,
    TIME_COL,
    add_lag_features,
    add_shifted_rolling_features,
    evaluate_holdout,
    select_correlated_features,
    split_dataframe_chronologically,
    time_series_cv_summary,
)


BASE_DIR = Path(__file__).resolve().parents[1]
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
    df = pd.read_csv(FEATURE_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    lag_df = build_lag_dataset(df)
    lag_df.to_csv(LAG_PATH, index=False)

    selected = select_correlated_features(
        lag_df.drop(columns=[TIME_COL]), threshold=0.3
    )
    model_df = lag_df[[TIME_COL, TARGET_COL] + selected].dropna().copy()

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
            estimator=DecisionTreeRegressor(max_depth=8, random_state=42),
            n_estimators=200,
            random_state=42,
        ),
    }

    holdout_rows = []
    cv_rows = []

    X_cv = model_df.drop(columns=[TIME_COL, TARGET_COL])
    y_cv = model_df[TARGET_COL]

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

        cv = time_series_cv_summary(model, X_cv, y_cv, n_splits=5)
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
