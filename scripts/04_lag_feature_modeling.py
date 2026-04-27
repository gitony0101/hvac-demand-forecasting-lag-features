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
    # ---------------------
    # Configuration for robustness evaluation
    # ---------------------
    SEEDS = [42, 123, 2023]
    SPLIT_TIMES = ["2018-06-01", "2018-07-01", "2018-08-01"]

    if not FEATURE_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {FEATURE_PATH}")

    df = pd.read_csv(FEATURE_PATH)
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    # Build lag features once (common for all runs)
    lag_df = build_lag_dataset(df)
    lag_df.to_csv(LAG_PATH, index=False)

    # Prepare containers to aggregate results across seeds and split times
    holdout_records = []
    cv_records = []

    for split_time in SPLIT_TIMES:
        # Split data according to current split_time
        model_df = lag_df.sort_values(TIME_COL).dropna().copy()
        train_df = model_df[model_df[TIME_COL] <= pd.Timestamp(split_time)].copy()
        test_df = model_df[model_df[TIME_COL] > pd.Timestamp(split_time)].copy()

        # Feature selection based on training split (no leakage)
        selected = select_correlated_features(
            train_df.drop(columns=[TIME_COL]), threshold=0.3
        )
        if not selected:
            raise ValueError("No features selected from training split at threshold=0.3.")

        X_test = test_df[selected]
        y_test = test_df[TARGET_COL]
        test_times = test_df[TIME_COL]

        for seed in SEEDS:
            # Define models with current random seed
            models = {
                "decision_tree": DecisionTreeRegressor(max_depth=8, random_state=seed),
                "random_forest": RandomForestRegressor(
                    n_estimators=300, max_depth=10, random_state=seed, n_jobs=-1
                ),
                "adaboost": AdaBoostRegressor(
                    estimator=DecisionTreeRegressor(max_depth=8, random_state=seed),
                    n_estimators=200,
                    random_state=seed,
                ),
            }

            # Train‑holdout evaluation
            for name, model in models.items():
                X_train = train_df[selected]
                y_train = train_df[TARGET_COL]
                holdout = evaluate_holdout(model, X_train, y_train, X_test, y_test, test_times)
                holdout_records.append(
                    {
                        "split_time": split_time,
                        "seed": seed,
                        "model": name,
                        "rmse": holdout.rmse,
                        "mae": holdout.mae,
                        "mape": holdout.mape,
                        "r2": holdout.r2,
                    }
                )
                # Save individual predictions (overwrites are fine – they are deterministic per run)
                holdout.predictions.to_csv(
                    DATA_DIR / f"lag_model_predictions_{name}_seed{seed}_split{split_time}.csv",
                    index=False,
                )

                # Time‑series CV on the full dataset using same seed‑specific model
                cv = time_series_cv_summary_with_train_only_feature_selection(
                    model,
                    model_df,
                    target_col=TARGET_COL,
                    threshold=0.3,
                    n_splits=5,
                    time_col=TIME_COL,
                )
                cv_records.append(
                    {
                        "split_time": split_time,
                        "seed": seed,
                        "model": name,
                        "avg_rmse": cv.avg_rmse,
                        "avg_mae": cv.avg_mae,
                        "avg_mape": cv.avg_mape,
                        "avg_r2": cv.avg_r2,
                    }
                )
                cv.detailed_results.to_csv(
                    DATA_DIR / f"lag_model_cv_details_{name}_seed{seed}_split{split_time}.csv",
                    index=False,
                )

    # -------------------------------------------------
    # Aggregate statistics across seeds and split times
    # -------------------------------------------------
    holdout_df = pd.DataFrame(holdout_records)
    cv_df = pd.DataFrame(cv_records)

    # Summary: mean and std of each metric per model
    holdout_summary = (
        holdout_df.groupby("model")
        .agg(
            rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"), mae_std=("mae", "std"),
            mape_mean=("mape", "mean"), mape_std=("mape", "std"),
            r2_mean=("r2", "mean"), r2_std=("r2", "std"),
        )
        .reset_index()
    )
    cv_summary = (
        cv_df.groupby("model")
        .agg(
            rmse_mean=("avg_rmse", "mean"), rmse_std=("avg_rmse", "std"),
            mae_mean=("avg_mae", "mean"), mae_std=("avg_mae", "std"),
            mape_mean=("avg_mape", "mean"), mape_std=("avg_mape", "std"),
            r2_mean=("avg_r2", "mean"), r2_std=("avg_r2", "std"),
        )
        .reset_index()
    )

    # Write aggregated summaries
    holdout_summary.to_csv(DATA_DIR / "lag_model_holdout_summary_stats.csv", index=False)
    cv_summary.to_csv(DATA_DIR / "lag_model_cv_summary_stats.csv", index=False)

    print("Saved lag feature datasets, individual predictions, and aggregated statistical summaries.")

if __name__ == "__main__":
    main()
