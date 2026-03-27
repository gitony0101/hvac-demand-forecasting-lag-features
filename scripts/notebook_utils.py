from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


def _feature_stem(target: str) -> str:
    return target.replace("_values", "")


def mape(y_true, y_pred) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    non_zero_mask = y_true != 0
    y_true = y_true[non_zero_mask]
    y_pred = y_pred[non_zero_mask]
    if len(y_true) == 0:
        return np.nan
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def evaluate_regression(y_true, y_pred) -> tuple[float, float, float, float]:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = np.mean(np.abs(y_true - y_pred))
    mape_score = mape(y_true, y_pred)
    r2 = np.nan if len(y_true) < 2 else r2_score(y_true, y_pred)
    return (
        float(round(rmse, 4)),
        float(round(mae, 4)),
        float(round(mape_score, 2)),
        float(round(r2, 4)),
    )


def split_by_time(
    df: pd.DataFrame, target: str, start: str, end: str, split_time: str
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    window = df.sort_index().loc[start:end]
    train = window.loc[:split_time]
    test = window.loc[split_time:]
    if not test.empty:
        test = test.iloc[1:]
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_test = test.drop(columns=[target])
    y_test = test[target]
    return X_train, y_train, X_test, y_test


def split_dataframe_chronologically(
    df: pd.DataFrame, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_index()
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    split_idx = int(len(df) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(df):
        raise ValueError("test_size creates an empty train or test split.")
    return df.iloc[:split_idx].copy(), df.iloc[split_idx:].copy()


def chronological_holdout_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X = X.sort_index()
    y = y.loc[X.index]
    split_idx = int(len(X) * (1 - test_size))
    if split_idx <= 0 or split_idx >= len(X):
        raise ValueError("test_size creates an empty train or test split.")
    return (
        X.iloc[:split_idx].copy(),
        X.iloc[split_idx:].copy(),
        y.iloc[:split_idx].copy(),
        y.iloc[split_idx:].copy(),
    )


def add_lag_features(
    df: pd.DataFrame, target: str, lags, prefix: str | None = None
) -> pd.DataFrame:
    result = df.copy()
    prefix = prefix or f"{_feature_stem(target)}_lag"
    for lag in lags:
        result[f"{prefix}_{lag}"] = result[target].shift(lag)
    return result


def add_shifted_rolling_features(
    df: pd.DataFrame, target_columns, window_sizes, shift: int = 1
) -> pd.DataFrame:
    result = df.copy()
    for target in target_columns:
        stem = _feature_stem(target)
        for window_size in window_sizes:
            rolled = result[target].rolling(window=window_size)
            result[f"{stem}_{window_size}d_mean"] = rolled.mean().shift(shift)
            result[f"{stem}_{window_size}d_std"] = rolled.std().shift(shift)
    return result


def select_correlated_features(
    df: pd.DataFrame, target: str, threshold: float
) -> tuple[pd.Series, list[str]]:
    correlations = df.corr(numeric_only=True)[target].sort_values(ascending=False)
    features = correlations[correlations.abs() > threshold].index.tolist()
    if target not in features:
        features.append(target)
    return correlations, features


def time_series_cv_evaluation(
    model, X: pd.DataFrame, y: pd.Series, n_splits: int = 5
) -> tuple[pd.DataFrame, dict[str, float]]:
    splitter = TimeSeriesSplit(n_splits=n_splits)
    fold_rows = []
    for fold_number, (train_idx, val_idx) in enumerate(splitter.split(X), start=1):
        estimator = clone(model)
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        estimator.fit(X_train, y_train)
        y_pred = estimator.predict(X_val)
        rmse, mae, mape_score, r2 = evaluate_regression(y_val, y_pred)
        fold_rows.append(
            {
                "Fold": fold_number,
                "RMSE": rmse,
                "MAE": mae,
                "MAPE(%)": mape_score,
                "R2": r2,
            }
        )

    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "AVG RMSE": float(round(fold_df["RMSE"].mean(), 4)),
        "AVG MAE": float(round(fold_df["MAE"].mean(), 4)),
        "AVG MAPE (%)": float(round(fold_df["MAPE(%)"].mean(), 4)),
        "AVG R2": float(round(fold_df["R2"].mean(), 4)),
    }
    return fold_df, summary


def correlation_threshold_time_series_train(
    df: pd.DataFrame,
    target: str,
    threshold: float,
    model_name: str,
    regressor,
    n_splits: int = 5,
    test_size: float = 0.2,
) -> dict[str, float]:
    train_df, _ = split_dataframe_chronologically(df, test_size=test_size)
    _, correlated_features = select_correlated_features(train_df, target, threshold)
    if len(correlated_features) <= 1:
        raise ValueError(
            f"Threshold {threshold} keeps no predictive features for target '{target}'."
        )
    X_train = train_df[correlated_features].drop(columns=[target])
    y_train = train_df[target]
    _, summary = time_series_cv_evaluation(regressor, X_train, y_train, n_splits=n_splits)
    return {
        "Model name": model_name,
        "corr_threshold": threshold,
        "Fold": n_splits,
        **summary,
    }
