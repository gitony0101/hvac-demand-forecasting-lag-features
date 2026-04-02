from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit


TARGET_COL = "electricity_demand_values"
TIME_COL = "Time"


CLOUD_COVER_MAP = {
    "no clouds": 0.0,
    "2/10–3/10.": 25.0,
    "4/10.": 40.0,
    "5/10.": 50.0,
    "7/10 – 8/10.": 75.0,
    "10/10.": 100.0,
    "Sky obscured by fog and/or other meteorological phenomena.": 100.0,
}


RAW_RENAME_MAP = {
    "air_pressure[mmHg]": "air_pressure",
    "air_temperature[degree celcius]": "air_temperature",
    "relative_humidity[%]": "relative_humidity",
    "wind_speed[M/S]": "wind_speed",
    "solar_irridiation[W/m²]": "solar_irridiation",
    "total_cloud_cover[from ten]": "total_cloud_cover",
    "electricity_demand_values[kw]": "electricity_demand_values",
    "heat_demand_values[kw]": "heat_demand_values",
}


@dataclass
class HoldoutResult:
    rmse: float
    mae: float
    mape: float
    r2: float
    predictions: pd.DataFrame


@dataclass
class CVSummary:
    avg_rmse: float
    avg_mae: float
    avg_mape: float
    avg_r2: float
    detailed_results: pd.DataFrame


def mape(y_true: Sequence[float], y_pred: Sequence[float]) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    values = np.abs((y_true - y_pred) / denom)
    return float(np.nanmean(values) * 100.0)


def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if TIME_COL not in df.columns:
        raise ValueError(f"Expected column '{TIME_COL}' in raw dataset.")

    df = df.rename(columns=RAW_RENAME_MAP).copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])
    df = df.sort_values(TIME_COL)
    return df


def preprocess_raw_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "total_cloud_cover" in df.columns:
        df["total_cloud_cover_percent"] = df["total_cloud_cover"].map(CLOUD_COVER_MAP)

    numeric_fill_cols = [
        "air_pressure",
        "air_temperature",
        "relative_humidity",
        "wind_speed",
        "total_cloud_cover_percent",
    ]
    for col in numeric_fill_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(df[col].median())

    for col in ["electricity_demand_values", "heat_demand_values"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].ffill()

    if "solar_irridiation" in df.columns:
        df["solar_irridiation_positive"] = df["solar_irridiation"].where(
            df["solar_irridiation"] > 0, 0
        )

    return df


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy().set_index(TIME_COL)

    # NOTE: build the daily index from the target resample directly.
    # Using temp.resample("D").mean() can fail when object columns are present.
    daily_index = temp[TARGET_COL].resample("D").sum().index
    daily = pd.DataFrame(index=daily_index)
    daily[TARGET_COL] = temp[TARGET_COL].resample("D").sum()

    if "heat_demand_values" in temp.columns:
        daily["heat_demand_values"] = temp["heat_demand_values"].resample("D").sum()

    mean_cols = [
        "air_pressure",
        "air_temperature",
        "relative_humidity",
        "wind_speed",
        "solar_irridiation_positive",
        "total_cloud_cover_percent",
    ]
    for col in mean_cols:
        if col in temp.columns:
            daily[col] = temp[col].resample("D").mean()

    if "wind_speed" in temp.columns:
        daily["wind_speed_range"] = (
            temp["wind_speed"].resample("D").max()
            - temp["wind_speed"].resample("D").min()
        )

    if "air_temperature" in temp.columns:
        daily["air_temperature_range"] = (
            temp["air_temperature"].resample("D").max()
            - temp["air_temperature"].resample("D").min()
        )

    daily = daily.reset_index().rename(columns={"index": TIME_COL})
    return daily


def _label_wind(value: float) -> str:
    return "Wind scale 2" if value < 5 else "Wind scale 3"


def _label_humidity(value: float) -> str:
    if value < 40:
        return "Humidity low"
    if value < 70:
        return "Humidity medium"
    return "Humidity high"


def _label_temperature(value: float) -> str:
    if value < 0:
        return "Temp below 0"
    if value < 10:
        return "Temp 0-10"
    if value < 20:
        return "Temp 10-20"
    return "Temp 20+"


def add_calendar_and_bucket_features(df_daily: pd.DataFrame) -> pd.DataFrame:
    df = df_daily.copy()
    df[TIME_COL] = pd.to_datetime(df[TIME_COL])

    if "wind_speed" in df.columns:
        df["wind_scale_label"] = df["wind_speed"].apply(_label_wind)
    if "relative_humidity" in df.columns:
        df["humidity_label"] = df["relative_humidity"].apply(_label_humidity)
    if "air_temperature" in df.columns:
        df["temperature_label"] = df["air_temperature"].apply(_label_temperature)

    df["weekday"] = df[TIME_COL].dt.day_name()
    df["month"] = df[TIME_COL].dt.month.astype(str)

    categorical_cols = [
        col
        for col in [
            "wind_scale_label",
            "humidity_label",
            "temperature_label",
            "weekday",
            "month",
        ]
        if col in df.columns
    ]
    if categorical_cols:
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=False)

    return df


def add_shifted_rolling_features(
    df: pd.DataFrame,
    target_columns: Iterable[str],
    window_sizes: Iterable[int],
    shift: int = 1,
) -> pd.DataFrame:
    out = df.copy()
    for target_col in target_columns:
        if target_col not in out.columns:
            continue
        for window in window_sizes:
            shifted = out[target_col].shift(shift)
            out[f"{target_col}_rolling_mean_{window}"] = shifted.rolling(window).mean()
            out[f"{target_col}_rolling_std_{window}"] = shifted.rolling(window).std()
    return out


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lag_days: Iterable[int],
    prefix: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    base = prefix or target_col
    for lag in lag_days:
        out[f"{base}_{lag}"] = out[target_col].shift(lag)
    return out


def select_correlated_features(
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    threshold: float = 0.3,
) -> list[str]:
    corr = df.corr(numeric_only=True)[target_col].dropna()
    selected = corr[abs(corr) > threshold].index.tolist()
    if target_col in selected:
        selected.remove(target_col)
    return selected


def split_dataframe_chronologically(
    df: pd.DataFrame,
    split_time: str,
    target_col: str = TARGET_COL,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = df.copy()
    data[TIME_COL] = pd.to_datetime(data[TIME_COL])
    data = data.sort_values(TIME_COL)

    train = data[data[TIME_COL] <= pd.Timestamp(split_time)].copy()
    test = data[data[TIME_COL] > pd.Timestamp(split_time)].copy()

    feature_cols = [c for c in data.columns if c not in [TIME_COL, target_col]]
    X_train = train[feature_cols]
    y_train = train[target_col]
    X_test = test[feature_cols]
    y_test = test[target_col]
    return X_train, X_test, y_train, y_test


def evaluate_holdout(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_times: Sequence[pd.Timestamp] | pd.Series,
) -> HoldoutResult:
    estimator = clone(model)
    estimator.fit(X_train, y_train)
    pred = estimator.predict(X_test)

    result_df = pd.DataFrame(
        {
            TIME_COL: pd.to_datetime(test_times),
            "y_true": y_test.to_numpy(),
            "y_pred": pred,
        }
    )
    return HoldoutResult(
        rmse=float(np.sqrt(mean_squared_error(y_test, pred))),
        mae=float(mean_absolute_error(y_test, pred)),
        mape=float(mape(y_test, pred)),
        r2=float(r2_score(y_test, pred)),
        predictions=result_df,
    )


def time_series_cv_summary(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
) -> CVSummary:
    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        estimator = clone(model)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        rows.append(
            {
                "fold": fold,
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "mae": float(mean_absolute_error(y_test, pred)),
                "mape": float(mape(y_test, pred)),
                "r2": float(r2_score(y_test, pred)),
            }
        )

    details = pd.DataFrame(rows)
    return CVSummary(
        avg_rmse=float(details["rmse"].mean()),
        avg_mae=float(details["mae"].mean()),
        avg_mape=float(details["mape"].mean()),
        avg_r2=float(details["r2"].mean()),
        detailed_results=details,
    )


def time_series_cv_summary_with_train_only_feature_selection(
    model,
    df: pd.DataFrame,
    target_col: str = TARGET_COL,
    threshold: float = 0.3,
    n_splits: int = 5,
    time_col: str = TIME_COL,
) -> CVSummary:
    """
    Time-series CV with correlation feature selection fit on each training fold only.
    This prevents validation-fold leakage during feature selection.
    """
    data = df.copy()
    if time_col in data.columns:
        data[time_col] = pd.to_datetime(data[time_col])
        data = data.sort_values(time_col).reset_index(drop=True)

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rows = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(data), start=1):
        train_df = data.iloc[train_idx].copy()
        test_df = data.iloc[test_idx].copy()

        selected = select_correlated_features(
            train_df.drop(columns=[c for c in [time_col] if c in train_df.columns]),
            target_col=target_col,
            threshold=threshold,
        )
        if not selected:
            continue

        estimator = clone(model)
        X_train = train_df[selected]
        y_train = train_df[target_col]
        X_test = test_df[selected]
        y_test = test_df[target_col]
        estimator.fit(X_train, y_train)
        pred = estimator.predict(X_test)
        rows.append(
            {
                "fold": fold,
                "n_features": len(selected),
                "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
                "mae": float(mean_absolute_error(y_test, pred)),
                "mape": float(mape(y_test, pred)),
                "r2": float(r2_score(y_test, pred)),
            }
        )

    if not rows:
        raise ValueError(
            "No features passed the correlation threshold in any training fold."
        )

    details = pd.DataFrame(rows)
    return CVSummary(
        avg_rmse=float(details["rmse"].mean()),
        avg_mae=float(details["mae"].mean()),
        avg_mape=float(details["mape"].mean()),
        avg_r2=float(details["r2"].mean()),
        detailed_results=details,
    )
