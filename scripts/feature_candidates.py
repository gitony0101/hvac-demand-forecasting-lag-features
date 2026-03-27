from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_datetime_index(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("DataFrame index must be a pandas DatetimeIndex.")


def clean_infinite_values(df: pd.DataFrame, dropna: bool = False) -> pd.DataFrame:
    result = df.replace([np.inf, -np.inf], np.nan)
    if dropna:
        result = result.dropna()
    return result


def add_heat_feature_candidates(
    df_daily: pd.DataFrame,
    hourly_df: pd.DataFrame,
    heat_column: str = "heat_demand_values",
) -> pd.DataFrame:
    _validate_datetime_index(df_daily)
    _validate_datetime_index(hourly_df)
    if heat_column not in hourly_df.columns:
        raise KeyError(f"'{heat_column}' not found in hourly_df.")

    result = df_daily.copy()
    daily_heat = hourly_df[heat_column].resample("D")

    if heat_column not in result.columns:
        result[heat_column] = daily_heat.sum()
    result["heat_demand_mean"] = daily_heat.mean()
    result["heat_demand_change"] = daily_heat.max() - daily_heat.min()
    result["heat_demand_growth_rate"] = result[heat_column].ffill().pct_change().fillna(0)
    return clean_infinite_values(result)


def add_growth_rate_features(
    df_daily: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    suffix: str = "_growth_rate",
) -> pd.DataFrame:
    _validate_datetime_index(df_daily)
    result = df_daily.copy()
    missing = [column for column in columns if column not in result.columns]
    if missing:
        raise KeyError(f"Columns not found in df_daily: {missing}")

    for column in columns:
        result[f"{column}{suffix}"] = result[column].ffill().pct_change().fillna(0)
    return clean_infinite_values(result)


def add_resampled_growth_rate_features(
    df_daily: pd.DataFrame,
    hourly_df: pd.DataFrame,
    columns: list[str] | tuple[str, ...],
    freq: str = "D",
    suffix: str = "_growth_rate",
) -> pd.DataFrame:
    _validate_datetime_index(df_daily)
    _validate_datetime_index(hourly_df)
    result = df_daily.copy()

    for column in columns:
        if column in hourly_df.columns:
            resampled = hourly_df[column].resample(freq).mean()
        elif column in result.columns:
            resampled = result[column]
        else:
            raise KeyError(
                f"'{column}' not found in hourly_df or df_daily for growth-rate construction."
            )
        result[f"{column}{suffix}"] = resampled.ffill().pct_change().fillna(0)
    return clean_infinite_values(result)


def add_calendar_dummies(
    df: pd.DataFrame,
    include_weekday: bool = True,
    include_month: bool = True,
    include_quarter: bool = False,
) -> pd.DataFrame:
    _validate_datetime_index(df)
    result = df.copy()
    dummy_specs: list[tuple[str, str]] = []

    if include_weekday:
        result["Day_of_Week"] = result.index.day_of_week
        dummy_specs.append(("Day_of_Week", "Is_Weekday"))
    if include_month:
        result["month"] = result.index.month
        dummy_specs.append(("month", "month"))
    if include_quarter:
        result["quarter"] = result.index.quarter
        dummy_specs.append(("quarter", "Quarter"))

    if not dummy_specs:
        return result

    result = pd.get_dummies(
        result,
        columns=[column for column, _ in dummy_specs],
        prefix=[prefix for _, prefix in dummy_specs],
    )

    dummy_columns = [
        column
        for column in result.columns
        if column.startswith("Is_Weekday_")
        or column.startswith("month_")
        or column.startswith("Quarter_")
    ]
    for column in dummy_columns:
        result[column] = result[column].astype(int)
    return result
