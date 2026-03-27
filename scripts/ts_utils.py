from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass

import numpy as np
import pandas as pd

from scripts.notebook_utils import evaluate_regression


@dataclass(frozen=True)
class RollingWindow:
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def _as_series(series: pd.Series) -> pd.Series:
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Series index must be a pandas DatetimeIndex.")
    return series.sort_index()


def _import_statsmodels():
    try:
        from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
        from statsmodels.tsa.arima.model import ARIMA
        from statsmodels.tsa.statespace.sarimax import SARIMAX
        from statsmodels.tsa.stattools import adfuller
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "statsmodels is required for ts_utils diagnostics and statistical forecasting helpers."
        ) from exc
    return plot_acf, plot_pacf, ARIMA, SARIMAX, adfuller


def adf_summary(series: pd.Series, autolag: str = "AIC") -> pd.Series:
    _, _, _, _, adfuller = _import_statsmodels()
    series = _as_series(series).dropna()
    adf_statistic, p_value, used_lag, nobs, critical_values, icbest = adfuller(
        series, autolag=autolag
    )
    return pd.Series(
        {
            "adf_statistic": adf_statistic,
            "p_value": p_value,
            "used_lag": used_lag,
            "nobs": nobs,
            "critical_value_1pct": critical_values["1%"],
            "critical_value_5pct": critical_values["5%"],
            "critical_value_10pct": critical_values["10%"],
            "icbest": icbest,
            "is_stationary_5pct": p_value <= 0.05,
        }
    )


def plot_acf_pacf_diagnostics(
    series: pd.Series,
    lags: int = 40,
    figsize: tuple[int, int] = (12, 8),
    acf_ylim: tuple[float, float] | None = None,
    pacf_ylim: tuple[float, float] | None = None,
):
    import matplotlib.pyplot as plt

    plot_acf, plot_pacf, _, _, _ = _import_statsmodels()
    series = _as_series(series).dropna()
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    plot_acf(series, lags=lags, ax=axes[0], alpha=0.05)
    plot_pacf(series, lags=lags, ax=axes[1], alpha=0.05)
    axes[0].set_title("ACF")
    axes[1].set_title("PACF")
    if acf_ylim is not None:
        axes[0].set_ylim(acf_ylim)
    if pacf_ylim is not None:
        axes[1].set_ylim(pacf_ylim)
    plt.tight_layout()
    return fig, axes


def chronological_series_split(
    series: pd.Series, train_fraction: float = 0.8
) -> tuple[pd.Series, pd.Series]:
    series = _as_series(series)
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1.")
    split_idx = int(len(series) * train_fraction)
    if split_idx <= 0 or split_idx >= len(series):
        raise ValueError("train_fraction creates an empty train or test split.")
    return series.iloc[:split_idx].copy(), series.iloc[split_idx:].copy()


def fit_arima_holdout(
    train: pd.Series,
    test: pd.Series,
    order: tuple[int, int, int] = (1, 0, 1),
) -> dict[str, object]:
    _, _, ARIMA, _, _ = _import_statsmodels()
    train = _as_series(train).dropna()
    test = _as_series(test).dropna()
    model = ARIMA(train, order=order)
    fitted = model.fit()
    forecast = fitted.forecast(steps=len(test))
    forecast.index = test.index
    rmse, mae, mape_score, r2 = evaluate_regression(test, forecast)
    return {
        "model": fitted,
        "forecast": forecast,
        "metrics": {"RMSE": rmse, "MAE": mae, "MAPE(%)": mape_score, "R2": r2},
    }


def fit_sarimax_holdout(
    train: pd.Series,
    test: pd.Series,
    order: tuple[int, int, int] = (1, 0, 1),
    seasonal_order: tuple[int, int, int, int] = (1, 1, 1, 24),
) -> dict[str, object]:
    _, _, _, SARIMAX, _ = _import_statsmodels()
    train = _as_series(train).dropna()
    test = _as_series(test).dropna()
    model = SARIMAX(train, order=order, seasonal_order=seasonal_order)
    fitted = model.fit(disp=False)
    forecast = fitted.forecast(steps=len(test))
    forecast.index = test.index
    rmse, mae, mape_score, r2 = evaluate_regression(test, forecast)
    return {
        "model": fitted,
        "forecast": forecast,
        "metrics": {"RMSE": rmse, "MAE": mae, "MAPE(%)": mape_score, "R2": r2},
    }


def rolling_day_windows(
    start_date: str | pd.Timestamp,
    history_days: int,
    prediction_days: int,
    horizon_days: int = 1,
    step_days: int = 1,
) -> Iterable[RollingWindow]:
    if history_days <= 0 or prediction_days <= 0 or horizon_days <= 0 or step_days <= 0:
        raise ValueError("history_days, prediction_days, horizon_days, and step_days must be positive.")

    current_start = pd.Timestamp(start_date)
    for _ in range(prediction_days):
        train_end = current_start + pd.DateOffset(days=history_days) - pd.Timedelta(hours=1)
        test_start = train_end + pd.Timedelta(hours=1)
        test_end = test_start + pd.DateOffset(days=horizon_days) - pd.Timedelta(hours=1)
        yield RollingWindow(
            train_start=current_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
        )
        current_start += pd.DateOffset(days=step_days)


def walk_forward_evaluate(
    series: pd.Series,
    windows: Iterable[RollingWindow],
    fit_predict: Callable[[pd.Series, int], pd.Series | list[float]],
) -> pd.DataFrame:
    series = _as_series(series)
    rows = []
    for window in windows:
        train = series.loc[window.train_start : window.train_end]
        test = series.loc[window.test_start : window.test_end]
        if train.empty or test.empty:
            continue

        forecast = fit_predict(train, len(test))
        forecast = pd.Series(np.asarray(forecast), index=test.index)
        rmse, mae, mape_score, r2 = evaluate_regression(test, forecast)
        rows.append(
            {
                "train_start": window.train_start,
                "train_end": window.train_end,
                "test_start": window.test_start,
                "test_end": window.test_end,
                "RMSE": rmse,
                "MAE": mae,
                "MAPE(%)": mape_score,
                "R2": r2,
            }
        )
    return pd.DataFrame(rows)
