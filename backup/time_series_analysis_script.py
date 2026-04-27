import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# Load data
file_path = "/mnt/df/Load_df_01.csv"
df = pd.read_csv(file_path)
df["Time"] = pd.to_datetime(df["Time"])
df.set_index("Time", inplace=True)

# Plot time series
plt.figure(figsize=(15, 6))
plt.plot(df["electricity_demand_values"])
plt.title("Electricity demand time series")
plt.xlabel("Time")
plt.ylabel("Electricity demand")
plt.show()

# Plot ACF and PACF
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
plot_acf(df["electricity_demand_values"], ax=axes[0], lags=50)
axes[0].set_title("Autocorrelation Function (ACF)")
plot_pacf(df["electricity_demand_values"], ax=axes[1], lags=50)
axes[1].set_title("Partial Autocorrelation Function (PACF)")
plt.show()

# Fit ARIMA(1,0,1) model
model = ARIMA(df["electricity_demand_values"], order=(1, 0, 1))
model_fit = model.fit()
print(model_fit.summary())

# Model evaluation
train_size = int(len(df) * 0.8)
train, test = df[0:train_size], df[train_size:]
forecast = model_fit.predict(start=test.index[0], end=test.index[-1])
forecast.index = test.index  # Ensure alignment of predictions and test indices
mse = mean_squared_error(test["electricity_demand_values"], forecast)
rmse = sqrt(mse)
r2 = r2_score(test["electricity_demand_values"], forecast)
print(f"MSE: {mse}, RMSE: {rmse}, R²: {r2}")

# Seasonal periods to try
from statsmodels.tsa.statespace.sarimax import SARIMAX
seasonal_periods = [24, 7, 12]

# Record model performance for each seasonal period
model_performance = []

for period in seasonal_periods:
    # Fit SARIMA model
    sarima_model = SARIMAX(
        train["electricity_demand_values"],
        order=(1, 0, 1),
        seasonal_order=(1, 1, 1, period),
    )
    sarima_model_fit = sarima_model.fit(disp=False)

    # Predict on test set
    sarima_forecast = sarima_model_fit.predict(start=test.index[0], end=test.index[-1])
    sarima_forecast.index = test.index  # Ensure alignment

    # Compute MSE and R²
    mse = mean_squared_error(test["electricity_demand_values"], sarima_forecast)
    r2 = r2_score(test["electricity_demand_values"], sarima_forecast)

    # Record results
    model_performance.append((period, mse, r2))

model_performance_df = pd.DataFrame(
    model_performance, columns=["Seasonal Period", "MSE", "R2"]
)
model_performance_df

