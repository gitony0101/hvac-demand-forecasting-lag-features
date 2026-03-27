import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

# 加载数据
file_path = "/mnt/df/Load_df_01.csv"
df = pd.read_csv(file_path)
df["Time"] = pd.to_datetime(df["Time"])
df.set_index("Time", inplace=True)

# 绘制时间序列图
plt.figure(figsize=(15, 6))
plt.plot(df["electricity_demand_values"])
plt.title("电力需求值时间序列图")
plt.xlabel("时间")
plt.ylabel("电力需求值")
plt.show()

# 绘制ACF和PACF图
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
plot_acf(df["electricity_demand_values"], ax=axes[0], lags=50)
axes[0].set_title("自相关函数（ACF）")
plot_pacf(df["electricity_demand_values"], ax=axes[1], lags=50)
axes[1].set_title("偏自相关函数（PACF）")
plt.show()

# 拟合ARIMA(1,0,1)模型
model = ARIMA(df["electricity_demand_values"], order=(1, 0, 1))
model_fit = model.fit()
print(model_fit.summary())

# 模型评估
train_size = int(len(df) * 0.8)
train, test = df[0:train_size], df[train_size:]
forecast = model_fit.predict(start=test.index[0], end=test.index[-1])
forecast.index = test.index  # 确保预测值和测试集的索引对齐
mse = mean_squared_error(test["electricity_demand_values"], forecast)
rmse = sqrt(mse)
r2 = r2_score(test["electricity_demand_values"], forecast)
print(f"MSE: {mse}, RMSE: {rmse}, R²: {r2}")

# 尝试的季节性周期长度 seasonal_periods = [24, 7, 12]
from statsmodels.tsa.statespace.sarimax import SARIMAX

# 尝试的季节性周期长度
seasonal_periods = [24, 7, 12]

# 记录每种周期长度的模型性能
model_performance = []

for period in seasonal_periods:
    # 拟合SARIMA模型
    sarima_model = SARIMAX(
        train["electricity_demand_values"],
        order=(1, 0, 1),
        seasonal_order=(1, 1, 1, period),
    )
    sarima_model_fit = sarima_model.fit(disp=False)

    # 在测试集上进行预测
    sarima_forecast = sarima_model_fit.predict(start=test.index[0], end=test.index[-1])
    sarima_forecast.index = test.index  # 确保预测值和测试集的索引对齐

    # 计算MSE和R²
    mse = mean_squared_error(test["electricity_demand_values"], sarima_forecast)
    r2 = r2_score(test["electricity_demand_values"], sarima_forecast)

    # 将结果记录下来
    model_performance.append((period, mse, r2))

model_performance_df = pd.dfFrame(
    model_performance, columns=["Seasonal Period", "MSE", "R2"]
)
model_performance_df
