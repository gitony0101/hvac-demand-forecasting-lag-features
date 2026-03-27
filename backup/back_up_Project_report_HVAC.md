## $$\text{Residential building with HVAC System Electricity} $$
## $$\text{Demand Analysis and Prediction} $$









## Introduction

Predicting energy demand accurately is vital for the management and optimization of local energy systems. Understanding electricity demand patterns helps identify opportunities for improving energy efficiency within residential buildings.  By optimizing usage during off-peak hours and minimizing wasteful energy practices, residents can save money while still meeting their energy needs.By understanding how much power is needed, grid operators can allocate resources effectively and minimize disruptions to residential areas.

In this research project, sereral regression techniques were employed to forecast the daily electricity demand of a residential building using meteorological data. Data preprocessing and feature engineering methods were applied to enhance the model's performance. The evaluation of these techniques will be carried out as part of the feature selection process, aiming to optimize the predictive capabilities of the model.


## Data Description and Preprocessing

### Data Description


The dataset was from the IEEE dataports[^1]. Consists of 70080 hourly measurements of electricity and heating demands, along with meteorological data from December 2010 to November 2018. The input features include temperature, humidity, wind speed, and time indicators, while the outputs are the demands for electricity and heating.



![](./pic/table.png)

The dataset information is shown as below:

``````
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 70080 entries, 0 to 70079
Data columns (total 9 columns):
 #   Column                           Non-Null Count  Dtype  
---  ------                           --------------  -----  
 0   Time                             70080 non-null  object 
 1   air_pressure[mmHg]               69934 non-null  float64
 2   air_temperature[degree celcius]  69903 non-null  float64
 3   relative_humidity[%]             69903 non-null  float64
 4   wind_speed[M/S]                  69125 non-null  float64
 5   solar_irridiation[W/m²]          70080 non-null  int64  
 6   total_cloud_cover[from ten]      69837 non-null  object 
 7   electricity_demand_values[kw]    70073 non-null  float64
 8   heat_demand_values[kw]           70073 non-null  float64
dtypes: float64(6), int64(1), object(2)
memory usage: 4.8+ MB
```



- air_pressure: Air pressure measurements.
- air_temperature: Temperature readings.
- relative_humidity: Humidity levels.
- wind_speed: Speed of the wind.
- solar_irradiation: Solar radiation measurements.
- total_cloud_cover: Descriptive data on cloud cover.
- electricity_demand_values: The demand values for electricity.
- heat_demand_values: The demand values for heating.
- total_cloud_cover_percent: Cloud cover represented as a percentage.

Except for the attribute 'total_cloud_cover[from ten]', all the other attributes are numeric values.



### Data Preprocessing

Data preprocessing steps are mainly referred to the file 'Data Characteristics and Data Prepration Functions' in  IEEE dataports webpage[^1]

The data preprocessing steps are as follows:

1. Convert "total_cloud_cover" values to percentage values, drop the original "total_cloud_cover" columns.
2. As for "air_pressure", "air_temperature", "relative_humidity", "wind_speed" features, fill missing values in numerical columns with their respective median values.
3. Forward-fill missing values in the "electricity_demand_values" and "heat_demand_values" columns






Additionally, the hourly dataset is aggregated to daily data by taking the mean of each feature,except for the electricity demand values, which is summed up to get the daily demand values.

## Visualization

Transformed into daily timestep resolution, the dataset is visualized as below:

![Daily Electricity Demand 2010-2018](./pic/daily_electricity.png)


![Daily Heat Demand 2010-2018](./pic/daily_heat.png)

![Daily Air Temperature 2010-2018](./pic/daily_temp.png)

![Daily Air Pressure 2010-2018](./pic/daily_pressure.png)

![Daily Relative Humidity 2010-2018](./pic/daily_humidity.png)

![Daily Solar Irradiation 2010-2018](./pic/daily_solar.png)

![Daily Cloud Cover 2010-2018](./pic/daily_cloud.png)

![Daily Wind Speed 2010-2018](./pic/daily_wind.png)

From the daily visualization plots, we can find that there is a clear annual pattern with peaks and troughs corresponding to specific times of the year for the electricity and heat demand. As for the other features, there is no obvious pattern.

![Correlation matrix heatmap](./pic/corr_0.png)

Also,from the correlation matrix, we can find that the electricity demand is weakly correlated with the other features, the regression model could not be used to predict the electricity demand correctly. So we need more features to improve the performance of the model.

## Feature Engineering

### Feature creation

- New features were created, including labels for wind scale, humidity, and temperature ranges.
- Day of the week, and month were transformed into dummy variables to capture seasonal and weekly patterns.
- To avoid multicollinearity, the original features were dropped after creating the new features. Additionally, new created features should not be corelated with each other, such as season and month.

#### Models evaluation

The dataset was split into training and testing sets based on a specified date, ensuring a temporal split for predictive modeling. Then ,Decision Tree Regressor, RandomForestRegressor, and AdaBoostRegressor were trained. Features used for training included transformed environmental factors and time-related features.

Models were evaluated based on metrics such as RMSE (Root Mean Square Error), MAE (Mean Absolute Error),MAPE(Mean Absolute Percentage Error) and $R^2$ Score:





The initial models showed relatively lower performance, indicating a need for additional or more relevant features.


| Model | RMSE | MAE         | MAPE(%)	 | $R^2$      |
|:--:|:--:|:--:|:--:|:--:|
|   Decision Tree Regressor | 3675.4119  | 2479.7466 | 23.8 |0.0039  |
|  Random Forest Regressor  |2913.0264  |  2000.0461 | 19.11  | 0.3743  |
|  AdaBoostRegressor  | 3051.6157   | 2054.8736 | 19.03 |0.3133  |

![](./pic/dt-1.png)

![](./pic/rf-1.png)

![](./pic/ada-1.png)

Based on the results, the models' performance are improve while still not good enough. As the correlation heatmap shows, the electricity demand began to increase correlations with other new features,which created by the orignal ones.

![Correlation matrix heatmap](./pic/corr_1.png)

### Feature Addition: Lag Features



Lagged features are a feature engineering technique used to capture the temporal dependencies and patterns in time series data. A lagged feature is created by taking the value of a variable at a previous time point and including it as a feature in the model at the current time point.[^2]

Lag features are essentially the data points from prior time periods within a time series. They are critical in identifying and leveraging the autocorrelation within the dataset, which is needed for our dataset with low correlation between features and electricity demand.

In this  problem, Lag features were added to capture past patterns in electricity and heat demand.Rolling means and standard deviations of past electricity and heat demands were calculated for different range of previous days.

#### Models evaluation

With the same training and testing sets previously used, the models were trained again. The results are shown as below:

| Model   | RMSE   | MAE |MAPE(%)  | $R^2$   |
|:--:|:--:|:--:|:--:|:--:|
|   Decision Tree Regressor | 1787.2976 | 1101.0262  | 10.12  |0.7644   |
|  Random Forest Regressor | 1372.4725	|881.147	|7.91	|0.8611|
|  AdaBoostRegressor  | 1288.5808|	765.3942|	7.02|0.8776 |


![](./pic/dt-2.png)

![](./pic/rf-2.png)

![](./pic/ada-2.png)

From the results, we can find that the models' performance are improved significantly. The Random Forest Regressor and AdaBoostRegressor models have a better performance than the Decision Tree Regressor model. The AdaBoostRegressor model has the best performance with the lowest RMSE, MAE and the highest $R^2$ score.Lag features helped the model to predict electricity demand and heating demand better on all evaluation metrics.



### Feature Selection

Among these features, we would like to find which features are the most important ones for the model. So we can use these features to train the model and get a better performance.

In order to select these features, we try to remove the group of features, find the performance changes of the model with the same training and testing sets previously used:

- Remove meteorological features.
- Remove time features.
- Remove the rolling mean and standard deviation features.
- Remove the electricity demand lag features, keep the rolling mean and standard deviation features

#### 1. Remove meteorological features.

By removing air pressure, positive solar irradiation, wind speed,humidity,temperature features, the decision tree regressor, random forest regressor, and adaboost regressor were trained again. The results are shown as below:

| Model Name | RMSE          | MAE       | MAPE(%)  | R2   |        
| ---------- | ------------- | --------- | -------- | ---- | 
|  Decision Tree | 1730.3962 | 977.4776 | 8.91 | 0.7792 |
| Random Forest | 1369.6512 | 879.3371 | 7.89 | 0.8617 |
| AdaBoost      | 1176.9446 | 676.8089 | 6.34 | 0.8979 |

From the result we can find the performance of the model had improved slightly. The meteorological features may not contibute to the model, so we can remove these features.

#### 2. Remove time features.

Remove the weekday,month features.

| Model Name    | RMSE      | MAE      | MAPE(%) | R2     |
| ------------- | --------- | -------- | ------- | ------ |
| Decision Tree | 1640.1313 | 908.1021 | 8.49    | 0.8016 |
| Random Forest | 1385.5789 | 884.0704 | 7.90    | 0.8584 |
| AdaBoost      | 1126.1640 | 677.7310 | 6.30    | 0.9065 |

From the result we can find the performance of the results of all models continued to improve slightly. Time features may not be essential for the models. Removing these features can help the model to train faster with lower error rates.

#### 3. Remove the rolling mean and standard deviation features.

Remove the rolling mean and standard deviation features of electricity demand and heat demand.

| Model Name    | RMSE      | MAE      | MAPE(%) | R2     |
| ------------- | --------- | -------- | ------- | ------ |
| Decision Tree | 1811.8924 | 908.6708 | 8.56    | 0.7579 |
| Random Forest | 1270.0618 | 856.8795 | 7.82    | 0.8811 |
| AdaBoost      | 876.4978  | 483.9728 | 4.75    | 0.9433 |

The models showed slight improvement after removing the rolling mean and standard deviation features,suggesting that they may not be essential for our predictive framework. so these features can be removed.

#### 4. Remove the electricity demand lag features, keep the rolling mean and standard deviation features

| Model Name    | RMSE      | MAE       | MAPE(%) | R2     |
| ------------- | --------- | --------- | ------- | ------ |
| Decision Tree | 2137.7112 | 1313.4009 | 11.89   | 0.6630 |
| Random Forest | 1574.1340 | 1031.4958 | 9.22    | 0.8173 |
| AdaBoost      | 1682.3125 | 1029.2823 | 9.02    | 0.7913 |

Performance is worse than the model with lag features for electricity demand. So we can keep electricity demand lag features.







In conclusion, the feature selection result indicates electricity demand lag features' importace to the model's predictive capacity. Consider the electricity demand seasonality, we will examine the model's performance with more electricity demand lag features next.

### Add more electricity demand lag features

In this part, we will add 60 days of electricity demand lag features to the model, select the best number of days as lag features with the correlation with electricity demand.



#### Correlation heatmap of electricity demand lag features

First, we will review the correlation heatmap of electricity demand lag features.




![Correlation coefficients heatmap with
different thresholds](./pic/Correlation%20coefficients%20with%20diffrerent%20thresholds.png)






By reviewing the correlation heatmap of electricity demand lag features with different correlation threshold, we can find that the correlation between electricity demand and lag features is getting higher with the increase of the correlation threshold. In another word, the threshold may be a measure of the importance of the lag features. 







Then the model(adaboost regressor) was trained with the lage features selected by different correlation threshold, the results are shown as below:

| Model name | corr_threshold | Fold | AVG RMSE | AVG MAE   | AVG MAPE (%) | AVG R2  |        
| -------------- | ---- | -------- | --------- | ------------ | ------- | ------ |
|  AdaBoost       | 0.0  | 5        | 400.2068  | 222.1167     | 2.4644  | 0.9844 |
| AdaBoost       | 0.1  | 5        | 425.1105  | 224.5642     | 2.5049  | 0.9822 |
| AdaBoost       | 0.3  | 5        | 463.8254  | 240.7469     | 2.7016  | 0.9791 |
| AdaBoost       | 0.5  | 5        | 543.9827  | 305.2649     | 3.4391  | 0.9714 |
|  AdaBoost       | 0.6  | 5        | 819.8412  | 473.5267     | 5.2226  | 0.9350 |
|  AdaBoost       | 0.7  | 5        | 2192.0806 | 1331.6177    | 13.7826 | 0.5410 |
|  AdaBoost       | 0.8  | 5        | 2192.0806 | 1331.6177    | 13.7826 | 0.5410 |

From the result, we can find the more lag features we add, the better performance the model has. But when the correlation threshold is 0.7 and 0.8, the model's performance is worse than the model with the correlation threshold of 0.6. On the other hand, the model with the correlation threshold of 0.3 could be a good choice, because it has a better performance than the model with the correlation threshold of 0.6 on the RMSE , MAE and MAPE metrics.

Or in another word, we can select 32 days of electricity demand lag features to train the model. Afer training the adaboost regressor with 32 days of electricity demand lag features with cross validation, the result is shown as below:

| AVG RMSE   | AVG MAE    | AVG MAPE(%) | AVG R2   |
| ---------- | ---------- | ----------- | -------- |
| 463.825351 | 240.746929 | 2.701643    | 0.979094 |

These results demonstrate the effectiveness of using 32 days of lag feature configuration for achieving a high level of model performance.


![predicted demand and test demand](./Pic/cv_test_plot.png)



## Conclusion

The electricity demand analysis and prediction project demonstrates the potential of regression techniques in understanding and forecasting energy demand patterns. 
This electricity demand regression model reached 97.9% R-squared score on average in cross validation test without Time Series methods (ARIMA or SARIMA), meanwhile with lower computational cost.
By employing target lag features, the project provides valuable insights that can aid in efficient energy management. By correlation coefficient matrix, we can find the relationships between present electricity demand and the number of previous days’ demand. With high correlated lag features, the model can predict the future demand with high accuracy as mentioned above.






[^1]: [8 YEARS OF HOURLY HEAT AND ELECTRICITY DEMAND FOR A RESIDENTIAL BUILDING](https://ieee-dataport.org/open-access/8-years-hourly-heat-and-electricity-demand-residential-building)

[^2]:[Laged Features](https://www.hopsworks.ai/dictionary/lagged-features#:~:text=Lagged%20features%20are%20a%20feature,at%20the%20current%20time%20point)







