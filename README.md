# hvac-demand-forecasting-lag-features

## Experiment Overview
This project completes the HVAC electricity demand forecasting experiment through the following steps:
1. **Data preprocessing**: Load the original CSV, fill missing values, standardize features, and aggregate to daily data (`data/df_daily.csv`).
2. **Feature engineering**: Add calendar features, binning features, and rolling statistical features to the daily data, generating `data/df_daily_feature_creation.csv`.
3. **Lag feature construction**: From the daily features, create rolling mean, standard deviation, and specified lag‑day features, saving as `data/df_daily_feature_lags.csv`.
4. **Robustness evaluation experiment**: For the three models `DecisionTreeRegressor`, `RandomForestRegressor`, and `AdaBoostRegressor`, perform **hold‑out** evaluation and **time‑series cross‑validation** (train‑only feature selection) under three different time splits (`2018-06-01`, `2018-07-01`, `2018-08-01`) and three random seeds (42, 123, 2023). Each run records prediction results and metrics.
   - Prediction files: `data/lag_model_predictions_<model>_seed<seed>_split<split>.csv`
   - Detailed cross‑validation results: `data/lag_model_cv_details_<model>_seed<seed>_split<split>.csv`
   - Summary statistics files: `data/lag_model_holdout_summary_stats.csv`, `data/lag_model_cv_summary_stats.csv`

## Experimental Results
### Hold‑out Statistics (mean ± std)
| Model | RMSE (mean) | RMSE std | MAE (mean) | MAE std | MAPE (mean %) | MAPE std | R² (mean) | R² std |
|------|-------------|----------|------------|--------|----------------|---------|-----------|--------|
| AdaBoost | 898.22 | 127.03 | 602.50 | 52.16 | 6.50 | 0.19 | 0.9013 | 0.0204 |
| DecisionTree | 1553.26 | 373.50 | 873.77 | 146.88 | 8.98 | 1.29 | 0.7152 | 0.0666 |
| RandomForest | 1080.84 | 164.11 | 652.27 | 91.04 | 6.60 | 0.43 | 0.8586 | 0.0225 |

### Time‑Series Cross‑Validation Statistics (mean ± std)
| Model | RMSE (mean) | RMSE std | MAE (mean) | MAE std | MAPE (mean %) | MAPE std | R² (mean) | R² std |
|------|-------------|----------|------------|--------|----------------|---------|-----------|--------|
| AdaBoost | 857.00 | 3.97 | 516.75 | 3.79 | 5.73 | 0.03 | 0.9292 | 0.0007 |
| DecisionTree | 1474.29 | 13.13 | 750.50 | 5.96 | 7.69 | 0.06 | 0.7911 | 0.0037 |
| RandomForest | 1033.29 | 2.10 | 606.87 | 1.39 | 6.14 | 0.01 | 0.8984 | 0.0004 |

## Conclusions and Recommendations
- **AdaBoost** consistently shows the most robust performance across both evaluation methods, with RMSE ranging between 857–898, R² above 0.90, and extremely low metric variance, indicating insensitivity to random seeds and time splits.
- **RandomForest** performs next best; its overall error is slightly higher but remains within acceptable bounds (RMSE ≈ 1033‑1081, R² ≈ 0.86‑0.90).
- **DecisionTree** exhibits markedly larger errors and higher variance; it is not recommended for production use.
- The **robustness experiment** confirms reproducibility of model performance across different time splits and random seeds, especially highlighting AdaBoost’s strong stability.

## Risks and Future Work
- Current evaluation only covers the first half of 2018. Validation on **unseen periods** (e.g., 2019‑2020) is required to mitigate potential seasonal or trend drift risks.
- Explore **different feature‑selection thresholds**, **longer lag windows** (e.g., 30‑day, 60‑day), and **model ensembling** (Stacking) to potentially improve performance.
- Consider integrating **model deployment** with a **real‑time feature computation pipeline** to assess online latency and resource consumption.

---
*All experimental files are stored under the `data/` directory. Refer to the generated CSV files for detailed per‑run metrics.*
