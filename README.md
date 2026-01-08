# Stock-Predictor-Using-ML

## Introduction
In this project I built a stock prediction pipeline that uses classic machine learning to estimate **NVIDIA (NVDA)’s next-day return** (percentage change). Instead of predicting the raw closing price directly, the model predicts tomorrow’s return using price-based features, then converts that return into a next-day price estimate for interpretability. This project demonstrates an end-to-end workflow: data collection, feature engineering, model training, evaluation on unseen data, and a final next-day prediction.

## Data Fetching and Preprocessing
Historical NVDA price data is downloaded using Yahoo Finance. The model uses **Adjusted Close** prices to account for stock splits and dividend adjustments. Daily returns are calculated from the adjusted close series, and rolling-window indicators are created. Any rows that contain missing values caused by shifts or rolling calculations are removed before model training.

## Feature Engineering
To transform the time series into a supervised learning dataset, the project creates features designed to capture short-term trend, momentum, and volatility:
- **Daily return (`ret1`)**
- **Lagged returns (`ret_lag_1` to `ret_lag_5`)** to represent recent momentum
- **Moving averages (5-day and 20-day)** to represent short vs. medium trend
- **Momentum vs. 20-day average (`mom_20`)** to measure deviation from baseline
- **Rolling volatility (`vol_20`)** to represent recent risk/variability
- **RSI (14-day)** to capture overbought/oversold conditions

The **target** is defined as the **next-day return**, which makes the prediction task more realistic than fitting a trending price series directly

## Model Building
I trained two regression models:
- **Ridge Regression:** a strong linear baseline with regularization to reduce overfitting
- **Random Forest Regressor:** a nonlinear ensemble model that can capture more complex relationships between indicators and next-day returns

I used a time-based split is used (train on the first 80% of the dataset, test on the most recent 20%) to mimic real forecasting where the model is evaluated on future data that it hasn't seen

## Baselines and Evaluation
To validate whether the ML models outperform simple guesses, I used two baselines:
- **Zero-return baseline:** predicts tomorrow’s return is always 0
- **Naive baseline:** predicts tomorrow’s return equals today’s return

The project reports:
- **MAE (Mean Absolute Error)** on predicted returns
- **R²** on predicted returns
- **Direction accuracy** (how often the model predicts up vs. down correctly)

## Next-Day Prediction
After training, the Random Forest model is used to estimate the next-day return using the most recent feature row. This predicted return is converted into a next-day price estimate:
- `predicted_price = latest_price * (1 + predicted_return)`
