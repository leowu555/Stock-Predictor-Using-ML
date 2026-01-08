import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


def rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / (avg_loss + 1e-12)
    return 100 - (100 / (1 + rs))


def make_features(df):
    if "Adj Close" not in df.columns:
        raise ValueError("Missing 'Adj Close' column. Try auto_adjust=False or check your download.")

    price = df["Adj Close"].copy()
    ret = price.pct_change()

    feat = pd.DataFrame(index=df.index)
    feat["ret1"] = ret

    for k in range(1, 6):
        feat[f"ret_lag_{k}"] = ret.shift(k)

    feat["ma_5"] = price.rolling(5).mean()
    feat["ma_20"] = price.rolling(20).mean()
    feat["mom_20"] = (price / feat["ma_20"]) - 1
    feat["vol_20"] = ret.rolling(20).std()
    feat["rsi_14"] = rsi(price, 14)

    feat["target_next_ret"] = ret.shift(-1)
    feat["price"] = price

    return feat.dropna()


def report(name, y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    dir_acc = (np.sign(y_pred) == np.sign(y_true)).mean()

    print(f"\n{name}")
    print(f"MAE (return): {mae:.6f}")
    print(f"R2  (return): {r2:.6f}")
    print(f"Direction accuracy: {dir_acc * 100:.2f}%")


def main():
    ticker = "NVDA"
    df = yf.download(ticker, period="10y", auto_adjust=False)

    if df.empty:
        print("No data downloaded. Try a different ticker or timeframe.")
        return

    feat = make_features(df)

    feature_cols = [c for c in feat.columns if c not in ["target_next_ret", "price"]]
    X = feat[feature_cols]
    y = feat["target_next_ret"].values

    split = int(len(feat) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    pred_zero = np.zeros_like(y_test)
    pred_naive = X_test["ret1"].values

    report("Baseline: Zero return", y_test, pred_zero)
    report("Baseline: Naive (predict today's return)", y_test, pred_naive)

    ridge = Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge(alpha=1.0))
    ])
    ridge.fit(X_train, y_train)
    pred_ridge = ridge.predict(X_test)
    report("Ridge", y_test, pred_ridge)

    rf = RandomForestRegressor(
        n_estimators=400,
        random_state=42,
        max_depth=8,
        min_samples_leaf=10,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    pred_rf = rf.predict(X_test)
    report("Random Forest", y_test, pred_rf)

    plt.plot(y_test, label="Actual next-day return")
    plt.plot(pred_zero, label="Baseline: Zero")
    plt.plot(pred_naive, label="Baseline: Naive")
    plt.plot(pred_rf, label="Predicted (RF)")
    plt.title(f"{ticker}: Next-day return prediction")
    plt.xlabel("Time")
    plt.ylabel("Return")
    plt.legend()
    plt.show()

    latest_X = X.iloc[[-1]]
    latest_price = float(feat["price"].iloc[-1])

    tomorrow_ret = float(rf.predict(latest_X)[0])
    tomorrow_price = latest_price * (1 + tomorrow_ret)

    print(f"\nLatest price: {latest_price:.2f}")
    print(f"Predicted next-day return (RF): {tomorrow_ret:.6f}")
    print(f"Predicted next-day price (RF): {tomorrow_price:.2f}")


if __name__ == "__main__":
    main()
