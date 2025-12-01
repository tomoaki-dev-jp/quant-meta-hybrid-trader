# bf_garch_vol.py
import os
import numpy as np
import pandas as pd
import ccxt
from dotenv import load_dotenv
from arch import arch_model


PAIR = "BTC/JPY"
TIMEFRAME = "1min"   # pandas の resample 用


def fetch_ohlcv_from_trades(exchange, limit=2000):
    """
    bitFlyer は fetch_ohlcv 非対応なので、
    trades から 1分足OHLCV を自作する。
    """
    trades = exchange.fetch_trades(PAIR, limit=limit)
    df = pd.DataFrame(trades)

    # timestamp → datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)

    # 1分足OHLCV
    ohlc = df["price"].resample(TIMEFRAME).ohlc()
    vol = df["amount"].resample(TIMEFRAME).sum()
    ohlc["volume"] = vol

    # 欠損を落とす
    ohlc = ohlc.dropna()

    return ohlc


def main():
    load_dotenv()

    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")

    exchange = ccxt.bitflyer({
        "apiKey": api_key,
        "secret": api_secret,
    })

    # 1. OHLCV生成
    df = fetch_ohlcv_from_trades(exchange, limit=2000)
    print("OHLCV rows:", len(df))

    # 2. ログリターン（%表現にするのがGARCH界隈の慣習）
    df["log_ret"] = np.log(df["close"]).diff() * 100
    returns = df["log_ret"].dropna()

    print("リターンのサンプル:")
    print(returns.tail())

    # 3. GARCH(1,1) モデルを当てる
    #   r_t = μ + ε_t
    #   ε_t ~ N(0, σ_t^2)
    #   σ_t^2 = ω + α * ε_{t-1}^2 + β * σ_{t-1}^2
    am = arch_model(returns, vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(update_freq=10, disp="off")

    print("\n=== GARCH(1,1) 推定結果 ===")
    print(res.summary())

    # 4. 10ステップ先までのボラ予測
    forecast = res.forecast(horizon=10)
    var_forecast = forecast.variance.iloc[-1]
    print("\n=== 先行ボラ予測（分散） ===")
    print(var_forecast)

    # 標準偏差（ボラ）に変換
    sigma_forecast = np.sqrt(var_forecast)
    print("\n=== 先行ボラ予測（標準偏差, %） ===")
    print(sigma_forecast)


if __name__ == "__main__":
    main()
