# bf_volatility.py
import os
import numpy as np
import pandas as pd
import ccxt
from dotenv import load_dotenv


PAIR = "BTC/JPY"
TIMEFRAME = "1min"   # pandas の resample 用表記


def fetch_ohlcv_from_trades(exchange, limit=1000):
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

    # 欠損を落とす（出来高0の足など）
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

    # 1. OHLCV取得（トレードから生成）
    df = fetch_ohlcv_from_trades(exchange, limit=2000)
    print("OHLCV rows:", len(df))
    print(df.tail())

    # 2. ログリターン計算
    #   r_t = log(P_t / P_{t-1})
    df["log_ret"] = np.log(df["close"]).diff()
    returns = df["log_ret"].dropna()

    print("\nリターンサンプル:")
    print(returns.tail())

    # 3. ボラティリティ計算
    # 1分足なので → 1日の分数 = 1440
    per_min_vol = returns.std()
    daily_vol = per_min_vol * np.sqrt(1440)      # 日次ボラ
    annual_vol = daily_vol * np.sqrt(365)        # 年率ボラ

    print("\n=== ボラティリティ推定結果 ===")
    print(f"サンプル数: {len(returns)}")
    print(f"1分足の標準偏差      : {per_min_vol:.6f} （log return）")
    print(f"日次ボラティリティ    : {daily_vol:.4f}  ≒ {daily_vol*100:.2f} %")
    print(f"年率ボラティリティ    : {annual_vol:.4f}  ≒ {annual_vol*100:.2f} %")


if __name__ == "__main__":
    main()
