# bf_returns_vol.py
import os
from dotenv import load_dotenv
import ccxt
import pandas as pd
import numpy as np

def main():
    load_dotenv()

    exchange = ccxt.bitflyer({
        "apiKey": os.getenv("API_KEY"),
        "secret": os.getenv("API_SECRET"),
    })

    ohlcv = exchange.fetch_ohlcv("BTC/JPY", timeframe="1h", limit=500)
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)

    # ログリターン
    df["log_ret"] = np.log(df["close"]).diff()
    returns = df["log_ret"].dropna()

    # 日次ボラに換算（1h足なので 24 の平方根をかける）
    hourly_vol = returns.std()
    daily_vol = hourly_vol * np.sqrt(24)
    annual_vol = daily_vol * np.sqrt(365)

    print("サンプル数 :", len(returns))
    print("時間足     : 1h")
    print("日次ボラ   :", daily_vol)
    print("年率ボラ   :", annual_vol)

if __name__ == "__main__":
    main()
