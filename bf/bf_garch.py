# bf_garch.py
import os
from dotenv import load_dotenv
import ccxt
import pandas as pd
import numpy as np
from arch import arch_model

def main():
    load_dotenv()

    exchange = ccxt.bitflyer({
        "apiKey": os.getenv("API_KEY"),
        "secret": os.getenv("API_SECRET"),
    })

    ohlcv = exchange.fetch_ohlcv("BTC/JPY", timeframe="1h", limit=1000)
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)

    df["log_ret"] = np.log(df["close"]).diff() * 100  # %表現にすることが多い
    returns = df["log_ret"].dropna()

    print("サンプル数:", len(returns))

    # GARCH(1,1) モデル
    am = arch_model(returns, vol="Garch", p=1, q=1, dist="normal")
    res = am.fit(update_freq=10, disp="off")
    print(res.summary())

    # 10ステップ先までのボラ予測
    forecast = res.forecast(horizon=10)
    print("\nVar forecast (first few):")
    print(forecast.variance.tail())

if __name__ == "__main__":
    main()
