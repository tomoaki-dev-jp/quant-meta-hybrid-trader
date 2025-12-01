# bf_ohlcv.py
import os
from dotenv import load_dotenv
import ccxt
import pandas as pd

def main():
    load_dotenv()

    exchange = ccxt.bitflyer({
        "apiKey": os.getenv("API_KEY"),
        "secret": os.getenv("API_SECRET"),
    })

    # 直近500件の trade データを取得
    trades = exchange.fetch_trades("BTC/JPY", limit=500)

    df = pd.DataFrame(trades)

    # timestamp → datetime に変換
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)

    # ローソク足に変換（例：1分足）
    ohlcv = df["price"].resample("1min").ohlc()
    vol = df["amount"].resample("1min").sum()

    ohlcv["volume"] = vol

    print(ohlcv.tail())

    # 保存 or 可視化
    ohlcv.to_csv("bitflyer_1m_ohlcv.csv")
    print("saved: bitflyer_1m_ohlcv.csv")

if __name__ == "__main__":
    main()
