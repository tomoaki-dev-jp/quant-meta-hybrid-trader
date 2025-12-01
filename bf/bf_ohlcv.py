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

    # ====== 最新500件のトレード（約定履歴）取得 ======
    trades = exchange.fetch_trades("BTC/JPY", limit=500)

    # pandas DataFrame に変換
    df = pd.DataFrame(trades)

    # timestamp → datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)

    # ====== OHLCV 1分足を生成 ======
    ohlc = df["price"].resample("1min").ohlc()
    vol = df["amount"].resample("1min").sum()

    # ボリューム列追加
    ohlc["volume"] = vol

    print("===== OHLCV (tail) =====")
    print(ohlc.tail())

    # ====== 保存 ======
    ohlc.to_csv("bitflyer_1m_ohlcv.csv")
    print("saved: bitflyer_1m_ohlcv.csv")

if __name__ == "__main__":
    main()
