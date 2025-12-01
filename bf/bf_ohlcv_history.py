# bf_ohlcv_history.py
import os
import time
from datetime import datetime
from dotenv import load_dotenv
import ccxt
import pandas as pd


# ======== 設定 =========
SYMBOL = "BTC/JPY"
TIMEFRAME = "1min"          # pandas resampleの単位
LIMIT = 500                 # 1回の fetch_trades で取る件数（bitFlyerの上限に近い）
MAX_TRADES = 50000          # 欲張り上限（ここを増やすともっと遡れる）
SLEEP_SEC = 0.3             # レートリミット配慮用のスリープ


def main():
    load_dotenv()

    exchange = ccxt.bitflyer({
        "apiKey": os.getenv("API_KEY"),
        "secret": os.getenv("API_SECRET"),
        "enableRateLimit": True,
    })

    all_trades = []
    since = None

    print(f"Fetching trades for {SYMBOL} ...")

    while True:
        try:
            # ccxtのfetch_trades: sinceはミリ秒
            trades = exchange.fetch_trades(SYMBOL, since=since, limit=LIMIT)
        except Exception as e:
            print("Error while fetching trades:", e)
            print("Waiting a bit and retrying...")
            time.sleep(5)
            continue

        if not trades:
            print("No more trades returned. Stopping.")
            break

        all_trades.extend(trades)

        # 進捗表示
        latest_dt = datetime.utcfromtimestamp(trades[-1]["timestamp"] / 1000)
        print(
            f"Got {len(trades)} trades "
            f"(total: {len(all_trades)}), "
            f"last ts: {latest_dt} (UTC)"
        )

        # 次のループ用にsinceを更新（最後のtimestampより1ms後）
        since = trades[-1]["timestamp"] + 1

        # 上限に達したら終了
        if len(all_trades) >= MAX_TRADES:
            print(f"Reached MAX_TRADES={MAX_TRADES}. Stopping fetch.")
            break

        time.sleep(SLEEP_SEC)

    if not all_trades:
        print("No trades fetched at all.")
        return

    # ======== DataFrame化 =========
    print("Building DataFrame...")
    df = pd.DataFrame(all_trades)

    # timestamp → datetime
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)

    # ======== OHLCV生成（1分足） =========
    print("Resampling to OHLCV 1min...")

    ohlc = df["price"].resample(TIMEFRAME).ohlc()
    vol = df["amount"].resample(TIMEFRAME).sum()
    ohlc["volume"] = vol

    # 欠損行を削除
    ohlc = ohlc.dropna()

    print("Tail of OHLCV:")
    print(ohlc.tail())

    # ======== CSV保存 =========
    out_file = "bitflyer_1m_ohlcv_history.csv"
    ohlc.to_csv(out_file)
    print(f"saved: {out_file}")
    print(f"total OHLCV rows: {len(ohlc)}")


if __name__ == "__main__":
    main()
