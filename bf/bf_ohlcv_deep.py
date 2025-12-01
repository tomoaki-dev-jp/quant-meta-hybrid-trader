# bf_ohlcv_deep.py
import os
import time
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv
import ccxt
import pandas as pd


# ======== 設定（ここを好みで調整） =========
SYMBOL = "BTC/JPY"
TIMEFRAME = "1min"          # pandas の resample 単位
LIMIT = 500                 # fetch_trades の1回あたり件数（bitFlyer上限付近）
MAX_TRADES = 200_000        # 取得する約定履歴の上限件数（欲張り度）
SLEEP_SEC = 0.4             # API負荷を抑えるスリープ時間
OUT_CSV = "bitflyer_1m_ohlcv_history.csv"


def fetch_trades_deep(
    exchange: ccxt.Exchange,
    symbol: str,
    since: Optional[int] = None,
    limit: int = LIMIT,
    max_trades: int = MAX_TRADES,
) -> list:
    """
    bitFlyer の fetch_trades を since 付きで再帰的に叩いて
    できるだけ深く約定履歴を集める。
    """
    all_trades = []

    print(f"Start fetch_trades: symbol={symbol}, limit={limit}, max_trades={max_trades}")

    while True:
        try:
            trades = exchange.fetch_trades(symbol, since=since, limit=limit)
        except Exception as e:
            print(f"[WARN] fetch_trades error: {e}")
            print("       wait 5 sec and retry...")
            time.sleep(5)
            continue

        if not trades:
            print("[INFO] No trades returned (reached history limit or gap).")
            break

        all_trades.extend(trades)

        # 進捗表示
        last_ts = trades[-1]["timestamp"]
        last_dt = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
        print(
            f"  +{len(trades)} trades (total={len(all_trades)}), "
            f"last_ts={last_dt.isoformat()}"
        )

        # 次のsince(最後のtimestampの1ms後)を設定
        since = last_ts + 1

        # 上限に達したら終了
        if len(all_trades) >= max_trades:
            print(f"[INFO] Reached MAX_TRADES={max_trades}")
            break

        time.sleep(SLEEP_SEC)

    return all_trades


def main():
    load_dotenv()

    exchange = ccxt.bitflyer({
        "apiKey": os.getenv("API_KEY"),
        "secret": os.getenv("API_SECRET"),
        "enableRateLimit": True,
    })

    # ========= 深めに履歴を取得 =========
    trades = fetch_trades_deep(exchange, SYMBOL)

    if not trades:
        print("[ERROR] No trades fetched. Abort.")
        return

    # ========= DataFrame 化 =========
    print("[INFO] Building DataFrame...")
    df = pd.DataFrame(trades)

    # timestamp → datetime に変換
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)

    # もし重複トレードIDがあれば削っておく（念のため）
    if "id" in df.columns:
        before = len(df)
        df = df[~df["id"].duplicated(keep="first")]
        after = len(df)
        if after != before:
            print(f"[INFO] Removed {before - after} duplicated trades by id")

    # ========= OHLCV 生成（1分足） =========
    print(f"[INFO] Resampling to OHLCV ({TIMEFRAME})...")

    # price → OHLC
    ohlc = df["price"].resample(TIMEFRAME).ohlc()

    # amount → volume
    vol = df["amount"].resample(TIMEFRAME).sum()
    ohlc["volume"] = vol

    # 欠損行（取引がなかった足）を消したいならここで dropna
    ohlc = ohlc.dropna()

    print("[INFO] Tail of OHLCV:")
    print(ohlc.tail())

    # ========= CSV 保存 =========
    ohlc.to_csv(OUT_CSV)
    print(f"[INFO] saved: {OUT_CSV}")
    print(f"[INFO] total OHLCV rows: {len(ohlc)}")


if __name__ == "__main__":
    main()
