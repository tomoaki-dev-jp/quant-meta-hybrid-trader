import time
from datetime import datetime
import os

import pybitflyer
import pandas as pd

# ======== 設定 =========
PRODUCT_CODE = "BTC_JPY"     # bitFlyerのproduct_code[web:4]
TIMEFRAME = "1min"
LIMIT = 500                  # 1回の取得件数（count）[web:4]
MAX_TRADES = 2000            # テスト用にちょい少なめ
SLEEP_SEC = 0.3


def fetch_all_executions():
    api = pybitflyer.API()   # 公開APIだけならキー不要[web:13]

    all_execs = []
    before = None  # pagination用: idベース[web:4]

    print(f"Fetching executions for {PRODUCT_CODE} ...")

    while True:
        params = {
            "product_code": PRODUCT_CODE,
            "count": LIMIT,
        }
        if before is not None:
            params["before"] = before  # このidより「前」を取る[web:4]

        try:
            execs = api.executions(**params)  # GET /v1/executions[web:13][web:4]
        except Exception as e:
            print("Error while fetching executions:", e)
            print("Waiting a bit and retrying...")
            time.sleep(5)
            continue

        if not execs:
            print("No more executions returned. Stopping.")
            break

        all_execs.extend(execs)

        latest = execs[-1]
        # exec_date は ISO8601 文字列[web:4]
        latest_dt = datetime.fromisoformat(
            latest["exec_date"].replace("Z", "+00:00")
        )
        print(
            f"Got {len(execs)} executions "
            f"(total: {len(all_execs)}), "
            f"last ts: {latest_dt} (UTC)"
        )

        # 次のループ用にbeforeを更新（最後のidより小さいものを取る）
        before = latest["id"]

        if len(all_execs) >= MAX_TRADES:
            print(f"Reached MAX_TRADES={MAX_TRADES}. Stopping fetch.")
            break

        time.sleep(SLEEP_SEC)

    return all_execs


def build_ohlcv(execs):
    print("Building DataFrame...")
    df = pd.DataFrame(execs)
    print("raw columns:", df.columns)

    # exec_date を datetime に変換（フォーマット混在対応）[web:42]
    df["datetime"] = pd.to_datetime(df["exec_date"], format="mixed", utc=True)
    df.set_index("datetime", inplace=True)

    print("index min/max:", df.index.min(), df.index.max())

    print(f"Resampling to OHLCV {TIMEFRAME}...")
    ohlc = df["price"].resample(TIMEFRAME).ohlc()
    vol = df["size"].resample(TIMEFRAME).sum()
    ohlc["volume"] = vol

    print("before dropna len:", len(ohlc))
    ohlc = ohlc.dropna()
    print("after dropna len:", len(ohlc))

    return ohlc


def main():
    print("=== main start ===")

    execs = fetch_all_executions()
    print("len(execs) =", len(execs))

    if not execs:
        print("No executions fetched at all. RETURN here.")
        return

    ohlc = build_ohlcv(execs)
    print("after build_ohlcv, len(ohlc) =", len(ohlc))

    print("Tail of OHLCV:")
    print(ohlc.tail())

    out_file = "bitflyer_1m_ohlcv_history_pybitflyer.csv"
    fullpath = os.path.abspath(out_file)
    print("will save to:", fullpath)

    ohlc.to_csv(out_file)
    print(f"saved: {out_file}")
    print(f"total OHLCV rows: {len(ohlc)}")


if __name__ == "__main__":
    print("=== if __name__ == '__main__' ===")
    main()
