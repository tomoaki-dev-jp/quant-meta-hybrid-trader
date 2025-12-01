# yf_ohlcv_history.py
import yfinance as yf
import pandas as pd

# ======== 設定 ========

# BTC/JPY なら bitFlyerじゃなくBTC/JPYペアを提供してる取引所次第だけど、
# とりあえず FX なら USD/JPY が鉄板
SYMBOL = "USDJPY=X"       # yfinance のティッカー（USD/JPY）
INTERVAL = "1m"           # "1m", "5m", "15m", "1h", "1d" など
PERIOD = "max"            # "1d", "5d", "1mo", "3mo", "1y", "max" など

OUT_CSV = f"yf_{SYMBOL.replace('=','')}_{INTERVAL}_{PERIOD}.csv"


def main():
    print(f"Downloading {SYMBOL}, interval={INTERVAL}, period={PERIOD} ...")

    df = yf.download(
        tickers=SYMBOL,
        interval=INTERVAL,
        period=PERIOD,
        auto_adjust=False,     # Trueにすると配当とか調整入る
        progress=False,
    )

    if df.empty:
        print("No data returned from yfinance.")
        return

    # インデックスは DatetimeIndex のはず
    df = df.rename(
        columns={
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Adj Close": "adj_close",
            "Volume": "volume",
        }
    )

    # adj_close いらんなら消してもOK
    # df = df[["open", "high", "low", "close", "volume"]]

    print(df.head())
    print(df.tail())
    print("rows:", len(df))

    df.to_csv(OUT_CSV)
    print("saved:", OUT_CSV)


if __name__ == "__main__":
    main()
