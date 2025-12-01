# bf_bot_template.py
import os
import time
from dotenv import load_dotenv
import ccxt
import pandas as pd

PAIR = "BTC/JPY"
TIMEFRAME = "5m"
LIMIT = 200

SHORT_MA = 10
LONG_MA = 30

def fetch_ohlcv_df(exchange):
    ohlcv = exchange.fetch_ohlcv(PAIR, timeframe=TIMEFRAME, limit=LIMIT)
    df = pd.DataFrame(
        ohlcv,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("datetime", inplace=True)
    return df

def get_signal(df):
    df["ma_short"] = df["close"].rolling(SHORT_MA).mean()
    df["ma_long"] = df["close"].rolling(LONG_MA).mean()
    latest = df.dropna().iloc[-1]

    if latest["ma_short"] > latest["ma_long"]:
        return "BUY"
    elif latest["ma_short"] < latest["ma_long"]:
        return "SELL"
    else:
        return "HOLD"

def main():
    load_dotenv()

    exchange = ccxt.bitflyer({
        "apiKey": os.getenv("API_KEY"),
        "secret": os.getenv("API_SECRET"),
    })

    # ドライラン用の仮ポジション
    position = None  # "LONG" or "NONE"

    while True:
        try:
            df = fetch_ohlcv_df(exchange)
            signal = get_signal(df)
            last_price = df["close"].iloc[-1]

            print(f"[{df.index[-1]}] price={last_price}, signal={signal}, position={position}")

            # ===== 実際のロジック例（今はprintだけ） =====
            if signal == "BUY" and position is None:
                print("→ BUYシグナルだけど、今はドライランなので発注しないよ")
                # 実トレードならこんな感じ:
                # order = exchange.create_market_buy_order(PAIR, 0.0001)
                # print("BUY ORDER:", order)
                position = "LONG"

            elif signal == "SELL" and position == "LONG":
                print("→ SELLシグナル（利益確定 or 損切り）※ドライラン")
                # 実トレードなら:
                # order = exchange.create_market_sell_order(PAIR, 0.0001)
                # print("SELL ORDER:", order)
                position = None

            # ====================================

        except Exception as e:
            print("Error:", e)

        # 5分足に合わせて適当にsleep（厳密にやるなら次の足開始時刻計算する）
        time.sleep(60)

if __name__ == "__main__":
    main()
