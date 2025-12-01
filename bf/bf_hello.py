import os

from dotenv import load_dotenv
import ccxt


def main():
    # .env 読み込み
    load_dotenv()

    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")

    # 取引所インスタンス作成
    exchange = ccxt.bitflyer({
        "apiKey": api_key,
        "secret": api_secret,
    })

    # 1. 公開情報：BTC/JPYの現在価格（ticker）
    ticker = exchange.fetch_ticker("BTC/JPY")
    print("=== TICKER BTC/JPY ===")
    print("last :", ticker["last"])
    print("bid  :", ticker["bid"])
    print("ask  :", ticker["ask"])

    # 2. 認証必要：アカウント残高（読み取り専用キーなら安全）
    balance = exchange.fetch_balance()
    jpy = balance.get("JPY", {})
    btc = balance.get("BTC", {})

    print("\n=== BALANCE ===")
    print("JPY:", jpy)
    print("BTC:", btc)


if __name__ == "__main__":
    main()
