import pybitflyer

def get_btc_jpy_price_pybitflyer():
    api = pybitflyer.API()  # Public APIなのでキーなしでOK[web:8][web:13]
    ticker = api.ticker(product_code="BTC_JPY")  # ティッカー取得[web:8]

    ltp = ticker["ltp"]
    best_bid = ticker["best_bid"]
    best_ask = ticker["best_ask"]

    print("BTC/JPY LTP:", ltp)
    print("best_bid   :", best_bid)
    print("best_ask   :", best_ask)

if __name__ == "__main__":
    get_btc_jpy_price_pybitflyer()
