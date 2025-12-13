# ================================================================================
# fx_ohlcv_history.py
#
# FX OHLCV Historical Data Downloader
#
# Downloads FX (Foreign Exchange) price data from Yahoo Finance and saves
# it to CSV format for backtesting and analysis.
# ================================================================================

import yfinance as yf
import pandas as pd

# ======== Configuration ========

# For crypto pairs like BTC/JPY, availability depends on the exchange providing that pair.
# However, for FX, USD/JPY is the most standard and reliable pair.
# 
# You can also use other FX pairs:
# - "EURUSD=X"  (EUR/USD)
# - "GBPUSD=X"  (GBP/USD)
# - "JPYUSD=X"  (JPY/USD - inverse of USD/JPY)
# 
# Note: yfinance uses "=X" suffix for currency pairs.

SYMBOL = "USDJPY=X"  # yfinance ticker symbol (USD/JPY)

# Interval options: "1m", "5m", "15m", "30m", "1h", "1d", "1wk", "1mo"
# Note: 1m and 5m data may have limited history (typically last 7-30 days)
INTERVAL = "5m"

# Period options: "1d", "5d", "1mo", "3mo", "6mo", "1y", "2y", "5y", "10y", "ytd", "max"
# "max" downloads all available historical data
PERIOD = "max"

# Output CSV filename (auto-generated based on parameters)
OUT_CSV = f"yf_{SYMBOL.replace('=','')}_{INTERVAL}_{PERIOD}.csv"


def main():
    """
    Download FX price data from Yahoo Finance and save to CSV.
    
    Process:
    1. Download OHLCV (Open, High, Low, Close, Volume) data from yfinance
    2. Standardize column names to lowercase
    3. Display sample data (first/last rows and total row count)
    4. Save to CSV file for further analysis
    """
    print(f"Downloading {SYMBOL}, interval={INTERVAL}, period={PERIOD} ...")
    
    # Download historical data from Yahoo Finance
    df = yf.download(
        tickers=SYMBOL,
        interval=INTERVAL,
        period=PERIOD,
        auto_adjust=False,  # If True, includes dividend adjustments (usually not needed for FX)
        progress=False,      # Suppress progress bar output
    )
    
    # Check if data was successfully retrieved
    if df.empty:
        print("No data returned from yfinance.")
        return
    
    # Standardize column names to lowercase for consistency
    # yfinance returns: "Open", "High", "Low", "Close", "Adj Close", "Volume"
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
    
    # Optional: If you don't need adjusted close price, uncomment to keep only OHLCV
    # df = df[["open", "high", "low", "close", "volume"]]
    
    # Display first few rows (head)
    print(df.head())
    
    # Display last few rows (tail)
    print(df.tail())
    
    # Display total number of rows
    print("rows:", len(df))
    
    # Save to CSV file
    df.to_csv(OUT_CSV)
    
    print("saved:", OUT_CSV)


if __name__ == "__main__":
    main()
