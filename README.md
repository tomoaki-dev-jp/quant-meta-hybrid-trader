## yf_ohlcv_history.py とは？

このスクリプトは、Yahoo! Finance から **USD/JPY の5分足ヒストリカルデータを全部ダウンロードして、CSVに保存するツール** だよ。  
通貨ペアや時間足をちょっと書き換えるだけで、他の銘柄にも使い回せるようになってる。

---

## 主な設定パラメータ

- `SYMBOL`  
  - Yahoo! Finance でのティッカー名。  
  - 例:  
    - USD/JPY: `"USDJPY=X"`  
    - EUR/USD: `"EURUSD=X"`  
- `INTERVAL`  
  - ローソクの時間足。`"1m"`, `"5m"`, `"15m"`, `"1h"`, `"1d"` などに変更可能。  
- `PERIOD`  
  - どの期間までさかのぼるか。`"1d"`, `"5d"`, `"1mo"`, `"1y"`, `"max"` など。  
- `OUT_CSV`  
  - 保存されるCSVファイル名。デフォルトだと  
    `yf_USDJPYX_5m_max.csv`  
    みたいな名前になる。

---

## メイン処理の流れ

df = yf.download(
tickers=SYMBOL,
interval=INTERVAL,
period=PERIOD,
auto_adjust=False,
progress=False,
)

1. `yf.download` で指定したティッカー・時間足・期間のOHLCVを取得。  
2. データが空ならメッセージを出して終了。  

---

## カラム名の整形

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


- yfinance が返す列名（`Open`, `High`, ...）を、  
  Python や機械学習コードで扱いやすいように  
  **全部小文字スネークケース** に統一している。  
- `adj_close` が不要ならコメントにある通り削ってもOK。

---

## CSVへの保存

print(df.head())
print(df.tail())
print("rows:", len(df))

df.to_csv(OUT_CSV)
print("saved:", OUT_CSV)


- 先頭・末尾数行と総行数を表示して、ちゃんと取れているか軽くチェック。  
- 最後に `OUT_CSV` で指定したファイル名で **そのままCSV保存**。  
- 生成されたCSVは、さっきの `quant_meta_hybrid_trader_v2.py` の  
  `PAIR_CSV_LIST` にそのまま突っ込んで使える想定。

---

## 使い方まとめ

1. Python環境で `yfinance` と `pandas` をインストールしておく。  
2. 必要なら `SYMBOL`, `INTERVAL`, `PERIOD` を好みのものに変更。  
3. スクリプトを実行すると、カレントディレクトリに  
   `yf_〇〇〇_△△_□□.csv` が出力される。  
4. そのCSVを学習・バックテスト用のデータとして使う。

こんな感じで README に載せれば、「このスクリプトでデータ作ってから、全部乗せトレーダーに食わせる」という流れがわかりやすくなると思う！
