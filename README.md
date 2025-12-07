Follow me on [X (Twitter)](https://x.com/x_tomoaki_x)

# 🚀 Quant Meta Hybrid Trader v3 (GPU Optimized)

**⚠️ 研究・実験専用 / For Research & Experimentation Only ⚠️**

---

## 📖 日本語説明

### 概要

**Quant Meta Hybrid Trader v3** は、最先端のディープラーニング技術とGPU最適化を駆使した超高性能な量的トレーディングフレームワークです。🎯

複数の予測モデル（LSTM・Transformer・CNN）を組み合わせ、強化学習（PPO）でトレーディング戦略を自動最適化します。GPU上での混合精度学習（FP16）や並列メタサーチにより、研究用途として極限まで性能を引き上げています。

### ✨ 主な特徴

#### 🔥 GPU最適化の極み
- **混合精度学習（FP16）** - 学習速度＆メモリ効率を大幅UP
- **グラデーション累積** - 大バッチサイズ相当の効果
- **データセットGPUプリロード** - データ転送オーバーヘッド削減
- **CosineAnnealing + Warmup** - 学習率スケジューリング

#### 🧠 マルチモデルアンサンブル
- **LSTM** - 時系列の長期依存関係をキャッチ（Residual接続付き）
- **Transformer** - 注意機構で複雑なパターンを学習
- **Regime CNN** - 市場レジーム（トレンド/レンジ/ボラティリティ）を分類
- **Fusion Network** - 全モデルの予測を統合

#### 🎮 強化学習（PPO）
- **PPO (Proximal Policy Optimization)** - 安定した方策学習
- **GAE (Generalized Advantage Estimation)** - 分散削減
- **並列メタサーチ** - 複数ハイパーパラメータを同時探索

#### 📊 リアルな取引コストモデル
- スプレッド・スリッページ・取引手数料を考慮
- 損失ペナルティ係数によるリスク管理
- トレンド検出時のリワードブースト

### 🛠️ 技術スタック

- **Python 3.8+**
- **PyTorch 2.0+** (CUDA対応)
- **NumPy / Pandas** - データ処理
- **Matplotlib** - 可視化
- **yfinance** - FXデータ取得

### 📦 インストール

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib yfinance
```

### 🚀 使い方

#### 1. データ取得
```bash
python fx_ohlcv_history.py
```
USD/JPYの5分足データを `yf_USDJPYX_5m_max.csv` として保存します。

#### 2. トレーダー学習＆実行
```bash
python quant_meta_hybrid_trader_v3_gpu_optimized.py
```

#### 3. 出力
- コンソールに学習ログ＆シミュレーション結果が表示されます
- `simulation_equity_curve_v3.png` に資産曲線グラフが保存されます

### ⚙️ 設定カスタマイズ

`Config` クラスでパラメータを調整できます：

```python
@dataclass
class Config:
    # データ設定
    PAIR_CSV_LIST: List[str] = ["yf_USDJPYX_5m_max.csv"]
    
    # モデル設定
    LSTM_HIDDEN: int = 128
    LSTM_LAYERS: int = 3
    TF_D_MODEL: int = 128
    TF_NHEAD: int = 8
    
    # 強化学習
    EPISODES_PER_PAIR: int = 30
    GAMMA: float = 0.99
    
    # メタサーチ
    META_TRIALS: int = 15
    USE_FP16: bool = True
```

### 📈 パフォーマンス最適化

v3では以下の最適化を実装：

1. ✅ 混合精度学習（FP16）
2. ✅ グラデーション累積
3. ✅ 並列メタサーチ
4. ✅ より深いネットワーク＋Residual接続
5. ✅ 学習率スケジューラ
6. ✅ データセットGPUプリロード
7. ✅ Residual LSTM Block
8. ✅ BatchNorm / Dropout

### ⚠️ 重要な注意事項

**🔴 このコードは研究・実験専用です。実運用は絶対に禁止！**

- バックテストと実運用は全く別物です
- 過学習・データスヌーピングのリスクがあります
- 市場マイクロストラクチャーの完全なモデル化は困難です
- 実際の取引には金融ライセンスが必要です

### 🔗 リンク

- **X (Twitter)**: [Follow me on X](https://x.com/x_tomoaki_x)
- **GitHub**: [View on GitHub](#)
- **PyTorch Docs**: [pytorch.org](https://pytorch.org/)

---

## 📖 English Description

### Overview

**Quant Meta Hybrid Trader v3** is an ultra-high-performance quantitative trading framework powered by state-of-the-art deep learning and GPU optimization. 🎯

It combines multiple forecasting models (LSTM, Transformer, CNN) with reinforcement learning (PPO) to automatically optimize trading strategies. GPU-accelerated mixed precision training (FP16) and parallel meta-search push performance to the limit for research purposes.

### ✨ Key Features

#### 🔥 Extreme GPU Optimization
- **Mixed Precision Training (FP16)** - Faster training & better memory efficiency
- **Gradient Accumulation** - Effective large batch size training
- **GPU Dataset Preloading** - Reduced data transfer overhead
- **CosineAnnealing + Warmup** - Advanced learning rate scheduling

#### 🧠 Multi-Model Ensemble
- **LSTM** - Captures long-term dependencies (with Residual connections)
- **Transformer** - Learns complex patterns via attention mechanism
- **Regime CNN** - Classifies market regimes (trend/range/volatility)
- **Fusion Network** - Integrates predictions from all models

#### 🎮 Reinforcement Learning (PPO)
- **PPO (Proximal Policy Optimization)** - Stable policy learning
- **GAE (Generalized Advantage Estimation)** - Variance reduction
- **Parallel Meta-Search** - Simultaneous hyperparameter exploration

#### 📊 Realistic Trading Cost Model
- Accounts for spread, slippage, and transaction fees
- Loss penalty factor for risk management
- Reward boost during detected trends

### 🛠️ Tech Stack

- **Python 3.8+**
- **PyTorch 2.0+** (CUDA-enabled)
- **NumPy / Pandas** - Data processing
- **Matplotlib** - Visualization
- **yfinance** - FX data fetching

### 📦 Installation

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib yfinance
```

### 🚀 Usage

#### 1. Fetch Data
```bash
python fx_ohlcv_history.py
```
Downloads USD/JPY 5-minute data and saves it as `yf_USDJPYX_5m_max.csv`.

#### 2. Train & Run Trader
```bash
python quant_meta_hybrid_trader_v3_gpu_optimized.py
```

#### 3. Output
- Training logs & simulation results displayed in console
- Equity curve graph saved as `simulation_equity_curve_v3.png`

### ⚙️ Configuration

Customize parameters via the `Config` class:

```python
@dataclass
class Config:
    # Data settings
    PAIR_CSV_LIST: List[str] = ["yf_USDJPYX_5m_max.csv"]
    
    # Model settings
    LSTM_HIDDEN: int = 128
    LSTM_LAYERS: int = 3
    TF_D_MODEL: int = 128
    TF_NHEAD: int = 8
    
    # Reinforcement learning
    EPISODES_PER_PAIR: int = 30
    GAMMA: float = 0.99
    
    # Meta-search
    META_TRIALS: int = 15
    USE_FP16: bool = True
```

### 📈 Performance Optimizations

v3 implements the following optimizations:

1. ✅ Mixed precision training (FP16)
2. ✅ Gradient accumulation
3. ✅ Parallel meta-search
4. ✅ Deeper networks + Residual connections
5. ✅ Learning rate scheduling
6. ✅ GPU dataset preloading
7. ✅ Residual LSTM blocks
8. ✅ BatchNorm / Dropout

### ⚠️ Important Warnings

**🔴 This code is for RESEARCH & EXPERIMENTATION ONLY. DO NOT use in live trading!**

- Backtesting ≠ Live trading
- Risk of overfitting and data snooping
- Market microstructure is difficult to model completely
- Live trading requires financial licenses

### 🔗 Links

- **X (Twitter)**: [Follow me on X](https://x.com/)
- **GitHub**: [View on GitHub](#)
- **PyTorch Docs**: [pytorch.org](https://pytorch.org/)

---

## 📄 License

This project is for **educational and research purposes only**. Not licensed for commercial use or live trading.

## 🙏 Acknowledgments

Built with ❤️ using PyTorch, NumPy, and the power of GPU acceleration.

**Happy Researching! 🚀📊🤖**
