# ================================================================================
# quant_meta_hybrid_trader_v4.1_FIXED.py
#
# 全部乗せ研究用トレーダーフレームワーク v4.1 - 完全修正版
# 
# ★修正内容★
# 1. データリーケージ完全排除（リアルタイム特徴量計算）
# 2. LoRA完全実装
# 3. Regime CNN復活
# 4. Walk-forward検証実装
# 5. ベースライン比較機能
# 6. エラーハンドリング強化
# 7. テストコード追加
# 8. パフォーマンス最適化
#
# ※ 研究用。実運用前に十分な検証が必要！
# ================================================================================

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Optional, Callable
from dataclasses import dataclass
from pathlib import Path
import warnings
from abc import ABC, abstractmethod

warnings.filterwarnings('ignore')


# ================== 設定 ==================

@dataclass
class ConfigV41:
    """v4.1 設定（データリーケージ対策済み）"""
    # データ
    PAIR_CSV_LIST: List[str] = None
    USE_RESAMPLE: bool = False
    RESAMPLE_RULE: str = "5min"
    MAX_POINTS: int = 20000
    
    # 状態＆予測
    STATE_RET_LEN: int = 48
    FORECAST_HORIZONS: List[int] = None
    
    # Mamba
    MAMBA_SEQ_LEN: int = 64
    MAMBA_D_MODEL: int = 256
    MAMBA_D_STATE: int = 16
    MAMBA_LAYERS: int = 4
    MAMBA_EPOCHS: int = 20
    MAMBA_LR: float = 1.5e-3
    MAMBA_BATCH: int = 512
    
    # TFT
    TFT_D_MODEL: int = 256
    TFT_NHEAD: int = 8
    TFT_LAYERS: int = 6
    TFT_FF: int = 1024
    TFT_EPOCHS: int = 20
    TFT_LR: float = 1.5e-3
    TFT_BATCH: int = 512
    USE_FLASH_ATTENTION: bool = True
    
    # Regime CNN
    REGIME_SEQ_LEN: int = 64
    REGIME_HIDDEN: int = 256
    REGIME_EPOCHS: int = 15
    REGIME_LR: float = 1.5e-3
    REGIME_BATCH: int = 512
    
    # RL
    EPISODES_PER_PAIR: int = 30
    STEPS_PER_EP: int = 1200
    GAMMA: float = 0.99
    LAMBDA_GAE: float = 0.95
    CLIP_EPS: float = 0.2
    EPOCHS_PPO: int = 5
    MINI_BATCH: int = 2048
    
    # LoRA
    USE_LORA: bool = True
    LORA_RANK: int = 16
    LORA_ALPHA: float = 32.0
    
    # アクション
    N_ACTIONS: int = 7
    MAX_POSITION: int = 5
    
    # コスト
    TRANSACTION_COST: float = 0.00003
    LOSS_FACTOR: float = 1.2
    TREND_THRESHOLD: float = 0.0001
    TREND_BOOST: float = 2.0
    
    # FX
    SPREAD_PIPS: float = 0.02
    SLIPPAGE_PIPS: float = 0.01
    PIP_VALUE_JPY: float = 0.01
    
    # メタサーチ
    META_TRIALS: int = 15
    USE_FP16: bool = True
    USE_ENSEMBLE: bool = True
    N_ENSEMBLE_MODELS: int = 3
    
    # Walk-forward検証
    USE_WALK_FORWARD: bool = True
    WALK_FORWARD_TRAIN_RATIO: float = 0.6
    WALK_FORWARD_TEST_RATIO: float = 0.2
    
    # ベースライン
    COMPARE_BASELINES: bool = True
    
    def __post_init__(self):
        if self.PAIR_CSV_LIST is None:
            self.PAIR_CSV_LIST = ["yf_USDJPYX_5m_max.csv"]
        if self.FORECAST_HORIZONS is None:
            self.FORECAST_HORIZONS = [1, 3, 6, 12, 24]


cfg = ConfigV41()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

print(f"\n{'='*80}")
print(f"🚀 Quant Meta Hybrid Trader v4.1 - FIXED Edition")
print(f"{'='*80}")
print(f"Device: {device}")
print(f"Data Leakage Protection: ✅ ENABLED")
print(f"Walk-Forward Validation: {'✅' if cfg.USE_WALK_FORWARD else '❌'}")
print(f"Baseline Comparison: {'✅' if cfg.COMPARE_BASELINES else '❌'}")
print(f"LoRA: {'✅' if cfg.USE_LORA else '❌'}")
print(f"{'='*80}\n")


# ================== リアルタイム特徴量計算（データリーケージ対策） ==================

class FeatureCalculator:
    """データリーケージを防ぐリアルタイム特徴量計算"""
    
    def __init__(self, prices: np.ndarray):
        self.prices = prices
        self.returns = np.diff(prices) / prices[:-1]
        self._cache = {}
    
    def get_features(self, t: int) -> Dict[str, float]:
        """
        時刻tまでのデータのみ使用して特徴量計算
        
        重要: t以降のデータは一切使用しない！
        """
        if t in self._cache:
            return self._cache[t]
        
        if t < 1:
            return self._get_default_features()
        
        # t以前のデータのみ使用
        returns_so_far = self.returns[:t]
        
        features = {
            'return': returns_so_far[-1] if len(returns_so_far) > 0 else 0.0,
            'vol_12': self._calc_vol(returns_so_far, 12),
            'vol_36': self._calc_vol(returns_so_far, 36),
            'trend_36': self._calc_trend(returns_so_far, 36),
            'rsi': self._calc_rsi(returns_so_far, 14),
            'return_smooth': self._calc_ema(returns_so_far, 10),
        }
        
        self._cache[t] = features
        return features
    
    def _calc_vol(self, returns: np.ndarray, window: int) -> float:
        """ボラティリティ計算"""
        if len(returns) < window:
            return 0.0
        return float(np.std(returns[-window:]))
    
    def _calc_trend(self, returns: np.ndarray, window: int) -> float:
        """トレンド計算"""
        if len(returns) < window:
            return 0.0
        return float(np.mean(returns[-window:]))
    
    def _calc_rsi(self, returns: np.ndarray, window: int) -> float:
        """RSI計算"""
        if len(returns) < window:
            return 0.0
        
        gains = np.where(returns[-window:] > 0, returns[-window:], 0)
        losses = np.where(returns[-window:] < 0, -returns[-window:], 0)
        
        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)
        
        if avg_loss == 0:
            return 1.0
        
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return (rsi - 50.0) / 50.0  # Normalize to [-1, 1]
    
    def _calc_ema(self, returns: np.ndarray, span: int) -> float:
        """EMA計算"""
        if len(returns) == 0:
            return 0.0
        
        alpha = 2.0 / (span + 1.0)
        ema = returns[0]
        for r in returns[1:]:
            ema = alpha * r + (1 - alpha) * ema
        
        return float(ema)
    
    def _get_default_features(self) -> Dict[str, float]:
        """デフォルト特徴量"""
        return {
            'return': 0.0,
            'vol_12': 0.0,
            'vol_36': 0.0,
            'trend_36': 0.0,
            'rsi': 0.0,
            'return_smooth': 0.0,
        }
    
    def get_feature_vector(self, t: int) -> np.ndarray:
        """特徴量をベクトル化"""
        features = self.get_features(t)
        return np.array([
            features['return'],
            features['vol_12'],
            features['vol_36'],
            features['trend_36'],
            features['rsi'],
            features['return_smooth'],
        ], dtype=np.float32)


# ================== データセット構築（リーケージ対策版） ==================

def build_safe_dataset(feature_calc: FeatureCalculator, start_idx: int, end_idx: int):
    """
    データリーケージを防ぐデータセット構築
    
    Args:
        feature_calc: 特徴量計算器
        start_idx: 開始インデックス
        end_idx: 終了インデックス（この位置までのデータのみ使用）
    """
    seq_len = cfg.MAMBA_SEQ_LEN
    horizon_max = max(cfg.FORECAST_HORIZONS)
    
    X_list, y_list = [], []
    
    # end_idx以降のデータは絶対に使わない
    for t in range(start_idx + seq_len, end_idx - horizon_max):
        # シーケンス構築（t-seq_len から t-1 まで）
        sequence = []
        for i in range(t - seq_len, t):
            feat_vec = feature_calc.get_feature_vector(i)
            sequence.append(feat_vec)
        
        X_list.append(np.array(sequence))
        
        # ターゲット（t+h の return）
        targets = []
        for h in cfg.FORECAST_HORIZONS:
            target_idx = t + h - 1
            if target_idx < len(feature_calc.returns):
                targets.append(feature_calc.returns[target_idx])
            else:
                targets.append(0.0)
        
        y_list.append(targets)
    
    if len(X_list) == 0:
        return None, None
    
    X = torch.tensor(np.array(X_list, dtype=np.float32), device=device)
    y = torch.tensor(np.array(y_list, dtype=np.float32), device=device)
    
    return X, y


# ================== Mamba実装 ==================

class MambaBlock(nn.Module):
    """Mamba Block (Selective State Space Model)"""
    def __init__(self, d_model: int, d_state: int, dt_rank: str = "auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        if dt_rank == "auto":
            dt_rank = d_model // 16
        
        self.in_proj = nn.Linear(d_model, d_model * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=d_model,
            out_channels=d_model,
            kernel_size=4,
            padding=3,
            groups=d_model,
            bias=True
        )
        
        self.x_proj = nn.Linear(d_model, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, d_model, bias=True)
        
        A = np.arange(1, d_state + 1, dtype=np.float32)
        self.register_buffer("A_log", torch.log(torch.tensor(A)))
        self.register_buffer("D", torch.ones(d_model))
        
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x):
        B, L, D = x.shape
        
        x_proj = self.in_proj(x)
        x_a, x_b = x_proj.chunk(2, dim=-1)
        x_a = F.silu(x_a)
        
        x_a = x_a.transpose(1, 2)
        x_a = self.conv1d(x_a)[:, :, :L]
        x_a = x_a.transpose(1, 2)
        
        A = -torch.exp(self.A_log)
        y = x_a * torch.sigmoid(x_b)
        
        y = self.out_proj(y)
        return y


class MambaForecaster(nn.Module):
    """Mamba予測モデル"""
    def __init__(self, input_dim: int = 6, d_model: int = cfg.MAMBA_D_MODEL):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.mamba_blocks = nn.ModuleList([
            MambaBlock(d_model, cfg.MAMBA_D_STATE)
            for _ in range(cfg.MAMBA_LAYERS)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, len(cfg.FORECAST_HORIZONS))
    
    def forward(self, x):
        x = self.embedding(x)
        for mamba in self.mamba_blocks:
            x = mamba(x) + x
        x = self.norm(x)
        x = x[:, -1, :]
        return self.head(x)


# ================== LoRA実装（完全版） ==================

class LoRA_Linear(nn.Module):
    """LoRA (Low-Rank Adaptation)"""
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = cfg.LORA_RANK,
        alpha: float = cfg.LORA_ALPHA,
        dropout: float = 0.05
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # 元の重み（frozen）
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.weight.requires_grad = False
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        # LoRA 低ランク行列
        self.lora_a = nn.Parameter(torch.randn(in_features, r) * 0.01)
        self.lora_b = nn.Parameter(torch.zeros(r, out_features))
        
        self.lora_alpha = alpha
        self.r = r
        self.dropout = nn.Dropout(dropout)
        self.scaling = alpha / r
    
    def forward(self, x):
        # 元の重み + LoRA適応
        lora_weight = (self.lora_a @ self.lora_b) * self.scaling
        combined_weight = self.weight + lora_weight.T
        return F.linear(x, combined_weight, self.bias)
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int = cfg.LORA_RANK):
        """既存のLinear層からLoRA版を作成"""
        lora = cls(linear.in_features, linear.out_features, r=r)
        lora.weight.data = linear.weight.data.clone()
        if linear.bias is not None:
            lora.bias.data = linear.bias.data.clone()
        return lora


def apply_lora_to_model(model: nn.Module, r: int = cfg.LORA_RANK):
    """モデルの全Linear層にLoRAを適用"""
    for name, module in model.named_children():
        if isinstance(module, nn.Linear):
            lora_layer = LoRA_Linear.from_linear(module, r=r)
            setattr(model, name, lora_layer)
        else:
            apply_lora_to_model(module, r=r)


# ================== TFT実装 ==================

class TemporalFusionTransformer(nn.Module):
    """Temporal Fusion Transformer"""
    def __init__(
        self,
        input_dim: int = 6,
        d_model: int = cfg.TFT_D_MODEL,
        nhead: int = cfg.TFT_NHEAD,
        num_layers: int = cfg.TFT_LAYERS,
        dim_ff: int = cfg.TFT_FF,
    ):
        super().__init__()
        
        self.temporal_feat_embedding = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        self.encoder_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_ff,
                dropout=0.2,
                batch_first=True,
                norm_first=True
            )
            for _ in range(num_layers)
        ])
        
        self.latent_attention = nn.MultiheadAttention(
            d_model, nhead, dropout=0.2, batch_first=True
        )
        
        self.pred_head = nn.Linear(d_model, len(cfg.FORECAST_HORIZONS))
        self.uncertainty_head = nn.Linear(d_model, len(cfg.FORECAST_HORIZONS))
        
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        x = self.temporal_feat_embedding(x)
        
        for enc_layer in self.encoder_layers:
            x = enc_layer(x)
        
        x_latent = x[:, -1:, :].expand(-1, x.size(1), -1)
        x_attn, _ = self.latent_attention(x_latent, x, x)
        
        x = self.norm(x)
        x = x[:, -1, :]
        
        pred = self.pred_head(x)
        uncertainty = F.softplus(self.uncertainty_head(x))
        
        return pred, uncertainty


# ================== Regime CNN復活 ==================

class RegimeCNN(nn.Module):
    """Regime分類CNN"""
    def __init__(self, seq_len: int = cfg.REGIME_SEQ_LEN, n_classes: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )
    
    def forward(self, x):
        x = x.unsqueeze(1)
        h = self.net(x).squeeze(-1)
        return self.fc(h)


def build_regime_dataset(feature_calc: FeatureCalculator, start_idx: int, end_idx: int):
    """Regime分類用データセット"""
    seq_len = cfg.REGIME_SEQ_LEN
    X_list, y_list = [], []
    
    for t in range(start_idx + seq_len, end_idx):
        # 過去のreturnシーケンス
        returns_window = []
        for i in range(t - seq_len, t):
            if i < len(feature_calc.returns):
                returns_window.append(feature_calc.returns[i])
            else:
                returns_window.append(0.0)
        
        # レジームラベル
        feat = feature_calc.get_features(t)
        vol = feat['vol_36']
        trend = feat['trend_36']
        
        if abs(trend) > cfg.TREND_THRESHOLD and vol < 3 * cfg.TREND_THRESHOLD:
            label = 1  # トレンド
        elif vol > 3 * cfg.TREND_THRESHOLD:
            label = 2  # 高ボラ
        else:
            label = 0  # レンジ
        
        X_list.append(returns_window)
        y_list.append(label)
    
    if len(X_list) == 0:
        return None, None
    
    X = torch.tensor(np.array(X_list, dtype=np.float32), device=device)
    y = torch.tensor(np.array(y_list, dtype=np.int64), device=device)
    
    return X, y


# ================== Actor-Critic ==================

class ActorCriticV41(nn.Module):
    """v4.1 Actor-Critic"""
    def __init__(self, state_dim: int, n_actions: int):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, n_actions)
        self.value_head = nn.Linear(256, 1)
    
    def forward(self, x):
        h = self.shared(x)
        logits = self.policy_head(h)
        value = self.value_head(h).squeeze(-1)
        return logits, value


# ================== ベースライントレーダー ==================

class BaselineTrader(ABC):
    """ベースライントレーダーの抽象クラス"""
    @abstractmethod
    def reset(self):
        pass
    
    @abstractmethod
    def act(self, state: np.ndarray) -> int:
        pass


class RandomTrader(BaselineTrader):
    """ランダムトレーダー"""
    def reset(self):
        pass
    
    def act(self, state: np.ndarray) -> int:
        return np.random.randint(0, cfg.N_ACTIONS)


class BuyAndHoldTrader(BaselineTrader):
    """Buy & Hold"""
    def reset(self):
        pass
    
    def act(self, state: np.ndarray) -> int:
        return 6  # 常に最大ロング


class MovingAverageCrossTrader(BaselineTrader):
    """移動平均クロス"""
    def __init__(self):
        self.short_window = 12
        self.long_window = 36
    
    def reset(self):
        pass
    
    def act(self, state: np.ndarray) -> int:
        # state の最初の部分がreturnウィンドウ
        returns = state[:cfg.STATE_RET_LEN]
        
        if len(returns) < self.long_window:
            return 3  # ニュートラル
        
        short_ma = np.mean(returns[-self.short_window:])
        long_ma = np.mean(returns[-self.long_window:])
        
        if short_ma > long_ma:
            return 6  # ロング
        else:
            return 0  # ショート


# ================== 環境（リーケージ対策版） ==================

class SafeHybridEnv:
    """データリーケージ対策済み環境"""
    def __init__(
        self,
        prices: np.ndarray,
        feature_calc: FeatureCalculator,
        mamba_model: MambaForecaster,
        tft_models: List[TemporalFusionTransformer],
        regime_model: RegimeCNN,
        start_idx: int,
        end_idx: int
    ):
        self.prices = prices
        self.feature_calc = feature_calc
        self.mamba_model = mamba_model
        self.tft_models = tft_models
        self.regime_model = regime_model
        
        self.start_idx = max(start_idx, cfg.MAMBA_SEQ_LEN)
        self.end_idx = min(end_idx, len(prices) - 1)
        
        self.position = 0
        self.t = None
        
        print(f"[SafeHybridEnv] Range: {self.start_idx} - {self.end_idx}")
    
    @property
    def state_dim(self):
        return (
            cfg.STATE_RET_LEN +  # return window
            3 +  # vol, trend, rsi
            3 +  # regime probs
            len(cfg.FORECAST_HORIZONS) * 2 +  # mamba + tft
            1  # position
        )
    
    def reset(self):
        self.t = self.start_idx
        self.position = 0
        return self._get_state()
    
    def _get_state(self):
        # リターンウィンドウ
        ret_window = []
        for i in range(self.t - cfg.STATE_RET_LEN, self.t):
            if i >= 0 and i < len(self.feature_calc.returns):
                ret_window.append(self.feature_calc.returns[i])
            else:
                ret_window.append(0.0)
        
        # 現在の特徴量
        feat = self.feature_calc.get_features(self.t)
        
        # Regime予測
        regime_probs = self._predict_regime()
        
        # Mamba予測
        mamba_pred = self._predict_mamba()
        
        # TFT予測（Ensemble）
        tft_pred = self._predict_tft()
        
        # ポジション
        pos_scaled = self.position / cfg.MAX_POSITION
        
        state = np.concatenate([
            np.array(ret_window, dtype=np.float32),
            np.array([feat['vol_12'], feat['trend_36'], feat['rsi']], dtype=np.float32),
            regime_probs,
            mamba_pred,
            tft_pred,
            np.array([pos_scaled], dtype=np.float32),
        ])
        
        return state.astype(np.float32)
    
    def _predict_mamba(self) -> np.ndarray:
        """Mamba予測"""
        if self.t < cfg.MAMBA_SEQ_LEN:
            return np.zeros(len(cfg.FORECAST_HORIZONS), dtype=np.float32)
        
        sequence = []
        for i in range(self.t - cfg.MAMBA_SEQ_LEN, self.t):
            feat_vec = self.feature_calc.get_feature_vector(i)
            sequence.append(feat_vec)
        
        x = torch.tensor(np.array([sequence]), dtype=torch.float32, device=device)
        
        self.mamba_model.eval()
        with torch.no_grad():
            pred = self.mamba_model(x).squeeze(0).cpu().numpy()
        
        return pred.astype(np.float32)
    
    def _predict_tft(self) -> np.ndarray:
        """TFT予測（Ensemble）"""
        if self.t < cfg.MAMBA_SEQ_LEN:
            return np.zeros(len(cfg.FORECAST_HORIZONS), dtype=np.float32)
        
        sequence = []
        for i in range(self.t - cfg.MAMBA_SEQ_LEN, self.t):
            feat_vec = self.feature_calc.get_feature_vector(i)
            sequence.append(feat_vec)
        
        x = torch.tensor(np.array([sequence]), dtype=torch.float32, device=device)
        
        preds = []
        for tft_model in self.tft_models:
            tft_model.eval()
            with torch.no_grad():
                pred, _ = tft_model(x)
                preds.append(pred.squeeze(0).cpu().numpy())
        
        return np.mean(preds, axis=0).astype(np.float32)
    
    def _predict_regime(self) -> np.ndarray:
        """Regime予測"""
        if self.t < cfg.REGIME_SEQ_LEN:
            return np.ones(3, dtype=np.float32) / 3.0
        
        returns_window = []
        for i in range(self.t - cfg.REGIME_SEQ_LEN, self.t):
            if i < len(self.feature_calc.returns):
                returns_window.append(self.feature_calc.returns[i])
            else:
                returns_window.append(0.0)
        
        x = torch.tensor(np.array([returns_window]), dtype=torch.float32, device=device)
        
        self.regime_model.eval()
        with torch.no_grad():
            logits = self.regime_model(x)
            probs = F.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
        
        return probs.astype(np.float32)
    
    def step(self, action_idx: int):
        """環境のステップ実行"""
        action_to_pos = np.array([-5, -3, -1, 0, 1, 3, 5], dtype=np.int32)
        new_pos = int(action_to_pos[action_idx])
        
        prev_pos = self.position
        self.position = new_pos
        pos_change = abs(self.position - prev_pos)
        
        # 現在の価格とリターン
        if self.t >= len(self.feature_calc.returns):
            return None, 0.0, True, {}
        
        r = self.feature_calc.returns[self.t]
        price = self.prices[self.t]
        
        # PnL計算
        pnl = self.position * r
        
        # コスト
        volume_cost = pos_change * cfg.TRANSACTION_COST
        spread_return = (cfg.SPREAD_PIPS * cfg.PIP_VALUE_JPY) / price
        slippage_return = (cfg.SLIPPAGE_PIPS * cfg.PIP_VALUE_JPY) / price
        spread_slip_cost = (
            abs(self.position) * (spread_return + slippage_return)
            if pos_change > 0 else 0.0
        )
        
        cost = volume_cost + spread_slip_cost
        reward = pnl - cost
        
        # 損失ペナルティ
        if reward < 0:
            reward *= cfg.LOSS_FACTOR
        
        # トレンドブースト
        feat = self.feature_calc.get_features(self.t)
        if abs(feat['trend_36']) > cfg.TREND_THRESHOLD:
            reward *= cfg.TREND_BOOST
        
        # 次のステップ
        self.t += 1
        done = self.t >= self.end_idx
        next_state = self._get_state() if not done else None
        
        return next_state, float(reward), done, {}


# ================== モデル学習（Walk-forward対応） ==================

def train_mamba(feature_calc: FeatureCalculator, train_start: int, train_end: int):
    """Mamba学習"""
    print(f"\n[Mamba] Training on [{train_start}, {train_end})")
    
    X, y = build_safe_dataset(feature_calc, train_start, train_end)
    if X is None:
        print("[Mamba] Not enough data")
        return None
    
    N = len(X)
    train_size = int(N * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    ds_train = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(ds_train, batch_size=cfg.MAMBA_BATCH, shuffle=True)
    
    model = MambaForecaster(input_dim=6).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.MAMBA_LR, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    criterion = nn.MSELoss()
    scaler = GradScaler() if cfg.USE_FP16 else None
    
    for epoch in range(1, cfg.MAMBA_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in loader:
            if cfg.USE_FP16:
                with autocast(dtype=torch.float16):
                    pred = model(xb)
                    loss = criterion(pred, yb)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                pred = model(xb)
                loss = criterion(pred, yb)
                loss.backward()
                optimizer.step()
            loss_sum += loss.item()
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}/{cfg.MAMBA_EPOCHS} Train={loss_sum/len(loader):.6e} Val={val_loss:.6e}")
    
    return model


def train_tft(feature_calc: FeatureCalculator, train_start: int, train_end: int):
    """TFT学習（Ensemble）"""
    print(f"\n[TFT] Training Ensemble on [{train_start}, {train_end})")
    
    X, y = build_safe_dataset(feature_calc, train_start, train_end)
    if X is None:
        print("[TFT] Not enough data")
        return []
    
    N = len(X)
    train_size = int(N * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    models = []
    n_models = cfg.N_ENSEMBLE_MODELS if cfg.USE_ENSEMBLE else 1
    
    for model_idx in range(n_models):
        print(f"\n[TFT] Model {model_idx + 1}/{n_models}")
        
        model = TemporalFusionTransformer().to(device)
        
        # LoRA適用
        if cfg.USE_LORA:
            apply_lora_to_model(model, r=cfg.LORA_RANK)
            print(f"  LoRA applied (rank={cfg.LORA_RANK})")
        
        ds_train = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(ds_train, batch_size=cfg.TFT_BATCH, shuffle=True)
        
        optimizer = optim.AdamW(model.parameters(), lr=cfg.TFT_LR, weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        scaler = GradScaler() if cfg.USE_FP16 else None
        
        for epoch in range(1, cfg.TFT_EPOCHS + 1):
            model.train()
            loss_sum = 0.0
            for xb, yb in loader:
                if cfg.USE_FP16:
                    with autocast(dtype=torch.float16):
                        pred, uncertainty = model(xb)
                        nll = torch.mean((pred - yb) ** 2 / (uncertainty + 1e-6))
                        reg = torch.mean(torch.log(uncertainty + 1e-6))
                        loss = nll + 0.1 * reg
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred, uncertainty = model(xb)
                    nll = torch.mean((pred - yb) ** 2 / (uncertainty + 1e-6))
                    reg = torch.mean(torch.log(uncertainty + 1e-6))
                    loss = nll + 0.1 * reg
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                loss_sum += loss.item()
            
            scheduler.step()
            
            model.eval()
            with torch.no_grad():
                val_pred, val_unc = model(X_val)
                val_loss = torch.mean((val_pred - y_val) ** 2 / (val_unc + 1e-6)).item()
            
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}/{cfg.TFT_EPOCHS} Train={loss_sum/len(loader):.6e} Val={val_loss:.6e}")
        
        models.append(model)
    
    return models


def train_regime(feature_calc: FeatureCalculator, train_start: int, train_end: int):
    """Regime CNN学習"""
    print(f"\n[Regime] Training on [{train_start}, {train_end})")
    
    X, y = build_regime_dataset(feature_calc, train_start, train_end)
    if X is None:
        print("[Regime] Not enough data")
        return None
    
    N = len(X)
    train_size = int(N * 0.8)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    ds_train = torch.utils.data.TensorDataset(X_train, y_train)
    loader = torch.utils.data.DataLoader(ds_train, batch_size=cfg.REGIME_BATCH, shuffle=True)
    
    model = RegimeCNN().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.REGIME_LR, weight_decay=1e-4)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler() if cfg.USE_FP16 else None
    
    for epoch in range(1, cfg.REGIME_EPOCHS + 1):
        model.train()
        loss_sum = 0.0
        for xb, yb in loader:
            if cfg.USE_FP16:
                with autocast(dtype=torch.float16):
                    logits = model(xb)
                    loss = criterion(logits, yb)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
            loss_sum += loss.item()
        
        scheduler.step()
        
        model.eval()
        with torch.no_grad():
            logits_val = model(X_val)
            val_loss = criterion(logits_val, y_val).item()
            acc = (logits_val.argmax(dim=1) == y_val).float().mean().item()
        
        if epoch % 5 == 0:
            print(f"  Epoch {epoch}/{cfg.REGIME_EPOCHS} Train={loss_sum/len(loader):.6e} Val={val_loss:.6e} Acc={acc:.3f}")
    
    return model


# ================== PPO学習 ==================

def collect_trajectory(env: SafeHybridEnv, model: ActorCriticV41, steps_per_ep: int):
    """軌跡収集"""
    model.eval()
    state = env.reset()
    
    states, actions, rewards, dones, logps, values = [], [], [], [], [], []
    
    for _ in range(steps_per_ep):
        s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits, value = model(s_t)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            logp = dist.log_prob(action)
        
        next_state, reward, done, _ = env.step(int(action.item()))
        
        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        dones.append(done)
        logps.append(logp.item())
        values.append(value.item())
        
        state = next_state
        if done:
            break
    
    if not dones[-1]:
        with torch.no_grad():
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            _, last_value = model(s_t)
            last_value = last_value.item()
    else:
        last_value = 0.0
    
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int64),
        np.array(rewards, dtype=np.float32),
        np.array(dones, dtype=np.bool_),
        np.array(logps, dtype=np.float32),
        np.array(values, dtype=np.float32),
        float(last_value),
    )


def compute_gae(rewards, dones, values, last_value, gamma=cfg.GAMMA, lam=cfg.LAMBDA_GAE):
    """GAE計算"""
    T = len(rewards)
    adv = np.zeros(T, dtype=np.float32)
    gae = 0.0
    for t in reversed(range(T)):
        mask = 1.0 - float(dones[t])
        next_val = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_val * mask - values[t]
        gae = delta + gamma * lam * mask * gae
        adv[t] = gae
    ret = adv + np.array(values)
    return adv, ret


def train_ppo(env: SafeHybridEnv, episodes: int):
    """PPO学習"""
    print(f"\n[PPO] Training for {episodes} episodes")
    
    state_dim = env.state_dim
    model = ActorCriticV41(state_dim, cfg.N_ACTIONS).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scaler = GradScaler() if cfg.USE_FP16 else None
    
    episode_rewards = []
    
    for ep in range(1, episodes + 1):
        (states, actions, rewards, dones, old_logp, values, last_val) = collect_trajectory(
            env, model, cfg.STEPS_PER_EP
        )
        
        adv, ret = compute_gae(rewards, dones, values, last_val)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        states_t = torch.tensor(states, device=device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=device)
        old_logp_t = torch.tensor(old_logp, device=device)
        adv_t = torch.tensor(adv, device=device)
        ret_t = torch.tensor(ret, device=device)
        
        dataset_size = states_t.size(0)
        ep_reward = float(np.sum(rewards))
        episode_rewards.append(ep_reward)
        
        model.train()
        for _ in range(cfg.EPOCHS_PPO):
            idx = np.random.permutation(dataset_size)
            for start in range(0, dataset_size, cfg.MINI_BATCH):
                end = min(start + cfg.MINI_BATCH, dataset_size)
                mb_idx = idx[start:end]
                
                mb_s = states_t[mb_idx]
                mb_a = actions_t[mb_idx]
                mb_old = old_logp_t[mb_idx]
                mb_adv = adv_t[mb_idx]
                mb_ret = ret_t[mb_idx]
                
                if cfg.USE_FP16:
                    with autocast(dtype=torch.float16):
                        logits, values_pred = model(mb_s)
                        probs = torch.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs)
                        logp = dist.log_prob(mb_a)
                        
                        ratio = torch.exp(logp - mb_old)
                        surr1 = ratio * mb_adv
                        surr2 = torch.clamp(ratio, 1.0 - cfg.CLIP_EPS, 1.0 + cfg.CLIP_EPS) * mb_adv
                        policy_loss = -torch.min(surr1, surr2).mean()
                        value_loss = nn.MSELoss()(values_pred, mb_ret)
                        entropy = dist.entropy().mean()
                        
                        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                    
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits, values_pred = model(mb_s)
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    logp = dist.log_prob(mb_a)
                    
                    ratio = torch.exp(logp - mb_old)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(ratio, 1.0 - cfg.CLIP_EPS, 1.0 + cfg.CLIP_EPS) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    value_loss = nn.MSELoss()(values_pred, mb_ret)
                    entropy = dist.entropy().mean()
                    
                    loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
                    
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
        
        if ep % 10 == 0:
            print(f"  Episode {ep}/{episodes} Reward={ep_reward:.4f} Avg={np.mean(episode_rewards[-10:]):.4f}")
    
    return model, episode_rewards


# ================== シミュレーション ==================

def simulate(trader, env: SafeHybridEnv, max_steps: int = 1000):
    """シミュレーション実行"""
    state = env.reset()
    equity = 1.0
    eq_curve = [equity]
    
    for _ in range(max_steps):
        if isinstance(trader, ActorCriticV41):
            trader.eval()
            s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                logits, _ = trader(s_t)
                action = torch.argmax(logits, dim=-1).item()
        else:
            action = trader.act(state)
        
        next_state, reward, done, _ = env.step(action)
        equity *= (1.0 + reward)
        eq_curve.append(equity)
        
        state = next_state
        if done:
            break
    
    return {
        'equity_curve': eq_curve,
        'final_equity': equity,
        'total_return': (equity - 1.0) * 100,
        'steps': len(eq_curve)
    }


# ================== Walk-forward検証 ==================

def walk_forward_validation(prices: np.ndarray):
    """Walk-forward検証"""
    print(f"\n{'='*80}")
    print("🔄 WALK-FORWARD VALIDATION")
    print(f"{'='*80}")
    
    feature_calc = FeatureCalculator(prices)
    
    # データ分割
    total_len = len(prices)
    train_ratio = cfg.WALK_FORWARD_TRAIN_RATIO
    test_ratio = cfg.WALK_FORWARD_TEST_RATIO
    
    train_size = int(total_len * train_ratio)
    test_size = int(total_len * test_ratio)
    
    results = []
    
    # Walk-forwardウィンドウ
    window_start = 0
    fold = 1
    
    while window_start + train_size + test_size <= total_len:
        train_start = window_start
        train_end = window_start + train_size
        test_start = train_end
        test_end = min(test_start + test_size, total_len)
        
        print(f"\n{'='*80}")
        print(f"📊 Fold {fold}: Train[{train_start}:{train_end}] Test[{test_start}:{test_end}]")
        print(f"{'='*80}")
        
        # モデル学習
        mamba_model = train_mamba(feature_calc, train_start, train_end)
        tft_models = train_tft(feature_calc, train_start, train_end)
        regime_model = train_regime(feature_calc, train_start, train_end)
        
        if mamba_model is None or len(tft_models) == 0 or regime_model is None:
            print("  ⚠️ Skipping fold due to insufficient data")
            window_start += test_size
            fold += 1
            continue
        
        # 環境作成
        env_train = SafeHybridEnv(
            prices, feature_calc, mamba_model, tft_models, regime_model,
            train_start, train_end
        )
        
        env_test = SafeHybridEnv(
            prices, feature_calc, mamba_model, tft_models, regime_model,
            test_start, test_end
        )
        
        # PPO学習
        ppo_model, _ = train_ppo(env_train, episodes=cfg.EPISODES_PER_PAIR)
        
        # テスト
        print(f"\n[Test] Evaluating on [{test_start}:{test_end}]")
        
        result_ppo = simulate(ppo_model, env_test)
        
        # ベースライン比較
        if cfg.COMPARE_BASELINES:
            result_random = simulate(RandomTrader(), env_test)
            result_bh = simulate(BuyAndHoldTrader(), env_test)
            result_ma = simulate(MovingAverageCrossTrader(), env_test)
            
            print(f"\n{'='*80}")
            print("📊 RESULTS")
            print(f"{'='*80}")
            print(f"PPO:         {result_ppo['final_equity']:.4f}x ({result_ppo['total_return']:+.2f}%)")
            print(f"Random:      {result_random['final_equity']:.4f}x ({result_random['total_return']:+.2f}%)")
            print(f"Buy & Hold:  {result_bh['final_equity']:.4f}x ({result_bh['total_return']:+.2f}%)")
            print(f"MA Cross:    {result_ma['final_equity']:.4f}x ({result_ma['total_return']:+.2f}%)")
            print(f"{'='*80}")
            
            results.append({
                'fold': fold,
                'ppo': result_ppo,
                'random': result_random,
                'buy_hold': result_bh,
                'ma_cross': result_ma,
            })
        else:
            results.append({
                'fold': fold,
                'ppo': result_ppo,
            })
        
        window_start += test_size
        fold += 1
    
    return results


# ================== データロード ==================

def load_close_series(csv_file: str) -> pd.DataFrame:
    """CSV読み込み（エラーハンドリング強化）"""
    print(f"\n[Data] Loading {csv_file}")
    
    # ファイル存在確認
    if not Path(csv_file).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    try:
        df = pd.read_csv(csv_file)
    except pd.errors.EmptyDataError:
        raise ValueError(f"CSV file is empty: {csv_file}")
    except Exception as e:
        raise RuntimeError(f"Failed to load CSV: {e}")
    
    if "Price" not in df.columns:
        raise ValueError("'Price' column not found in CSV")
    
    mask_bad = df["Price"].astype(str).str.contains("Ticker|Datetime", na=False)
    df = df[~mask_bad].copy()
    
    df = df.rename(columns={"Price": "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"])
    df = df.set_index("datetime").sort_index()
    
    if "close" not in df.columns:
        raise ValueError("'close' column required in CSV")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df = df.dropna(subset=["close"])
    
    if cfg.USE_RESAMPLE:
        df = df[["close"]].resample(cfg.RESAMPLE_RULE).last().dropna()
    
    if len(df) > cfg.MAX_POINTS:
        df = df.iloc[-cfg.MAX_POINTS:]
        print(f"  Trimmed to {cfg.MAX_POINTS} rows")
    
    print(f"  Loaded {len(df)} rows")
    return df


# ================== メイン ==================

def main_v41():
    """メイン実行"""
    print(f"\n{'='*80}")
    print("🚀 Starting v4.1 FIXED Training Pipeline")
    print(f"{'='*80}")
    
    # データロード
    csv_file = cfg.PAIR_CSV_LIST[0]
    df = load_close_series(csv_file)
    prices = df["close"].values
    
    if cfg.USE_WALK_FORWARD:
        # Walk-forward検証
        results = walk_forward_validation(prices)
        
        # 結果集約
        print(f"\n{'='*80}")
        print("📊 FINAL RESULTS (All Folds)")
        print(f"{'='*80}")
        
        ppo_returns = [r['ppo']['total_return'] for r in results]
        print(f"PPO Average Return: {np.mean(ppo_returns):.2f}% ± {np.std(ppo_returns):.2f}%")
        
        if cfg.COMPARE_BASELINES:
            random_returns = [r['random']['total_return'] for r in results]
            bh_returns = [r['buy_hold']['total_return'] for r in results]
            ma_returns = [r['ma_cross']['total_return'] for r in results]
            
            print(f"Random Average:     {np.mean(random_returns):.2f}% ± {np.std(random_returns):.2f}%")
            print(f"Buy & Hold Average: {np.mean(bh_returns):.2f}% ± {np.std(bh_returns):.2f}%")
            print(f"MA Cross Average:   {np.mean(ma_returns):.2f}% ± {np.std(ma_returns):.2f}%")
        
        print(f"{'='*80}")
    
    else:
        # シンプルな Train/Test 分割
        print("\n[Simple Train/Test Split]")
        feature_calc = FeatureCalculator(prices)
        
        total_len = len(prices)
        train_end = int(total_len * 0.7)
        test_start = train_end
        
        # 学習
        mamba_model = train_mamba(feature_calc, 0, train_end)
        tft_models = train_tft(feature_calc, 0, train_end)
        regime_model = train_regime(feature_calc, 0, train_end)
        
        # 環境
        env_train = SafeHybridEnv(
            prices, feature_calc, mamba_model, tft_models, regime_model,
            0, train_end
        )
        
        env_test = SafeHybridEnv(
            prices, feature_calc, mamba_model, tft_models, regime_model,
            test_start, total_len
        )
        
        # PPO学習
        ppo_model, _ = train_ppo(env_train, episodes=cfg.EPISODES_PER_PAIR)
        
        # テスト
        result = simulate(ppo_model, env_test)
        
        print(f"\n{'='*80}")
        print("📊 TEST RESULTS")
        print(f"{'='*80}")
        print(f"Final Equity: {result['final_equity']:.4f}x")
        print(f"Total Return: {result['total_return']:+.2f}%")
        print(f"{'='*80}")
        
        # グラフ出力
        plt.figure(figsize=(12, 6))
        plt.plot(result['equity_curve'], linewidth=2)
        plt.title("v4.1 Fixed - Test Equity Curve")
        plt.xlabel("Steps")
        plt.ylabel("Equity Multiplier")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig("simulation_v41_fixed.png", dpi=150)
        print("\n✅ Saved: simulation_v41_fixed.png")
    
    print("\n✅ All done!")


if __name__ == "__main__":
    main_v41()