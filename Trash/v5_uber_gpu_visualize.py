# ================================================================================
# quant_meta_hybrid_trader_v5_UBER_VISUALIZE.py
#
# 全部乗せ研究用トレーダーフレームワーク v5 - UBER最強版 + 可視化機能
# Mixed Precision対応 + matplotlib グラフ出力


import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from typing import List
from dataclasses import dataclass
import warnings
import time
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

warnings.filterwarnings('ignore')


# ================== GPU最適化設定 ==================

@dataclass
class ConfigV5:
    """v5 UBER GPU最適化設定"""
    PAIR_CSV_LIST: List[str] = None
    MAX_POINTS: int = 30000
    STATE_RET_LEN: int = 64
    FORECAST_HORIZONS: List[int] = None
    
    # Wave Attention
    WAVE_SEQ_LEN: int = 128
    WAVE_D_MODEL: int = 256
    WAVE_LAYERS: int = 6
    WAVE_HEADS: int = 8
    WAVE_EPOCHS: int = 30
    WAVE_LR: float = 2e-3
    WAVE_BATCH: int = 512
    
    # MoE
    MOE_NUM_EXPERTS: int = 8
    MOE_TOP_K: int = 2
    
    # Contrastive
    USE_CONTRASTIVE: bool = True
    CONTRASTIVE_TEMP: float = 0.07
    CONTRASTIVE_WEIGHT: float = 0.3
    
    # Data Augmentation
    USE_AUGMENTATION: bool = True
    MIXUP_ALPHA: float = 0.2
    CUTMIX_ALPHA: float = 0.5
    
    # Meta
    USE_FP16: bool = True
    N_ENSEMBLE_MODELS: int = 2
    VISUALIZE: bool = True
    
    def __post_init__(self):
        if self.PAIR_CSV_LIST is None:
            self.PAIR_CSV_LIST = ["yf_USDJPYX_5m_max.csv"]
        if self.FORECAST_HORIZONS is None:
            self.FORECAST_HORIZONS = [1, 3, 6, 12, 24]


cfg = ConfigV5()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)
np.random.seed(42)

print(f"\n{'='*80}")
print(f"🚀 Quant Meta Hybrid Trader v5 - UBER GPU Edition + Visualization")
print(f"{'='*80}")
print(f"Device: {device}")
print(f"Wave Attention: {cfg.WAVE_SEQ_LEN} seq, {cfg.WAVE_D_MODEL} dim")
print(f"MoE: {cfg.MOE_NUM_EXPERTS} experts")
print(f"Visualization: {cfg.VISUALIZE}")
print(f"{'='*80}\n")


# ================== Wave Attention ==================

class WaveAttentionBlock(nn.Module):
    """Wave Attention - 波状注意機構"""
    def __init__(self, d_model: int, nhead: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        assert d_model % nhead == 0
        self.d_k = d_model // nhead
        
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        self.wave_freq = nn.Parameter(torch.randn(nhead) * 0.1)
        self.wave_phase = nn.Parameter(torch.zeros(nhead))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """x: (B, L, D)"""
        B, L, D = x.shape
        
        q = self.q_proj(x).reshape(B, L, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(x).reshape(B, L, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(x).reshape(B, L, self.nhead, self.d_k).transpose(1, 2)
        
        pos = torch.arange(L, device=x.device, dtype=x.dtype)
        wave = torch.cos(2 * np.pi * self.wave_freq.unsqueeze(-1) * pos + self.wave_phase.unsqueeze(-1))
        wave = wave.unsqueeze(0).unsqueeze(2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) * (self.d_k ** -0.5)
        scores = scores * wave
        
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, L, D)
        return self.out_proj(out)


# ================== Expert ==================

class Expert(nn.Module):
    """単一エキスパート"""
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )
    
    def forward(self, x):
        return self.net(x)


# ================== MoE（型安全版） ==================

class MixtureOfExperts(nn.Module):
    """Mixture of Experts - Mixed Precision対応"""
    def __init__(self, d_model: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.experts = nn.ModuleList([Expert(d_model) for _ in range(num_experts)])
        self.router = nn.Linear(d_model, num_experts)
    
    def forward(self, x):
        """x: (B, L, D) - 型安全化版"""
        B, L, D = x.shape
        x_orig_dtype = x.dtype
        x_flat = x.reshape(-1, D)
        
        router_logits = self.router(x_flat)
        router_weights = torch.softmax(router_logits, dim=-1)
        
        top_k_weights, top_k_indices = torch.topk(router_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        
        outputs = []
        for i in range(self.top_k):
            combined = torch.zeros_like(x_flat, dtype=x_orig_dtype)
            
            for j in range(self.num_experts):
                mask = (top_k_indices[:, i] == j)
                if mask.sum() > 0:
                    expert_input = x_flat[mask]
                    expert_output = self.experts[j](expert_input)
                    expert_output = expert_output.to(x_orig_dtype)
                    combined[mask] = expert_output
            
            outputs.append(combined * top_k_weights[:, i:i+1])
        
        output = sum(outputs) if outputs else torch.zeros_like(x_flat, dtype=x_orig_dtype)
        return output.reshape(B, L, D)


# ================== Contrastive Loss ==================

class ContrastiveLoss(nn.Module):
    """Supervised Contrastive Loss"""
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, labels):
        """embeddings: (N, D), labels: (N,)"""
        batch_size = embeddings.shape[0]
        
        embeddings = F.normalize(embeddings, p=2, dim=1)
        sim_matrix = torch.matmul(embeddings, embeddings.t()) / self.temperature
        
        mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        mask.fill_diagonal_(0)
        
        exp_sim = torch.exp(sim_matrix) * (1 - torch.eye(batch_size, device=embeddings.device))
        log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        loss = -(mask * log_prob).sum(dim=1) / (mask.sum(dim=1).clamp(min=1e-6))
        
        return loss.mean()


# ================== Data Augmentation ==================

def mixup(x, y, alpha=0.2):
    """Mixup"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = lam * y + (1 - lam) * y[idx]
    
    return mixed_x, mixed_y


def cutmix(x, y, alpha=0.5):
    """CutMix"""
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    idx = torch.randperm(batch_size, device=x.device)
    
    seq_len = x.size(1)
    cut_size = max(1, int(seq_len * np.sqrt(1 - lam)))
    cut_start = np.random.randint(0, max(1, seq_len - cut_size + 1))
    cut_end = min(cut_start + cut_size, seq_len)
    
    mixed_x = x.clone()
    mixed_x[:, cut_start:cut_end, :] = x[idx, cut_start:cut_end, :]
    mixed_y = lam * y + (1 - lam) * y[idx]
    
    return mixed_x, mixed_y


# ================== Forecaster Model ==================

class WaveTFTForecaster(nn.Module):
    """Wave Attention + TFT + MoE"""
    def __init__(self, input_dim: int = 6, d_model: int = cfg.WAVE_D_MODEL, 
                 nhead: int = 8, num_layers: int = cfg.WAVE_LAYERS):
        super().__init__()
        
        self.embedding = nn.Linear(input_dim, d_model)
        self.wave_layers = nn.ModuleList([
            WaveAttentionBlock(d_model, nhead) for _ in range(num_layers)
        ])
        self.moe_layers = nn.ModuleList([
            MixtureOfExperts(d_model, cfg.MOE_NUM_EXPERTS, cfg.MOE_TOP_K)
            for _ in range(max(1, num_layers // 2))
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.pred_head = nn.Linear(d_model, len(cfg.FORECAST_HORIZONS))
        self.uncertainty_head = nn.Linear(d_model, len(cfg.FORECAST_HORIZONS))
        self.contrastive_head = nn.Linear(d_model, 128)
    
    def forward(self, x, return_contrastive=False):
        """x: (B, L, D)"""
        x = self.embedding(x)
        
        for i, wave_layer in enumerate(self.wave_layers):
            x = wave_layer(x) + x
            
            if i % 2 == 0 and i // 2 < len(self.moe_layers):
                moe_out = self.moe_layers[i // 2](x)
                x = moe_out + x
        
        x = self.norm(x)
        x_latent = x[:, -1, :]
        
        pred = self.pred_head(x_latent)
        uncertainty = F.softplus(self.uncertainty_head(x_latent))
        
        if return_contrastive:
            contrastive = self.contrastive_head(x_latent)
            return pred, uncertainty, contrastive
        
        return pred, uncertainty


# ================== Data Loader ==================

class AugmentedDataLoader:
    """Data Loader with Augmentation"""
    def __init__(self, X, y, batch_size=512, augment=True):
        self.X = X
        self.y = y
        self.batch_size = batch_size
        self.augment = augment
        self.n = len(X)
    
    def __iter__(self):
        indices = np.random.permutation(self.n)
        
        for start_idx in range(0, self.n, self.batch_size):
            batch_indices = indices[start_idx:start_idx + self.batch_size]
            
            X_batch = torch.tensor(self.X[batch_indices], dtype=torch.float32, device=device)
            y_batch = torch.tensor(self.y[batch_indices], dtype=torch.float32, device=device)
            
            if self.augment:
                if np.random.rand() < 0.3:
                    X_batch, y_batch = mixup(X_batch, y_batch, cfg.MIXUP_ALPHA)
                elif np.random.rand() < 0.3:
                    X_batch, y_batch = cutmix(X_batch, y_batch, cfg.CUTMIX_ALPHA)
            
            yield X_batch, y_batch


# ================== 可視化関数 ==================

def plot_predictions(y_true, y_pred, y_unc, horizon_names, model_idx, output_dir="./results"):
    """予測結果の可視化"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(len(cfg.FORECAST_HORIZONS), 1, figsize=(14, 3 * len(cfg.FORECAST_HORIZONS)))
    if len(cfg.FORECAST_HORIZONS) == 1:
        axes = [axes]
    
    for idx, (h_name, ax) in enumerate(zip(horizon_names, axes)):
        y_t = y_true[:, idx]
        y_p = y_pred[:, idx]
        y_u = y_unc[:, idx]
        
        # 実際値 vs 予測値
        ax.plot(y_t, 'o-', label='Actual', alpha=0.7, markersize=4)
        ax.plot(y_p, 's--', label='Predicted', alpha=0.7, markersize=4)
        
        # 不確実性帯
        ax.fill_between(range(len(y_p)), 
                         (y_p - y_u).numpy() if hasattr(y_p, 'numpy') else y_p - y_u,
                         (y_p + y_u).numpy() if hasattr(y_p, 'numpy') else y_p + y_u,
                         alpha=0.2, label='Uncertainty')
        
        # 誤差計算
        mse = np.mean((y_t.numpy() if hasattr(y_t, 'numpy') else y_t - 
                       y_p.numpy() if hasattr(y_p, 'numpy') else y_p) ** 2)
        mae = np.mean(np.abs(y_t.numpy() if hasattr(y_t, 'numpy') else y_t - 
                            y_p.numpy() if hasattr(y_p, 'numpy') else y_p))
        
        ax.set_title(f'Horizon {h_name} - MSE: {mse:.6f}, MAE: {mae:.6f}', fontsize=12)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Return')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = f"{output_dir}/predictions_model_{model_idx}.png"
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved: {filepath}")


def plot_training_history(train_losses, val_losses, model_idx, output_dir="./results"):
    """学習曲線の可視化"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(train_losses, 'o-', label='Train Loss', alpha=0.7)
    ax.plot(val_losses, 's--', label='Val Loss', alpha=0.7)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'Model {model_idx} - Training History')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    filepath = f"{output_dir}/training_history_model_{model_idx}.png"
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"📈 Saved: {filepath}")


def plot_ensemble_comparison(all_y_preds, y_true, horizon_names, output_dir="./results"):
    """アンサンブル予測の比較"""
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    fig, axes = plt.subplots(len(cfg.FORECAST_HORIZONS), 1, figsize=(14, 3 * len(cfg.FORECAST_HORIZONS)))
    if len(cfg.FORECAST_HORIZONS) == 1:
        axes = [axes]
    
    for idx, (h_name, ax) in enumerate(zip(horizon_names, axes)):
        y_t = y_true[:, idx]
        
        # 各モデルの予測
        for model_idx, y_p in enumerate(all_y_preds):
            ax.plot(y_p[:, idx], '--', alpha=0.6, label=f'Model {model_idx}')
        
        # 平均予測
        ensemble_pred = np.mean([y_p[:, idx] for y_p in all_y_preds], axis=0)
        ax.plot(y_t, 'o-', color='black', label='Actual', linewidth=2, markersize=4)
        ax.plot(ensemble_pred, 's-', color='red', label='Ensemble Mean', linewidth=2, markersize=4)
        
        ax.set_title(f'Ensemble Comparison - Horizon {h_name}', fontsize=12)
        ax.set_xlabel('Sample')
        ax.set_ylabel('Return')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filepath = f"{output_dir}/ensemble_comparison.png"
    plt.savefig(filepath, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"🎯 Saved: {filepath}")


# ================== Training ==================

def train_forecasters_v5(prices: np.ndarray):
    """Train Wave TFT + MoE with Visualization"""
    print("\n[Training] Wave TFT + MoE Forecasters")
    
    # Features
    returns = np.diff(prices) / (prices[:-1] + 1e-8)
    r_series = pd.Series(returns)
    
    vol_12 = r_series.rolling(12).std().fillna(0.0).values
    vol_36 = r_series.rolling(36).std().fillna(0.0).values
    trend_36 = r_series.rolling(36).mean().fillna(0.0).values
    
    up = r_series.clip(lower=0).rolling(14).mean()
    down = (-r_series.clip(upper=0)).rolling(14).mean()
    rsi = 100.0 * up / (up + down + 1e-9)
    rsi = ((rsi - 50.0) / 50.0).fillna(0.0).values
    
    returns_smooth = pd.Series(returns).ewm(span=10).mean().values
    
    feat_mat = np.stack([returns, vol_12, vol_36, trend_36, rsi, returns_smooth], axis=1).astype(np.float32)
    
    r = returns
    T = feat_mat.shape[0]
    seq_len = cfg.WAVE_SEQ_LEN
    horizon_max = max(cfg.FORECAST_HORIZONS)
    
    X_list, y_list = [], []
    for t in range(seq_len, T - horizon_max):
        X_list.append(feat_mat[t - seq_len: t])
        targets = [r[t + h - 1] for h in cfg.FORECAST_HORIZONS]
        y_list.append(targets)
    
    X_seq = np.array(X_list, dtype=np.float32)
    y_seq = np.array(y_list, dtype=np.float32)
    
    N = len(X_seq)
    train_size = int(N * 0.8)
    X_train, X_val = X_seq[:train_size], X_seq[train_size:]
    y_train, y_val = y_seq[:train_size], y_seq[train_size:]
    
    # Train models
    print("[WaveTFT] Training Ensemble...")
    models = []
    all_train_losses = []
    all_val_losses = []
    all_y_preds = []
    
    for model_idx in range(cfg.N_ENSEMBLE_MODELS):
        print(f"\n[WaveTFT] Model {model_idx + 1}/{cfg.N_ENSEMBLE_MODELS}")
        
        model = WaveTFTForecaster(input_dim=X_seq.shape[2]).to(device)
        
        scaler = GradScaler() if cfg.USE_FP16 else None
        optimizer = optim.AdamW(model.parameters(), lr=cfg.WAVE_LR, weight_decay=1e-4)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)
        
        contrastive_loss_fn = ContrastiveLoss(cfg.CONTRASTIVE_TEMP) if cfg.USE_CONTRASTIVE else None
        
        train_loader = AugmentedDataLoader(X_train, y_train, batch_size=cfg.WAVE_BATCH, augment=cfg.USE_AUGMENTATION)
        
        train_losses, val_losses = [], []
        
        for epoch in range(1, cfg.WAVE_EPOCHS + 1):
            model.train()
            loss_sum = 0.0
            batch_count = 0
            
            for xb, yb in train_loader:
                if cfg.USE_FP16:
                    with autocast(dtype=torch.float16):
                        pred, uncertainty, contrastive = model(xb, return_contrastive=True)
                        
                        main_loss = torch.mean((pred - yb) ** 2 / (uncertainty + 1e-6))
                        
                        if cfg.USE_CONTRASTIVE:
                            y_labels = (torch.sign(yb[:, 0]) + 1).long().clamp(0, 2)
                            cont_loss = contrastive_loss_fn(contrastive, y_labels)
                            loss = main_loss + cfg.CONTRASTIVE_WEIGHT * cont_loss
                        else:
                            loss = main_loss
                    
                    optimizer.zero_grad()
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    pred, uncertainty, contrastive = model(xb, return_contrastive=True)
                    main_loss = torch.mean((pred - yb) ** 2 / (uncertainty + 1e-6))
                    
                    if cfg.USE_CONTRASTIVE:
                        y_labels = (torch.sign(yb[:, 0]) + 1).long().clamp(0, 2)
                        cont_loss = contrastive_loss_fn(contrastive, y_labels)
                        loss = main_loss + cfg.CONTRASTIVE_WEIGHT * cont_loss
                    else:
                        loss = main_loss
                    
                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                
                loss_sum += loss.item()
                batch_count += 1
            
            scheduler.step()
            
            # Validation
            model.eval()
            with torch.no_grad():
                X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
                y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device)
                
                val_pred, val_unc, _ = model(X_val_t, return_contrastive=True)
                val_loss = torch.mean((val_pred - y_val_t) ** 2 / (val_unc + 1e-6)).item()
            
            avg_train_loss = loss_sum / max(1, batch_count)
            train_losses.append(avg_train_loss)
            val_losses.append(val_loss)
            
            if epoch % 5 == 0:
                print(f"[WaveTFT-{model_idx}] Ep {epoch}/{cfg.WAVE_EPOCHS} | Train: {avg_train_loss:.6e} | Val: {val_loss:.6e}")
        
        # 可視化
        if cfg.VISUALIZE:
            plot_training_history(train_losses, val_losses, model_idx)
            
            # 最終予測
            with torch.no_grad():
                X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
                val_pred, val_unc, _ = model(X_val_t, return_contrastive=True)
                plot_predictions(y_val, val_pred.cpu().numpy(), val_unc.cpu().numpy(), 
                                [f"H{h}" for h in cfg.FORECAST_HORIZONS], model_idx)
                all_y_preds.append(val_pred.cpu().numpy())
        
        models.append(model)
        all_train_losses.append(train_losses)
        all_val_losses.append(val_losses)
    
    # アンサンブル比較グラフ
    if cfg.VISUALIZE and len(all_y_preds) > 0:
        plot_ensemble_comparison(all_y_preds, y_val, [f"H{h}" for h in cfg.FORECAST_HORIZONS])
    
    print("\n✅ Training Complete!")
    return models


# ================== Main ==================

def main_v5():
    """Main"""
    print("\n[Data Loading]")
    
    for csv in cfg.PAIR_CSV_LIST:
        try:
            df = pd.read_csv(csv)
            prices = df["close"].values.astype(np.float32)
        except:
            print(f"⚠️  {csv} not found, generating synthetic data")
            prices = np.random.randn(cfg.MAX_POINTS).cumsum() * 0.001 + 150
        
        prices = np.maximum(prices, 1)
        
        print(f"\n[Training] {csv}")
        models = train_forecasters_v5(prices)
        print(f"✅ Trained {len(models)} models on {len(prices)} price points")
    
    print("\n" + "="*80)
    print("🚀 v5 UBER Edition Training Complete!")
    print("📊 Check ./results/ for visualizations!")
    print("="*80)


if __name__ == "__main__":
    main_v5()
