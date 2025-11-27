import os
import sys

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')


import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import wandb

# 引用你的模組
from model import config_new
from model.dataset import TuringDataset
from model.model import ParameterNet

# ============================================================
# 0. 固定隨機種子
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Random seed set to {seed}")

# ============================================================
# 1. Forward Surrogate (參數 -> 圖片)
# ============================================================
class ForwardSurrogate(nn.Module):
    def __init__(self):
        super(ForwardSurrogate, self).__init__()
        # Input: 4 params -> Output: 128x128 image
        self.fc = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 256 * 8 * 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 32 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),    # 64 -> 128
            nn.Sigmoid() 
        )

    def forward(self, p):
        x = self.fc(p).view(-1, 256, 8, 8)
        return self.decoder(x)

# ============================================================
# 2. Loss Functions (物理約束的核心)
# ============================================================
class FourierLoss(nn.Module):
    """比較頻譜幅度 (忽略相位/位置差異，專注於波長與方向)"""
    def __init__(self, radius=0.7):
        super(FourierLoss, self).__init__()
        self.radius = radius

    def forward(self, recon_img, true_img):
        B, C, H, W = recon_img.shape
        # 2D FFT
        fft_recon = torch.fft.fft2(recon_img)
        fft_true = torch.fft.fft2(true_img)
        
        # 取幅度譜 (Magnitude)
        mag_recon = torch.abs(fft_recon)
        mag_true = torch.abs(fft_true)
        
        # 建立低頻遮罩 (Mask)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=recon_img.device),
            torch.linspace(-1, 1, W, device=recon_img.device),
            indexing="ij"
        )
        rr = torch.sqrt(xx**2 + yy**2)
        mask = (rr < self.radius).float()
        
        # 計算遮罩後的 MSE
        diff = (mag_recon - mag_true) ** 2
        return torch.mean(diff * mask)

class HistogramLoss(nn.Module):
    """比較像素值分佈 (忽略位置，專注於黑白比例與對比度)"""
    def __init__(self):
        super(HistogramLoss, self).__init__()
        
    def forward(self, recon_img, true_img):
        # Flatten: (B, C, H, W) -> (B, H*W)
        b = recon_img.size(0)
        recon_flat = recon_img.view(b, -1)
        true_flat = true_img.view(b, -1)
        
        # Sorting (排序後比較 = 比較分佈)
        recon_sorted, _ = torch.sort(recon_flat, dim=1)
        true_sorted, _ = torch.sort(true_flat, dim=1)
        
        # L1 Loss on sorted pixels
        return torch.mean(torch.abs(recon_sorted - true_sorted))

# ============================================================
# 3. Validation Functions
# ============================================================
def validate_surrogate(model, loader, criterion_four, criterion_hist):
    model.eval()
    total_four = 0.0
    total_hist = 0.0
    
    with torch.no_grad():
        for u_batch, _, p_batch in loader:
            u_batch = u_batch.to(config_new.DEVICE)
            p_batch = p_batch.to(config_new.DEVICE)
            
            recon = model(p_batch)
            loss_f = criterion_four(recon, u_batch)
            loss_h = criterion_hist(recon, u_batch)
            
            total_four += loss_f.item()
            total_hist += loss_h.item()
            
    n = len(loader)
    return {"val_fourier": total_four/n, "val_hist": total_hist/n}

def validate_inverse(model, loader, surrogate=None, criterion_phy=None):
    model.eval()
    sum_squared = torch.zeros(4).to(config_new.DEVICE)
    count = 0
    phy_total = 0.0
    
    with torch.no_grad():
        for u_batch, _, p_batch in loader:
            u_batch = u_batch.to(config_new.DEVICE)
            p_batch = p_batch.to(config_new.DEVICE)
            
            p_pred = model(u_batch)
            
            # Data Error
            diff = p_pred - p_batch
            sum_squared += torch.sum(diff ** 2, dim=0)
            count += u_batch.size(0)
            
            # Physics Error
            if surrogate and criterion_phy:
                recon = surrogate(p_pred)
                loss_phy = criterion_phy(recon, u_batch)
                phy_total += loss_phy.item()
                
    mse = sum_squared / count
    rmse = torch.sqrt(mse)
    
    return {
        "rmse_avg": rmse.mean().item(),
        "rmse_delta": rmse[3].item(),
        "val_phy": phy_total / len(loader) if surrogate else 0.0
    }

# ============================================================
# 4. 主程式
# ============================================================
def main():
    set_seed(42)
    wandb.login(key="d969eaa6886920a565de70b8ca7ce8c9b13f6ddf")
    
    # ==========================
    # 設定開關 (手動切換)
    # ==========================
    TRAIN_SURROGATE = False  # True: 跑 Phase 1, False: 跑 Phase 2+3
    
    ckpt_dir = "checkpoint"
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    surrogate_path = os.path.join(ckpt_dir, "forward_surrogate.pth")
    inverse_path = os.path.join(ckpt_dir, "inverse_cnn_hybrid.pth")

    # --- 資料載入 ---
    dataset = TuringDataset(config_new.NPZ_PATH)
    dataset.scaler.to_device(config_new.DEVICE)
    
    train_size = int(len(dataset) * 0.8)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(
        dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_set, batch_size=config_new.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config_new.BATCH_SIZE, shuffle=False)
    
    print(f"📦 Data: Train={len(train_set)}, Val={len(val_set)}")

    # ============================================================
    # 🟦 Phase 1: 訓練 Forward Surrogate (物理模擬器)
    # ============================================================
    if TRAIN_SURROGATE:
        print("🚀 [Phase 1] Training Surrogate (Fourier + Histogram)...")
        wandb.init(project="Turing-Hybrid", name="Phase1-Surrogate")
        
        surrogate = ForwardSurrogate().to(config_new.DEVICE)
        optimizer_S = optim.Adam(surrogate.parameters(), lr=1e-3)
        scheduler_S = optim.lr_scheduler.ReduceLROnPlateau(optimizer_S, mode='min', factor=0.5, patience=5)
        
        # 定義 Loss: 頻率對齊 + 顏色分佈對齊
        criterion_four = FourierLoss(radius=0.7).to(config_new.DEVICE)
        criterion_hist = HistogramLoss().to(config_new.DEVICE)
        
        for epoch in range(60):
            surrogate.train()
            loss_sum_four = 0
            loss_sum_hist = 0
            
            for u_batch, _, p_batch in tqdm(train_loader, desc=f"Surrogate Ep{epoch+1}"):
                u_batch = u_batch.to(config_new.DEVICE)
                p_batch = p_batch.to(config_new.DEVICE)
                
                optimizer_S.zero_grad()
                recon = surrogate(p_batch)
                
                # 計算兩個 Loss
                loss_f = criterion_four(recon, u_batch)
                loss_h = criterion_hist(recon, u_batch)
                
                # 總 Loss (1:1 權重通常就夠了，也可以微調)
                loss = loss_f + loss_h
                
                loss.backward()
                optimizer_S.step()
                
                loss_sum_four += loss_f.item()
                loss_sum_hist += loss_h.item()
            
            # 驗證
            val_metrics = validate_surrogate(surrogate, val_loader, criterion_four, criterion_hist)
            scheduler_S.step(val_metrics['val_hist']) # 用 Histogram Loss 來監控收斂
            
            print(f"[Ep {epoch+1}] Train Hist={loss_sum_hist/len(train_loader):.4f} | Val Hist={val_metrics['val_hist']:.4f}")
            wandb.log({
                "train/hist_loss": loss_sum_hist/len(train_loader),
                "val/hist_loss": val_metrics['val_hist'],
                "val/four_loss": val_metrics['val_fourier']
            })
            
        torch.save(surrogate.state_dict(), surrogate_path)
        print(f"✅ Surrogate saved to {surrogate_path}")
        return

    # ============================================================
    # 🟩 Phase 2 & 3: 訓練 Inverse CNN (逆向預測)
    # ============================================================
    print("🚀 [Phase 2 & 3] Training Inverse CNN...")
    if not os.path.exists(surrogate_path):
        raise FileNotFoundError("❌ 請先跑 Phase 1 訓練 Surrogate")
    
    wandb.init(project="Turing-Hybrid", name="Phase2-3-Inverse")
    
    # 1. 載入並凍結 Surrogate
    surrogate = ForwardSurrogate().to(config_new.DEVICE)
    surrogate.load_state_dict(torch.load(surrogate_path))
    surrogate.eval()
    for p in surrogate.parameters(): p.requires_grad = False
    print("🔒 Surrogate loaded & frozen.")
    
    # 2. 準備 Inverse Model
    model = ParameterNet().to(config_new.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4) # 用小一點的 LR 求穩
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)
    
    criterion_phy = FourierLoss(radius=0.7).to(config_new.DEVICE)
    loss_weights = config_new.LOSS_WEIGHTS
    
    for epoch in range(config_new.EPOCHS):
        model.train()
        total_sup = 0
        total_phy = 0
        
        # Curriculum: 前 30 epoch 純 Data Loss, 之後加入 Physics
        lambda_phy = 0.0 if epoch < 30 else 0.01
        
        progress_bar = tqdm(train_loader, desc=f"Inverse Ep{epoch+1} (λ={lambda_phy})", leave=False)
        
        for u_batch, _, params_target in progress_bar:
            u_batch = u_batch.to(config_new.DEVICE)
            params_target = params_target.to(config_new.DEVICE)
            
            optimizer.zero_grad()
            preds_norm = model(u_batch)
            
            # Supervised Loss (L1)
            abs_diff = torch.abs(preds_norm - params_target)
            loss_sup = torch.mean(loss_weights * abs_diff)
            
            # Physics Loss (Fourier Cycle Consistency)
            loss_fourier = torch.tensor(0.0).to(config_new.DEVICE)
            if lambda_phy > 0:
                recon_img = surrogate(preds_norm)
                loss_fourier = criterion_phy(recon_img, u_batch)
                
            loss = loss_sup + lambda_phy * loss_fourier
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_sup += loss_sup.item()
            total_phy += loss_fourier.item()
            progress_bar.set_postfix({'Sup': loss_sup.item(), 'Phy': loss_fourier.item()})
            
        # 驗證
        val_metrics = validate_inverse(
            model, val_loader, 
            surrogate if lambda_phy > 0 else None, 
            criterion_phy
        )
        scheduler.step(val_metrics['rmse_avg'])
        
        avg_sup = total_sup / len(train_loader)
        
        print(f"[Ep {epoch+1}] Sup={avg_sup:.4f} | "
              f"Val NRMSE [δ={val_metrics['rmse_delta']:.2%} | Avg={val_metrics['rmse_avg']:.2%}]")
        
        wandb.log({
            "train/sup_loss": avg_sup,
            "train/phy_loss": total_phy / len(train_loader),
            "val/rmse_delta": val_metrics['rmse_delta'],
            "val/rmse_avg": val_metrics['rmse_avg']
        })
        
        torch.save(model.state_dict(), inverse_path)

    print(f"🎉 Training Finished! Model saved to {inverse_path}")

if __name__ == "__main__":
    main()