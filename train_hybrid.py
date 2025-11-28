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
from model import config_hybrid as config_new
from model.dataset import TuringDataset
from model.model import ParameterNet
from model.loss import GradNorm, PhysicsLoss

# ============================================================
# 0. 固定隨機種子
# ============================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 根據設備類型設置對應的隨機種子
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    
    print(f"🔒 Random seed set to {seed} (Device: {config_new.DEVICE})")

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
def validate_surrogate(model, loader, criterion_four, criterion_hist, criterion_physics=None):
    model.eval()
    total_four = 0.0
    total_hist = 0.0
    total_phy = 0.0
    
    with torch.no_grad():
        for u_batch, v_batch, p_batch in loader:
            u_batch = u_batch.to(config_new.DEVICE)
            v_batch = v_batch.to(config_new.DEVICE)
            p_batch = p_batch.to(config_new.DEVICE)
            
            recon = model(p_batch)
            loss_f = criterion_four(recon, u_batch)
            loss_h = criterion_hist(recon, u_batch)
            
            # 計算 Physical Loss (如果提供)
            loss_p = torch.tensor(0.0).to(config_new.DEVICE)
            if criterion_physics is not None:
                loss_p = criterion_physics(recon, v_batch, p_batch)
            
            total_four += loss_f.item()
            total_hist += loss_h.item()
            total_phy += loss_p.item()
            
    n = len(loader)
    result = {"val_fourier": total_four/n, "val_hist": total_hist/n}
    if criterion_physics is not None:
        result["val_physics"] = total_phy/n
    return result

def validate_inverse(model, loader, surrogate=None, criterion_phy=None):
    model.eval()
    sum_squared = torch.zeros(4).to(config_new.DEVICE)
    count = 0
    phy_total = 0.0
    
    # 收集所有真實參數和預測參數用於多維 NRMSE
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for u_batch, _, p_batch in loader:
            u_batch = u_batch.to(config_new.DEVICE)
            p_batch = p_batch.to(config_new.DEVICE)
            
            p_pred = model(u_batch)
            
            # Data Error
            diff = p_pred - p_batch
            sum_squared += torch.sum(diff ** 2, dim=0)
            count += u_batch.size(0)
            
            # 收集數據
            all_targets.append(p_batch.cpu())
            all_preds.append(p_pred.cpu())
            
            # Physics Error
            if surrogate and criterion_phy:
                recon = surrogate(p_pred)
                loss_phy = criterion_phy(recon, u_batch)
                phy_total += loss_phy.item()
                
    mse = sum_squared / count
    rmse = torch.sqrt(mse)
    
    # 計算多維 NRMSE (論文定義)
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    
    num_samples = all_targets.shape[0]        # m (樣本數量)
    num_parameters = all_targets.shape[1]     # d (參數數量，此處為 4)
    num_items = num_samples * num_parameters  # m * d (所有單一預測總數)
    
    total_squared_error = np.sum((all_targets - all_preds)**2)
    rmse_multi_dim = np.sqrt(total_squared_error / num_items)
    
    # 歸一化因子：真實參數平均向量的歐幾里得範數
    y_mean_vector = np.mean(all_targets, axis=0)
    norm_y_mean = np.linalg.norm(y_mean_vector)
    
    nrmse_multi_dim = rmse_multi_dim / norm_y_mean if norm_y_mean != 0 else 0.0
    
    return {
        "rmse_avg": rmse.mean().item(),
        "rmse_delta": rmse[3].item(),
        "nrmse_multi_dim": nrmse_multi_dim,
        "val_phy": phy_total / len(loader) if surrogate else 0.0
    }

# ============================================================
# 4. 主程式
# ============================================================
def main():
    set_seed(42)
    wandb.login(key="c45f78d1fb5c9023cf8d9787e2d3828bd0f891e1")
    
    # ==========================
    # 決定訓練階段
    # ==========================
    # 優先使用環境變數 (來自 run_experiment.py)
    TRAIN_PHASE = os.environ.get('HYBRID_TRAIN_PHASE', 'both')
    
    if TRAIN_PHASE not in ['both', 'surrogate_only', 'inverse_only']:
        print(f"❌ 無效的 HYBRID_TRAIN_PHASE: {TRAIN_PHASE}")
        print("應該是: 'both', 'surrogate_only', 或 'inverse_only'")
        sys.exit(1)
    
    TRAIN_SURROGATE = TRAIN_PHASE in ['both', 'surrogate_only']
    TRAIN_INVERSE = TRAIN_PHASE in ['both', 'inverse_only']
    
    print(f"🔧 訓練配置: TRAIN_PHASE = {TRAIN_PHASE}")
    print(f"   Phase 1 (Surrogate): {TRAIN_SURROGATE}")
    print(f"   Phase 2&3 (Inverse): {TRAIN_INVERSE}")
    print("-" * 50)
    
    # 定義模型路徑
    ckpt_dir = "checkpoint"
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    surrogate_path = os.path.join(ckpt_dir, "forward_surrogate.pth")
    best_surrogate_path = os.path.join(ckpt_dir, "forward_surrogate_best.pth")
    inverse_path = os.path.join(ckpt_dir, "inverse_cnn_hybrid.pth")
    best_inverse_path = os.path.join(ckpt_dir, "inverse_cnn_hybrid_best.pth")

    # --- 資料載入 ---
    dataset = TuringDataset(config_new.NPZ_PATH)
    dataset.scaler.to_device(config_new.DEVICE)
    
    train_size = int(len(dataset) * config_new.TRAIN_VAL_RATIO)
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
        wandb.init(project=config_new.WANDB_PROJECT, name=f"{config_new.WANDB_RUN_NAME}_Phase1-Surrogate")
        
        surrogate = ForwardSurrogate().to(config_new.DEVICE)
        optimizer_S = optim.AdamW(surrogate.parameters(), lr=1e-3, weight_decay=0.01)
        
        # 使用 Warm Up + Cosine Annealing LR Scheduler
        # 前10% epochs做warm up，後90%做cosine annealing
        total_epochs = 60
        warmup_epochs = int(0.1 * total_epochs)
        cosine_epochs = total_epochs - warmup_epochs
        
        # Warm up階段：從1e-6線性增加到初始LR
        warmup_scheduler_S = optim.lr_scheduler.LinearLR(
            optimizer_S, 
            start_factor=1e-6 / 1e-3,  # 從很小的lr開始
            end_factor=1.0,  # 增加到初始lr
            total_iters=warmup_epochs
        )
        
        # Cosine annealing階段
        cosine_scheduler_S = optim.lr_scheduler.CosineAnnealingLR(
            optimizer_S,
            T_max=cosine_epochs,
            eta_min=1e-6
        )
        
        # 組合調度器
        scheduler_S = optim.lr_scheduler.SequentialLR(
            optimizer_S,
            schedulers=[warmup_scheduler_S, cosine_scheduler_S],
            milestones=[warmup_epochs]
        )
        
        # 定義 Loss: 頻率對齊 + 顏色分佈對齊
        criterion_four = FourierLoss(radius=0.7).to(config_new.DEVICE)
        criterion_hist = HistogramLoss().to(config_new.DEVICE)
        
        # 初始化 Physical Loss (如果啟用)
        criterion_physics = None
        if config_new.USE_PHYSICS_PHASE1:
            criterion_physics = PhysicsLoss(dx=config_new.DX, s_diffusion=config_new.S_DIFFUSION).to(config_new.DEVICE)
            print("🔬 Phase 1 Physical Loss enabled")
        
        # 初始化最佳模型追蹤
        best_hist_loss = float('inf')
        
        for epoch in range(60):
            surrogate.train()
            loss_sum_four = 0
            loss_sum_hist = 0
            loss_sum_phy = 0
            
            for u_batch, v_batch, p_batch in tqdm(train_loader, desc=f"Surrogate Ep{epoch+1}"):
                u_batch = u_batch.to(config_new.DEVICE)
                v_batch = v_batch.to(config_new.DEVICE)
                p_batch = p_batch.to(config_new.DEVICE)
                
                optimizer_S.zero_grad()
                recon = surrogate(p_batch)
                
                # 計算兩個 Loss
                loss_f = criterion_four(recon, u_batch)
                loss_h = criterion_hist(recon, u_batch)
                
                # 計算 Physical Loss (如果啟用)
                loss_p = torch.tensor(0.0).to(config_new.DEVICE)
                if criterion_physics is not None:
                    # 使用生成的u和真實的v來計算physical loss
                    loss_p = criterion_physics(recon, v_batch, p_batch)
                
                # 總 Loss
                loss = loss_f + loss_h
                if criterion_physics is not None:
                    loss = loss + 0.1 * loss_p  # 使用小的權重
                
                loss.backward()
                optimizer_S.step()
                
                loss_sum_four += loss_f.item()
                loss_sum_hist += loss_h.item()
                loss_sum_phy += loss_p.item()
            
            # 驗證
            val_metrics = validate_surrogate(surrogate, val_loader, criterion_four, criterion_hist, criterion_physics)
            # Cosine Annealing LR: 每個epoch結束後自動衰減
            scheduler_S.step()
            
            # 準備log信息
            log_dict = {
                "train/hist_loss": loss_sum_hist/len(train_loader),
                "val/hist_loss": val_metrics['val_hist'],
                "val/four_loss": val_metrics['val_fourier'],
                "lr": optimizer_S.param_groups[0]['lr']
            }
            
            # 如果有physical loss，也加入log
            if criterion_physics is not None:
                log_dict["train/phy_loss"] = loss_sum_phy/len(train_loader)
                log_dict["val/phy_loss"] = val_metrics['val_physics']
            
            print(f"[Ep {epoch+1}] Train Hist={loss_sum_hist/len(train_loader):.4f} | Val Hist={val_metrics['val_hist']:.4f}", end="")
            if criterion_physics is not None:
                print(f" | Train Phy={loss_sum_phy/len(train_loader):.4f} | Val Phy={val_metrics['val_physics']:.4f}", end="")
            print()
            
            wandb.log(log_dict)
            
            # 檢查是否為最佳模型
            current_hist_loss = val_metrics['val_hist']
            if current_hist_loss < best_hist_loss:
                best_hist_loss = current_hist_loss
                torch.save(surrogate.state_dict(), best_surrogate_path)
                print(f"🏆 New best surrogate saved! Hist Loss: {best_hist_loss:.4f}")
            
        torch.save(surrogate.state_dict(), surrogate_path)
        print(f"✅ Final surrogate saved to {surrogate_path}")
        print(f"🏆 Best surrogate saved to {best_surrogate_path} (Hist Loss: {best_hist_loss:.4f})")
        
        # 如果只訓練 Surrogate，則結束
        if TRAIN_PHASE == 'surrogate_only':
            print("\n✅ 任務完成！只訓練了 Phase 1 (Surrogate)")
            return
        
        print("\n" + "="*50)
        print("現在開始 Phase 2&3: 訓練 Inverse CNN...")
        print("="*50 + "\n")
        
        # 結束 Phase 1 的 WandB run，開始新的 run 給 Phase 2&3
        wandb.finish()
        wandb.init(project=config_new.WANDB_PROJECT, name=f"{config_new.WANDB_RUN_NAME}_Phase2-3-Inverse")

    # ============================================================
    # 🟩 Phase 2 & 3: 訓練 Inverse CNN (逆向預測)
    # ============================================================
    
    # 初始化 WandB (如果是 inverse_only)
    if TRAIN_PHASE == 'inverse_only':
        wandb.init(project=config_new.WANDB_PROJECT, name=f"{config_new.WANDB_RUN_NAME}_Phase2-3-Inverse")
    
    # 只訓練 Inverse 時檢查 Surrogate 是否存在
    if not TRAIN_SURROGATE and not os.path.exists(surrogate_path):
        raise FileNotFoundError("❌ Surrogate 模型不存在，請先訓練 Phase 1 或使用 --phase both")
    
    print("🚀 [Phase 2 & 3] Training Inverse CNN...")
    
    # 1. 載入並凍結 Surrogate
    surrogate = ForwardSurrogate().to(config_new.DEVICE)
    surrogate.load_state_dict(torch.load(surrogate_path))
    surrogate.eval()
    for p in surrogate.parameters(): p.requires_grad = False
    print("🔒 Surrogate loaded & frozen.")
    
    # 2. 準備 Inverse Model
    model = ParameterNet().to(config_new.DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01) # 用小一點的 LR 求穩
    
    # 使用 Warm Up + Cosine Annealing LR Scheduler
    # 前10% epochs做warm up，後90%做cosine annealing
    total_epochs = config_new.EPOCHS
    warmup_epochs = int(0.1 * total_epochs)
    cosine_epochs = total_epochs - warmup_epochs
    
    # Warm up階段：從1e-6線性增加到初始LR
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1e-6 / 1e-4,  # 從很小的lr開始
        end_factor=1.0,  # 增加到初始lr
        total_iters=warmup_epochs
    )
    
    # Cosine annealing階段
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=1e-6
    )
    
    # 組合調度器
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )
    
    criterion_phy = FourierLoss(radius=0.7).to(config_new.DEVICE)
    loss_weights = config_new.LOSS_WEIGHTS
    
    # GradNorm for dynamic loss weighting (如果啟用)
    if config_new.USE_GRADNORM:
        gradnorm = GradNorm(num_tasks=2, alpha=1.5, device=config_new.DEVICE)
        initial_weights = torch.tensor([1.0, 0.01], device=config_new.DEVICE)  # lambda_phy初始值
        print("🔧 使用 GradNorm 動態權重調整 (Phase 2&3)")
    else:
        gradnorm = None
    
    # 初始化最佳模型追蹤
    best_nrmse = float('inf')
    
    for epoch in range(config_new.EPOCHS):
        model.train()
        total_sup = 0
        total_phy = 0
        
        # Curriculum: Gradual introduction of surrogate physics loss
        # 使用sigmoid函數實現平滑過渡，從epoch 20開始漸進增加到epoch 60達到最大值
        max_lambda = 0.01
        start_epoch = 20  # 開始引入的epoch
        end_epoch = 60    # 達到最大值的epoch
        
        if epoch < start_epoch:
            lambda_phy = 0.0
        elif epoch >= end_epoch:
            lambda_phy = max_lambda
        else:
            # Sigmoid-based gradual increase
            progress = (epoch - start_epoch) / (end_epoch - start_epoch)
            sigmoid_value = 1 / (1 + torch.exp(-10 * (progress - 0.5)))  # Sigmoid with steepness 10
            lambda_phy = max_lambda * sigmoid_value.item()
        
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
                
            if config_new.USE_GRADNORM and gradnorm is not None:
                # 使用 GradNorm 動態調整權重
                losses = [loss_sup, loss_fourier]
                current_weights = gradnorm.adjust_weights(model, losses, initial_weights)
                loss = current_weights[0] * loss_sup + current_weights[1] * loss_fourier
            else:
                # 使用固定權重 (Curriculum Learning)
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
        # Cosine Annealing LR: 每個epoch結束後自動衰減
        scheduler.step()
        
        avg_sup = total_sup / len(train_loader)
        
        print(f"[Ep {epoch+1}] Sup={avg_sup:.4f} | "
              f"Val NRMSE [δ={val_metrics['rmse_delta']:.2%} | Multi={val_metrics['nrmse_multi_dim']:.2%}]")
        
        wandb.log({
            "train/sup_loss": avg_sup,
            "train/phy_loss": total_phy / len(train_loader),
            "val/rmse_delta": val_metrics['rmse_delta'],
            "val/nrmse_multi_dim": val_metrics['nrmse_multi_dim'],
            "lr": optimizer.param_groups[0]['lr']
        })
        
        # 檢查是否為最佳模型
        current_nrmse = val_metrics['nrmse_multi_dim']
        if current_nrmse < best_nrmse:
            best_nrmse = current_nrmse
            torch.save(model.state_dict(), best_inverse_path)
            print(f"🏆 New best inverse model saved! NRMSE: {best_nrmse:.2%}")
        
        torch.save(model.state_dict(), inverse_path)

    print(f"🎉 Training Finished!")
    print(f"✅ Final inverse model saved to {inverse_path}")
    print(f"🏆 Best inverse model saved to {best_inverse_path} (NRMSE: {best_nrmse:.2%})")

if __name__ == "__main__":
    main()