import os
import sys
import random

# 1. 環境設定
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import wandb

# 引用模組 (記得去 model/model.py 加入 MLPNet)
from model import config_pinn as config
from model.dataset import TuringDataset
from model.model import ParameterNet, MLPNet # <--- 引入 MLPNet
from model.loss import PhysicsLoss, GradNorm

def set_seed(seed=42):
    """固定所有隨機因素，確保實驗可重現"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 根據設備類型設置對應的隨機種子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 如果有多張 GPU
        # 確保卷積算法是確定性的 (會稍微降低效能，但保證結果一致)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        # MPS 設備的隨機種子設置
        torch.mps.manual_seed(seed)
    
    print(f"🔒 Random seed set to {seed} (Device: {config.DEVICE})")

def main():
    # --- WandB 初始化 ---
    wandb.login(key="c45f78d1fb5c9023cf8d9787e2d3828bd0f891e1")
    set_seed(42)

    generator = torch.Generator().manual_seed(42)
    
    # 根據設定動態調整儲存檔名
    ckpt_dir = "checkpoint"
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)
        print(f"📂 Created directory: {ckpt_dir}")

    # 設定完整的儲存路徑
    filename = f"{config.MODEL_TYPE}_{'PINN' if config.USE_PHYSICS else 'Pure'}_newloss_{config.LOSS_TYPE}.pth"
    save_path = os.path.join(ckpt_dir, filename)
    
    # 設定最佳模型儲存路徑
    best_filename = f"{config.MODEL_TYPE}_{'PINN' if config.USE_PHYSICS else 'Pure'}_newloss_{config.LOSS_TYPE}_best.pth"
    best_save_path = os.path.join(ckpt_dir, best_filename)
    
    print(f"🧪 Experiment: {config.WANDB_RUN_NAME}")
    print(f"💾 Final model will be saved to: {save_path}")
    print(f"🏆 Best model will be saved to: {best_save_path}")

    print(f"🚀 Start Training using [{config.LOSS_TYPE}] Loss...")
    print(f"   Weights: {config.LOSS_WEIGHTS.cpu().numpy()}")

    wandb.init(
        project=config.WANDB_PROJECT,
        name=config.WANDB_RUN_NAME,
        config={
            "model_type": config.MODEL_TYPE,
            "use_physics": config.USE_PHYSICS,
            "lr": config.LEARNING_RATE,
            "epochs": config.EPOCHS
        }
    )
    
    # 2. 準備資料
    dataset = TuringDataset(config.NPZ_PATH)
    dataset.scaler.to_device(config.DEVICE)
    
    train_size = int(config.TRAIN_VAL_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset, 
        [train_size, val_size], 
        generator=generator  
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 3. 根據 Config 選擇模型
    if config.MODEL_TYPE == "CNN":
        model = ParameterNet().to(config.DEVICE)
    elif config.MODEL_TYPE == "MLP":
        model = MLPNet().to(config.DEVICE)
    else:
        raise ValueError("Unknown MODEL_TYPE in config")

    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=0.01)
    wandb.watch(model, log="all", log_freq=10)

    # 使用 Warm Up + Cosine Annealing LR Scheduler
    # 前10% epochs做warm up，後90%做cosine annealing
    warmup_epochs = int(0.1 * config.EPOCHS)
    cosine_epochs = config.EPOCHS - warmup_epochs
    
    # Warm up階段：從1e-6線性增加到初始LR
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=1e-6 / config.LEARNING_RATE,  # 從很小的lr開始
        end_factor=1.0,  # 增加到初始lr
        total_iters=warmup_epochs
    )
    
    # Cosine annealing階段
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
        eta_min=config.SCHEDULER_ETA_MIN
    )
    
    # 組合調度器
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs]
    )

    # Physics Loss (如果不用 Physics，這個物件還是可以建，只是不 call)
    criterion_physics = PhysicsLoss(dx=config.DX, s_diffusion=config.S_DIFFUSION).to(config.DEVICE)
    
    # GradNorm for dynamic loss weighting (如果啟用)
    if config.USE_GRADNORM and config.USE_PHYSICS:
        gradnorm = GradNorm(num_tasks=2, alpha=1.5, device=config.DEVICE)
        initial_weights = torch.tensor([1.0, config.LAMBDA_PHY], device=config.DEVICE)
        print("🔧 使用 GradNorm 動態權重調整")
    else:
        gradnorm = None
    
    # 初始化最佳模型追蹤
    best_nrmse = float('inf')
    
    # 4. 訓練迴圈
    for epoch in range(config.EPOCHS):
        model.train()
        total_data_loss = 0
        total_phy_loss = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.EPOCHS}", leave=False)
        
        for u_batch, v_batch, params_target in progress_bar:
            u_batch = u_batch.to(config.DEVICE)
            v_batch = v_batch.to(config.DEVICE)
            params_target = params_target.to(config.DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            params_pred_norm = model(u_batch)
            
            # --- Data Loss (MSE) ---
            if config.LOSS_TYPE == "MSE":
                # MSE = (pred - target)^2
                raw_diff = (params_pred_norm - params_target) ** 2
                
            elif config.LOSS_TYPE == "L1":
                # L1 = |pred - target|
                raw_diff = torch.abs(params_pred_norm - params_target)
            else:
                raise ValueError(f"❌ Unknown LOSS_TYPE: {config.LOSS_TYPE}")
            

            weighted_diff = raw_diff * config.LOSS_WEIGHTS
            loss_data = torch.mean(weighted_diff)
            
            # --- Physics Loss (條件執行) ---
            loss_phy = torch.tensor(0.0).to(config.DEVICE)
            
            if config.USE_PHYSICS:
                # 只有開關打開時才算這部分
                params_pred_real = dataset.scaler.inverse_transform_tensor(params_pred_norm)
                loss_phy = criterion_physics(u_batch, v_batch, params_pred_real)
                
                if config.USE_GRADNORM and gradnorm is not None:
                    # 使用 GradNorm 動態調整權重
                    losses = [loss_data, loss_phy]
                    current_weights = gradnorm.adjust_weights(model, losses, initial_weights)
                    loss = current_weights[0] * loss_data + current_weights[1] * loss_phy
                else:
                    # 使用固定權重
                    loss = loss_data + config.LAMBDA_PHY * loss_phy
            else:
                # 純資料驅動
                loss = loss_data
            
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_data_loss += loss_data.item()
            total_phy_loss += loss_phy.item()
            
            progress_bar.set_postfix({'Data': loss_data.item(), 'Phy': loss_phy.item()})

        # --- 驗證與 Log ---
        val_metrics, _ = validate_with_nrmse(model, val_loader, dataset.scaler)
        
        # Cosine Annealing LR: 每個epoch結束後自動衰減
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']
        log_dict = {
            "epoch": epoch + 1,
            "train/data_loss": total_data_loss / len(train_loader),
            "train/phy_loss": total_phy_loss / len(train_loader), # 如果沒開，這裡會是 0
            "val/nrmse_delta": val_metrics['nrmse_delta'],
            "val/nrmse_avg": val_metrics['nrmse_avg'],
            "val/nrmse_multi_dim": val_metrics['nrmse_multi_dim'],
            "train/learning_rate": current_lr
        }
        wandb.log(log_dict)
        print(f"Epoch {epoch+1}: "
              f"NRMSE [a={val_metrics['nrmse_a']:.2%} | "
              f"b={val_metrics['nrmse_b']:.2%} | "
              f"c={val_metrics['nrmse_c']:.2%} | "
              f"δ={val_metrics['nrmse_delta']:.2%}] "
              f"(Avg={val_metrics['nrmse_avg']:.2%} | Multi={val_metrics['nrmse_multi_dim']:.2%})")
        
        # 檢查是否為最佳模型
        current_nrmse = val_metrics['nrmse_avg']
        if current_nrmse < best_nrmse:
            best_nrmse = current_nrmse
            torch.save(model.state_dict(), best_save_path)
            print(f"🏆 New best model saved! NRMSE: {best_nrmse:.2%}")

    torch.save(model.state_dict(), save_path)
    wandb.save(save_path)
    print(f"✅ Final model saved to {save_path}")
    print(f"🏆 Best model saved to {best_save_path} (NRMSE: {best_nrmse:.2%})")
    wandb.finish()

def validate_with_nrmse(model, loader, scaler):
    model.eval()
    sum_squared_diff = torch.zeros(4).to(config.DEVICE)
    count = 0
    
    # 收集所有真實參數和預測參數用於多維 NRMSE
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for u_batch, _, params_target in loader:
            u_batch = u_batch.to(config.DEVICE)
            params_target = params_target.to(config.DEVICE)
            preds_norm = model(u_batch)
            diff = preds_norm - params_target
            sum_squared_diff += torch.sum(diff ** 2, dim=0)
            count += u_batch.size(0)
            
            # 收集數據
            all_targets.append(params_target.cpu())
            all_preds.append(preds_norm.cpu())
            
    mse = sum_squared_diff / count
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
        'nrmse_a': rmse[0].item(), 'nrmse_b': rmse[1].item(),
        'nrmse_c': rmse[2].item(), 'nrmse_delta': rmse[3].item(),
        'nrmse_avg': torch.mean(rmse).item(),
        'nrmse_multi_dim': nrmse_multi_dim
    }, None

if __name__ == "__main__":
    main()