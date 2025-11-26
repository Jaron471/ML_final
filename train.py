import os
import sys

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
from model import config
from model.dataset import TuringDataset
from model.model import ParameterNet, MLPNet # <--- 引入 MLPNet
from model.loss import PhysicsLoss

def main():
    # --- WandB 初始化 ---
    wandb.login(key="d969eaa6886920a565de70b8ca7ce8c9b13f6ddf")
    
    # 根據設定動態調整儲存檔名
    save_name = f"{config.MODEL_TYPE}_{'PINN' if config.USE_PHYSICS else 'Pure'}.pth"
    print(f"🧪 Experiment: {config.WANDB_RUN_NAME}")
    print(f"📂 Model will be saved to: {save_name}")

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
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 3. 根據 Config 選擇模型
    if config.MODEL_TYPE == "CNN":
        model = ParameterNet().to(config.DEVICE)
    elif config.MODEL_TYPE == "MLP":
        model = MLPNet().to(config.DEVICE)
    else:
        raise ValueError("Unknown MODEL_TYPE in config")
        
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    wandb.watch(model, log="all", log_freq=10)

    # Physics Loss (如果不用 Physics，這個物件還是可以建，只是不 call)
    criterion_physics = PhysicsLoss().to(config.DEVICE)
    
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
            diff = (params_pred_norm - params_target) ** 2
            loss_data = torch.mean(config.LOSS_WEIGHTS * diff)
            
            # --- Physics Loss (條件執行) ---
            loss_phy = torch.tensor(0.0).to(config.DEVICE)
            
            if config.USE_PHYSICS:
                # 只有開關打開時才算這部分
                params_pred_real = dataset.scaler.inverse_transform_tensor(params_pred_norm)
                loss_phy = criterion_physics(u_batch, v_batch, params_pred_real)
                loss = loss_data + config.LAMBDA_PHY * loss_phy
            else:
                # 純資料驅動
                loss = loss_data
            
            loss.backward()
            optimizer.step()
            
            total_data_loss += loss_data.item()
            total_phy_loss += loss_phy.item()
            
            progress_bar.set_postfix({'Data': loss_data.item(), 'Phy': loss_phy.item()})

        # --- 驗證與 Log ---
        val_metrics, _ = validate_with_nrmse(model, val_loader, dataset.scaler)
        
        log_dict = {
            "epoch": epoch + 1,
            "train/data_loss": total_data_loss / len(train_loader),
            "train/phy_loss": total_phy_loss / len(train_loader), # 如果沒開，這裡會是 0
            "val/nrmse_delta": val_metrics['nrmse_delta'],
            "val/nrmse_avg": val_metrics['nrmse_avg']
        }
        wandb.log(log_dict)
        print(f"Epoch {epoch+1}: Val NRMSE(δ)={val_metrics['nrmse_delta']:.2%}")

    torch.save(model.state_dict(), save_name)
    wandb.finish()

def validate_with_nrmse(model, loader, scaler):
    model.eval()
    sum_squared_diff = torch.zeros(4).to(config.DEVICE)
    count = 0
    with torch.no_grad():
        for u_batch, _, params_target in loader:
            u_batch = u_batch.to(config.DEVICE)
            params_target = params_target.to(config.DEVICE)
            preds_norm = model(u_batch)
            diff = preds_norm - params_target
            sum_squared_diff += torch.sum(diff ** 2, dim=0)
            count += u_batch.size(0)
            
    mse = sum_squared_diff / count
    rmse = torch.sqrt(mse)
    return {
        'nrmse_a': rmse[0].item(), 'nrmse_b': rmse[1].item(),
        'nrmse_c': rmse[2].item(), 'nrmse_delta': rmse[3].item(),
        'nrmse_avg': torch.mean(rmse).item()
    }, None

if __name__ == "__main__":
    main()