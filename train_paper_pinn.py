import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split # 引入 random_split
from tqdm import tqdm
import numpy as np
import wandb

# Import modules
from model import config
from model.dataset import TuringDataset
# 這裡引用最穩定的 PhysicsLoss (3x3)
from model.loss import PhysicsLoss 
from model.utils import calculate_multidim_nrmse, get_next_version

# ==========================================
# 使用範例:
# 1. CNN pure (5% data): 
#    python train_paper_pinn.py --use-loss pure --data-fraction 0.05 --nk 5 --np 5 --nf 5
#
# 2. CNN PINN (5% data): 
#    python train_paper_pinn.py --use-loss physical --data-fraction 0.05 --nk 5 --np 5 --nf 5 --phys-gradual
# ==========================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    print(f"🔒 Random seed set to {seed} (Device: {config.DEVICE})")


# ==========================================
# 1. 定義論文中的極簡 CNN 架構 (保持不變)
# ==========================================
class PaperMinimalCNN(nn.Module):
    """
    [cite: 1820-1822] 實作論文中的極簡 CNN (nk/np/nf)
    """
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN, self).__init__()
        
        # [cite: 1936] 輸入固定為 1 Channel (只看 u)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        
        out_dim = input_size - np_size + 1
        self.flat_features = nk * out_dim * out_dim
        
        self.fc1 = nn.Linear(self.flat_features, nf)
        self.fc_out = nn.Linear(nf, 4) 

    def forward(self, x):
        # x shape: (Batch, 1, 128, 128)
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1) # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return torch.sigmoid(x) # 輸出歸一化參數 [0, 1]

# ==========================================
# 2. 訓練流程 (加入 PINN + Scheduler + Data Fraction)
# ==========================================
def train(args):
    # WandB 名稱加入 fraction 資訊，方便識別
    wandb.init(project=config.WANDB_PROJECT, name=f"Paper-PINN-({args.nk}_{args.np}_{args.nf})-{args.use_loss}-frac{args.data_fraction}", reinit=True)
    
    print(f"🚀 Training Paper-Style CNN with {args.use_loss} Loss")
    print(f"   Architecture: nk={args.nk}, np={args.np}, nf={args.nf}")
    print(f"   Data Fraction: {args.data_fraction*100}%")
    
    # 1. Data Setup
    train_set = TuringDataset(config.TRAIN_PATH)
    val_set = TuringDataset(config.VAL_PATH)
    
    dataset = train_set
    dataset.scaler.to_device(config.DEVICE) # 確保 Scaler 在 GPU，計算 Physics Loss 需要
    
    # 🔥 Data Ablation: 根據 data-fraction 切分訓練集
    if args.data_fraction < 1.0:
        total_train = len(train_set)
        used_size = int(total_train * args.data_fraction)
        unused_size = total_train - used_size
        
        # 使用固定種子切分，確保 Pure 和 PINN 用的是同一組 "5%" 數據
        train_set, _ = random_split(
            train_set, [used_size, unused_size], 
            generator=torch.Generator().manual_seed(42)
        )
        total_val = len(val_set)
        used_val_size = int(total_val * args.data_fraction)
        unused_val_size = total_val - used_val_size

        val_set, _ = random_split(
            val_set, [used_val_size, unused_val_size],
            generator=torch.Generator().manual_seed(42) # 用相同種子確保一致性
        )

        print(f"📉 Data Ablation ({args.data_fraction:.1%}):")
        print(f"   Train: {total_train} -> {len(train_set)}")
        print(f"   Val  : {total_val} -> {len(val_set)}")

    
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 2. Model Setup
    model = PaperMinimalCNN(nk=args.nk, np_size=args.np, nf=args.nf, input_size=128).to(config.DEVICE)
    
    # [cite: 1932] 論文使用 Adam
    optimizer = optim.Adam(model.parameters(), lr=1e-3) 
    
    # Scheduler 設定
    total_epochs = config.EPOCHS
    warmup_epochs = int(0.1 * total_epochs) # 前 10% 用來熱身
    cosine_epochs = total_epochs - warmup_epochs
    
    # 1. Warmup
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6/1e-3, end_factor=1.0, total_iters=warmup_epochs
    )
    # 2. Cosine Annealing
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=1e-6
    )
    # 3. 串聯
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    
    # 3. Loss Setup
    criterion_mse = nn.MSELoss()
    
    # 🔥 初始化物理 Loss
    criterion_physics = PhysicsLoss(dx=config.DX, s_diffusion=config.S_DIFFUSION).to(config.DEVICE)
    
    # 4. Save Setup
    ckpt_dir = "paper_checkpoints"
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    # 檔名加入 fraction
    current_num = get_next_version(ckpt_dir, r"PaperPINN_.*_(?P<num>\d+)_.*\.pth")
    base_filename = f"PaperPINN_{args.use_loss}_nk{args.nk}_frac{args.data_fraction}_{current_num}"
    best_model_path = os.path.join(ckpt_dir, f"{base_filename}_best.pth")
    
    best_nrmse = float('inf')
    
    # ==========================
    # Training Loop
    # ==========================
    for epoch in range(config.EPOCHS):
        model.train()
        total_sup = 0
        total_phy = 0
        
        lambda_val = args.lambda_phy

        if args.phys_gradual and epoch < 20:
            lambda_val = 0.0

        progress_bar = tqdm(train_loader, desc=f"Ep {epoch+1} (λ={lambda_val})", leave=False)
        
        for u_batch, v_batch, params_target in progress_bar:
            u_batch = u_batch.to(config.DEVICE)
            v_batch = v_batch.to(config.DEVICE) 
            params_target = params_target.to(config.DEVICE)
            
            # --- 處理輸入 ---
            # 論文模型只看 Channel 0 (u)
            if u_batch.shape[1] > 1:
                u_input = u_batch[:, 0:1, :, :] 
            else:
                u_input = u_batch
            
            optimizer.zero_grad()
            
            # Forward
            preds_norm = model(u_input)
            
            # --- Loss 1: Supervised (MSE) ---
            loss_sup = criterion_mse(preds_norm, params_target)
            
            # --- Loss 2: Physics (PINN) ---
            loss_phy = torch.tensor(0.0).to(config.DEVICE)
            
            if lambda_val > 0 and args.use_loss == 'physical':
                params_real = dataset.scaler.inverse_transform_tensor(preds_norm)
                
                # 準備物理 Loss 的輸入 (需要 u 和 v)
                if u_batch.shape[1] > 1:
                    u_real = u_batch[:, 0:1, :, :]
                    v_real = u_batch[:, 1:2, :, :] 
                else:
                    # 當 Input 只有 1 channel 時，v 從 v_batch 拿
                    u_real = u_batch
                    v_real = v_batch 
                
                loss_phy = criterion_physics(u_real, v_real, params_real)
            
            # Total Loss
            loss = loss_sup + lambda_val * loss_phy
            
            loss.backward()
            optimizer.step()
            
            total_sup += loss_sup.item()
            total_phy += loss_phy.item()
        
        # 更新 Scheduler
        scheduler.step()
            
        # ==========================
        # Validation
        # ==========================
        model.eval()
        all_targets = []
        all_preds = []
        with torch.no_grad():
            for u_batch, _, p_batch in val_loader:
                u_batch = u_batch.to(config.DEVICE)
                
                if u_batch.shape[1] > 1:
                    u_input = u_batch[:, 0:1, :, :]
                else:
                    u_input = u_batch
                    
                p_pred = model(u_input)
                all_targets.append(p_batch.cpu())
                all_preds.append(p_pred.cpu())
        
        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_preds = torch.cat(all_preds, dim=0).numpy()
        nrmse = calculate_multidim_nrmse(all_targets, all_preds)
        
        avg_sup = total_sup / len(train_loader)
        avg_phy = total_phy / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Ep {epoch+1} | Sup: {avg_sup:.5f} | Phy: {avg_phy:.5f} | Val NRMSE: {nrmse:.2%}")
        
        wandb.log({
            "train/mse": avg_sup, 
            "train/physics": avg_phy, 
            "val/nrmse": nrmse,
            "lr": current_lr
        })
        
        if nrmse < best_nrmse:
            best_nrmse = nrmse
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 Saved Best: {nrmse:.2%}")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 模型參數
    parser.add_argument('--nk', type=int, default=5, help='Number of kernels')
    parser.add_argument('--np', type=int, default=5, help='Kernel size')
    parser.add_argument('--nf', type=int, default=5, help='Hidden neurons')
    
    # PINN 參數
    parser.add_argument('--use-loss', type=str, default='physical', choices=['pure', 'physical'])
    parser.add_argument('--lambda-phy', type=float, default=0.01, help='Weight for physics loss')
    parser.add_argument('--phys-gradual', action='store_true', default=True, help='Use warmup for physics loss')
    
    # 🔥 新增：Data Fraction
    parser.add_argument('--data-fraction', type=float, default=1.0, help='Fraction of training data to use (e.g., 0.05)')
    
    set_seed(42)
    args = parser.parse_args()
    train(args)


### 使用方式：

### 1.  **Pure CNN (5% 數據):**
###    python train_paper_pinn.py --use-loss pure --data-fraction 0.6 --nk 5 --np 7 --nf 5
### 2.  **PINN (5% 數據):**
###    python train_paper_pinn.py --use-loss physical --data-fraction 0.05 --nk 5 --np 5 --nf 5 --phys-gradual
