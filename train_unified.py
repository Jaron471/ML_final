import os
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
import wandb

# Import modules
from model import config
from model.dataset import TuringDataset
from model.model import MLPNet
from model.loss import PhysicsLoss, GradNorm
from model.utils import calculate_multidim_nrmse, get_next_version

# ==========================================
# 使用範例:
# 1. CNN pure (25% data): 
#    python train_unified.py --model-arch cnn2 --use-loss pure --data-fraction 0.25 --nk 5 --np 5 --nf 5
#
# 2. CNN PINN (25% data): 
#    python train_unified.py --model-arch cnn2 --use-loss physical --data-fraction 0.25 --nk 5 --np 5 --nf 5 --phys-gradual
#
# 3. MLP pure (10% data):
#    python train_unified.py --model-arch mlp --use-loss pure --data-fraction 0.1
#
# 4. MLP PINN (10% data):
#    python train_unified.py --model-arch mlp --use-loss physical --data-fraction 0.1 --phys-gradual
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
# CNN Models
# ==========================================
class PaperMinimalCNN(nn.Module):
    """
    [cite: 1820-1822] 實作論文中的極簡 CNN (nk/np/nf)
    """
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        
        out_dim = input_size - np_size + 1
        self.flat_features = nk * out_dim * out_dim
        
        self.fc1 = nn.Linear(self.flat_features, nf)
        self.fc_out = nn.Linear(nf, 4) 

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return torch.sigmoid(x)
    
class PaperMinimalCNN2(nn.Module):
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN2, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=nk, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        
        out_dim = input_size - 2 * np_size + 2
        self.flat_features = nk * out_dim * out_dim
        
        self.fc1 = nn.Linear(self.flat_features, nf)
        self.fc_out = nn.Linear(nf, 4) 

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return torch.sigmoid(x)

class PaperMinimalCNN2MaxPool(nn.Module):
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN2MaxPool, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=nk, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        d1 = input_size - np_size + 1
        d2 = d1 // 2
        d3 = d2 - np_size + 1
        d4 = d3 // 2
        
        self.flat_features = nk * d4 * d4
        
        self.fc1 = nn.Linear(self.flat_features, nf)
        self.fc_out = nn.Linear(nf, 4) 

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return torch.sigmoid(x)


class PaperMinimalCNN2Stride(nn.Module):
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN2Stride, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=np_size, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=nk, out_channels=nk, kernel_size=np_size, stride=2, padding=0)
        
        # Calculate output dimension
        # After Conv1 (stride 2)
        d1 = (input_size - np_size) // 2 + 1
        # After Conv2 (stride 2)
        d2 = (d1 - np_size) // 2 + 1
        
        self.flat_features = nk * d2 * d2
        
        self.fc1 = nn.Linear(self.flat_features, nf)
        self.fc_out = nn.Linear(nf, 4) 

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return torch.sigmoid(x)

# ==========================================
# Validation Functions
# ==========================================
def validate_model(model, loader, is_cnn=False):
    """通用驗證函數，支援 CNN 和 MLP"""
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for u_batch, _, p_batch in loader:
            u_batch = u_batch.to(config.DEVICE)
            
            # CNN 只使用第一個 channel (u)
            if is_cnn:
                if u_batch.shape[1] > 1:
                    u_input = u_batch[:, 0:1, :, :]
                else:
                    u_input = u_batch
            else:
                u_input = u_batch
                
            p_pred = model(u_input)
            all_targets.append(p_batch.cpu())
            all_preds.append(p_pred.cpu())
    
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    
    nrmse = calculate_multidim_nrmse(all_targets, all_preds)
    
    return {
        "nrmse_multi_dim": nrmse
    }


# ==========================================
# Unified Training Function
# ==========================================
def train(args):
    # WandB 初始化
    model_name = f"{args.model_arch.upper()}" if args.model_arch == 'mlp' else f"CNN-{args.model_arch}"
    wandb_name = f"{model_name}-{args.use_loss}-frac{args.data_fraction}"
    if args.model_arch != 'mlp':
        wandb_name += f"-nk{args.nk}_np{args.np}_nf{args.nf}"
    
    wandb.init(project=config.WANDB_PROJECT, name=wandb_name, reinit=True)
    
    print(f"🚀 Training {model_name} with {args.use_loss} Loss")
    if args.model_arch != 'mlp':
        print(f"   Architecture: nk={args.nk}, np={args.np}, nf={args.nf}")
    print(f"   Data Fraction: {args.data_fraction*100}%")
    print(f"   Learning Rate: {args.lr}")
    
    # 1. Data Setup
    train_set = TuringDataset(config.TRAIN_PATH)
    val_set = TuringDataset(config.VAL_PATH)
    
    dataset = train_set
    dataset.scaler.to_device(config.DEVICE)
    
    # Data Ablation
    if args.data_fraction < 1.0:
        total_train = len(train_set)
        used_size = int(total_train * args.data_fraction)
        unused_size = total_train - used_size
        
        train_set, _ = random_split(
            train_set, [used_size, unused_size], 
            generator=torch.Generator().manual_seed(42)
        )
        
        total_val = len(val_set)
        used_val_size = int(total_val * args.data_fraction)
        unused_val_size = total_val - used_val_size

        val_set, _ = random_split(
            val_set, [used_val_size, unused_val_size],
            generator=torch.Generator().manual_seed(42)
        )

        print(f"📉 Data Ablation ({args.data_fraction:.1%}):")
        print(f"   Train: {total_train} -> {len(train_set)}")
        print(f"   Val  : {total_val} -> {len(val_set)}")
    
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 2. Model Setup
    is_cnn = args.model_arch in ['cnn1', 'cnn2', 'cnn2pool', 'cnn2stride']
    
    if args.model_arch == 'mlp':
        model = MLPNet().to(config.DEVICE)
    elif args.model_arch == 'cnn1':
        model = PaperMinimalCNN(nk=args.nk, np_size=args.np, nf=args.nf, input_size=128).to(config.DEVICE)
    elif args.model_arch == 'cnn2':
        model = PaperMinimalCNN2(nk=args.nk, np_size=args.np, nf=args.nf, input_size=128).to(config.DEVICE)
    elif args.model_arch == 'cnn2pool':
        model = PaperMinimalCNN2MaxPool(nk=args.nk, np_size=args.np, nf=args.nf, input_size=128).to(config.DEVICE)
    elif args.model_arch == 'cnn2stride':
        model = PaperMinimalCNN2Stride(nk=args.nk, np_size=args.np, nf=args.nf, input_size=128).to(config.DEVICE)
    else:
        raise ValueError(f"Unknown model architecture: {args.model_arch}")
    
    # 3. Optimizer Setup
    if args.model_arch == 'mlp':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    else:
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. Scheduler Setup
    total_epochs = config.EPOCHS
    warmup_epochs = int(0.1 * total_epochs)
    cosine_epochs = total_epochs - warmup_epochs
    
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6/args.lr, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    
    # 5. Loss Setup
    criterion_mse = nn.MSELoss()
    criterion_physics = PhysicsLoss(dx=config.DX, s_diffusion=config.S_DIFFUSION).to(config.DEVICE)
    
    # GradNorm (僅 MLP 使用)
    gradnorm = None
    if args.model_arch == 'mlp' and config.USE_GRADNORM and args.use_loss == 'physical':
        gradnorm = GradNorm(num_tasks=2, alpha=1.5, device=config.DEVICE)
        initial_weights = torch.tensor([1.0, 0.1], device=config.DEVICE)
        print("🔧 Using GradNorm for dynamic loss weighting")
    
    # MLP 的 loss weights
    loss_weights = config.LOSS_WEIGHTS if args.model_arch == 'mlp' else None
    
    # 6. Save Setup
    ckpt_dir = "paper_checkpoints"
    if not os.path.exists(ckpt_dir): 
        os.makedirs(ckpt_dir)
    
    pattern_regex = f"{args.model_arch.upper()}_{args.use_loss}_(?P<num>\\d+)_.*\\.pth"
    current_num = get_next_version(ckpt_dir, pattern_regex)
    
    base_filename = f"{args.model_arch.upper()}_{args.use_loss}_{current_num}_frac{args.data_fraction}"
    if args.model_arch != 'mlp':
        base_filename += f"_nk{args.nk}"
    
    best_model_path = os.path.join(ckpt_dir, f"{base_filename}_best.pth")
    final_model_path = os.path.join(ckpt_dir, f"{base_filename}_last.pth")
    
    print(f"📝 Models will be saved as: {base_filename}_{{best/last}}.pth")
    
    best_nrmse = float('inf')
    
    # ==========================
    # Training Loop
    # ==========================
    for epoch in range(total_epochs):
        model.train()
        total_sup = 0
        total_phy = 0
        
        # Lambda scheduling for physics loss
        lambda_val = 0.0
        max_lambda = args.lambda_phy
        
        if args.use_loss == 'physical':
            if args.phys_gradual:
                if args.model_arch == 'mlp':
                    # MLP: sigmoid warmup from epoch 20 to 60
                    start_epoch = 20
                    end_epoch = 60
                    if epoch < start_epoch:
                        lambda_val = 0.0
                    elif epoch >= end_epoch:
                        lambda_val = max_lambda
                    else:
                        progress = (epoch - start_epoch) / (end_epoch - start_epoch)
                        sigmoid_value = 1 / (1 + np.exp(-10 * (progress - 0.5)))
                        lambda_val = max_lambda * sigmoid_value
                else:
                    # CNN: simple step at epoch 20
                    if epoch < 20:
                        lambda_val = 0.0
                    else:
                        lambda_val = max_lambda
            else:
                lambda_val = max_lambda
        
        progress_bar = tqdm(train_loader, desc=f"Ep{epoch+1} (λ={lambda_val:.4f})", leave=False)
        
        for u_batch, v_batch, params_target in progress_bar:
            u_batch = u_batch.to(config.DEVICE)
            v_batch = v_batch.to(config.DEVICE)
            params_target = params_target.to(config.DEVICE)
            
            # 處理輸入 (CNN 只用第一個 channel)
            if is_cnn:
                if u_batch.shape[1] > 1:
                    u_input = u_batch[:, 0:1, :, :]
                else:
                    u_input = u_batch
            else:
                u_input = u_batch
            
            optimizer.zero_grad()
            
            # Forward
            preds_norm = model(u_input)
            
            # --- Loss 1: Supervised ---
            if args.model_arch == 'mlp' and loss_weights is not None:
                # MLP: weighted L1
                abs_diff = torch.abs(preds_norm - params_target)
                loss_sup = torch.mean(loss_weights * abs_diff)
            else:
                # CNN: MSE
                loss_sup = criterion_mse(preds_norm, params_target)
            
            # --- Loss 2: Physics ---
            loss_phy = torch.tensor(0.0).to(config.DEVICE)
            
            if (lambda_val > 0 or gradnorm is not None) and args.use_loss == 'physical':
                params_real = dataset.scaler.inverse_transform_tensor(preds_norm)
                
                # 準備物理 Loss 的輸入
                if is_cnn:
                    if u_batch.shape[1] > 1:
                        u_real = u_batch[:, 0:1, :, :]
                        v_real = u_batch[:, 1:2, :, :]
                    else:
                        u_real = u_batch
                        v_real = v_batch
                else:
                    u_real = u_batch
                    v_real = v_batch
                
                loss_phy = criterion_physics(u_real, v_real, params_real)
            
            # --- Total Loss ---
            if gradnorm is not None and args.use_loss == 'physical':
                losses = [loss_sup, loss_phy]
                current_weights = gradnorm.adjust_weights(model, losses, initial_weights)
                loss = current_weights[0] * loss_sup + current_weights[1] * loss_phy
            else:
                loss = loss_sup + lambda_val * loss_phy
            
            loss.backward()
            
            # Gradient clipping (僅 MLP)
            if args.model_arch == 'mlp':
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_sup += loss_sup.item()
            total_phy += loss_phy.item()
            progress_bar.set_postfix({'Sup': loss_sup.item(), 'Phy': loss_phy.item()})
        
        # Update Scheduler
        scheduler.step()
        
        # ==========================
        # Validation
        # ==========================
        val_metrics = validate_model(model, val_loader, is_cnn=is_cnn)
        
        avg_sup = total_sup / len(train_loader)
        avg_phy = total_phy / len(train_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"[Ep {epoch+1}] Sup={avg_sup:.5f} | Phy={avg_phy:.5f} | "
              f"Val NRMSE={val_metrics['nrmse_multi_dim']:.2%} | LR={current_lr:.2e}")
        
        wandb.log({
            "train/sup_loss": avg_sup,
            "train/phy_loss": avg_phy,
            "val/nrmse": val_metrics['nrmse_multi_dim'],
            "lr": current_lr
        })
        
        if val_metrics['nrmse_multi_dim'] < best_nrmse:
            best_nrmse = val_metrics['nrmse_multi_dim']
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 New best model saved! NRMSE: {best_nrmse:.2%}")
        
        torch.save(model.state_dict(), final_model_path)
    
    print(f"🎉 Training Finished!")
    print(f"✅ Final model saved to {final_model_path}")
    print(f"🏆 Best model saved to {best_model_path} (NRMSE: {best_nrmse:.2%})")
    wandb.finish()


# ==========================================
# Main
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Unified Training Pipeline for MLP and CNN")
    
    # Model Architecture
    parser.add_argument('--model-arch', type=str, default='mlp', 
                        choices=['mlp', 'cnn1', 'cnn2', 'cnn2pool', 'cnn2stride'],
                        help='Model architecture: mlp, cnn1 (1-layer), cnn2 (2-layer), cnn2pool (2-layer+maxpool), cnn2stride (2-layer+stride)')
    
    # CNN-specific parameters
    parser.add_argument('--nk', type=int, default=5, help='Number of kernels (CNN only)')
    parser.add_argument('--np', type=int, default=5, help='Kernel size (CNN only)')
    parser.add_argument('--nf', type=int, default=5, help='Hidden neurons (CNN only)')
    
    # Training parameters
    parser.add_argument('--use-loss', type=str, default='physical', 
                        choices=['pure', 'physical'],
                        help='Loss type: pure (supervised) or physical (PINN)')
    parser.add_argument('--lambda-phy', type=float, default=0.01, help='Weight for physics loss')
    parser.add_argument('--phys-gradual', action='store_true', default=True,
                        help='Use gradual introduction for physics loss')
    
    # Data parameters
    parser.add_argument('--data-fraction', type=float, default=0.1,
                        help='Fraction of training data to use (e.g., 0.05 for 5%%)')
    
    # Optimizer parameters
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 1e-3 for CNN, 1e-4 for MLP)')
    
    args = parser.parse_args()
    
    # Set default learning rate based on model architecture
    if args.lr is None:
        args.lr = 1e-3 if args.model_arch in ['cnn1', 'cnn2', 'cnn2pool', 'cnn2stride'] else 1e-4
    
    set_seed(42)
    
    # WandB login (如果需要)
    # wandb.login(key="your_key_here")
    
    print(f"📂 Starting unified training pipeline...")
    train(args)


if __name__ == "__main__":
    main()
