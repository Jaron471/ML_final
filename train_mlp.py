import os
import sys
import argparse
import random
import re
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

# ============================================================
# Default Configuration
# ============================================================
DEFAULT_ARGS = {
    "use_loss": "physical",           # 'pure' (supervised only) or 'physical' (PINN)
    "phys_gradual": False,        # Gradual introduction of physical loss
    "data_fraction": 0.1,         # Data ablation fraction
    "lr": 1e-4,
}

# ============================================================
# Utils
# ============================================================
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

def validate_mlp(model, loader):
    model.eval()
    sum_squared = torch.zeros(4).to(config.DEVICE)
    count = 0
    
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for u_batch, _, p_batch in loader:
            u_batch = u_batch.to(config.DEVICE)
            p_batch = p_batch.to(config.DEVICE)
            
            p_pred = model(u_batch)
            
            diff = p_pred - p_batch
            sum_squared += torch.sum(diff ** 2, dim=0)
            count += u_batch.size(0)
            
            all_targets.append(p_batch.cpu())
            all_preds.append(p_pred.cpu())
            
    mse = sum_squared / count
    rmse = torch.sqrt(mse)
    
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    
    nrmse_multi_dim = calculate_multidim_nrmse(all_targets, all_preds)
    
    return {
        "rmse_avg": rmse.mean().item(),
        "rmse_delta": rmse[3].item(),
        "nrmse_multi_dim": nrmse_multi_dim
    }

# ============================================================
# Training
# ============================================================
def train_mlp(train_loader, val_loader, dataset, args):
    print(f"🚀 Training MLP with {args.use_loss} loss...")
    wandb.init(project=config.WANDB_PROJECT, name=f"MLP-{args.use_loss}", reinit=True)
    
    # Initialize Model
    model = MLPNet().to(config.DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    total_epochs = config.EPOCHS
    warmup_epochs = int(0.1 * total_epochs)
    cosine_epochs = total_epochs - warmup_epochs
    
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6 / args.lr, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    
    # Loss Functions
    loss_weights = config.LOSS_WEIGHTS
    criterion_phy_pinn = PhysicsLoss(dx=config.DX, s_diffusion=config.S_DIFFUSION).to(config.DEVICE)
    
    # GradNorm Setup
    gradnorm = None
    if config.USE_GRADNORM and args.use_loss == 'physical':
        gradnorm = GradNorm(num_tasks=2, alpha=1.5, device=config.DEVICE)
        initial_weights = torch.tensor([1.0, 0.1], device=config.DEVICE)
        print("🔧 Using GradNorm for dynamic loss weighting")

    best_nrmse = float('inf')
    ckpt_dir = "checkpoint"
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    
    # Naming
    pattern_regex = f"MLP_{args.use_loss}_(?P<num>\\d+)_.*\\.pth"
    current_num = get_next_version(ckpt_dir, pattern_regex)
    
    wandb_name = wandb.run.name if wandb.run.name else "unknown"
    base_filename = f"MLP_{args.use_loss}_{current_num}_frac{args.data_fraction}_{wandb_name}"
    
    best_model_path = os.path.join(ckpt_dir, f"{base_filename}_best.pth")
    final_model_path = os.path.join(ckpt_dir, f"{base_filename}_last.pth")
    
    print(f"📝 Models will be saved as: {base_filename}_{{best/last}}.pth")

    for epoch in range(total_epochs):
        model.train()
        total_sup = 0
        total_phy = 0
        
        lambda_val = 0.0
        max_lambda = 0.01 # Default weight for physical loss
        
        if args.use_loss == 'physical':
            if args.phys_gradual:
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
                lambda_val = max_lambda
        
        progress_bar = tqdm(train_loader, desc=f"Ep{epoch+1} (λ={lambda_val:.4f})", leave=False)
        
        for u_batch, v_batch, params_target in progress_bar:
            u_batch = u_batch.to(config.DEVICE)
            v_batch = v_batch.to(config.DEVICE)
            params_target = params_target.to(config.DEVICE)
            
            optimizer.zero_grad()
            preds_norm = model(u_batch)
            
            # Supervised Loss
            abs_diff = torch.abs(preds_norm - params_target)
            loss_sup = torch.mean(loss_weights * abs_diff)
            
            # Auxiliary Loss (Physics)
            loss_aux = torch.tensor(0.0).to(config.DEVICE)
            if lambda_val > 0 or (gradnorm is not None):
                if args.use_loss == 'physical':
                    params_pred_real = dataset.scaler.inverse_transform_tensor(preds_norm)
                    loss_aux = criterion_phy_pinn(u_batch, v_batch, params_pred_real)
            
            if gradnorm is not None and args.use_loss == 'physical':
                losses = [loss_sup, loss_aux]
                current_weights = gradnorm.adjust_weights(model, losses, initial_weights)
                loss = current_weights[0] * loss_sup + current_weights[1] * loss_aux
            else:
                loss = loss_sup + lambda_val * loss_aux
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_sup += loss_sup.item()
            total_phy += loss_aux.item()
            progress_bar.set_postfix({'Sup': loss_sup.item(), 'Aux': loss_aux.item()})
            
        val_metrics = validate_mlp(model, val_loader)
        scheduler.step()
        
        avg_sup = total_sup / len(train_loader)
        avg_phy = total_phy / len(train_loader)
        
        print(f"[Ep {epoch+1}] Sup={avg_sup:.4f} | Aux={avg_phy:.4f} | "
              f"Val NRMSE [δ={val_metrics['rmse_delta']:.2%} | Multi={val_metrics['nrmse_multi_dim']:.2%}]")
        
        wandb.log({
            "train/sup_loss": avg_sup,
            "train/aux_loss": avg_phy,
            "val/rmse_delta": val_metrics['rmse_delta'],
            "val/nrmse_multi_dim": val_metrics['nrmse_multi_dim'],
            "lr": optimizer.param_groups[0]['lr']
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

# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="MLP Training Pipeline")
    parser.add_argument('--use-loss', type=str, default=DEFAULT_ARGS['use_loss'], choices=['pure', 'physical'], help='Loss type: pure (supervised) or physical (PINN)')
    parser.add_argument('--phys-gradual', action='store_true', default=DEFAULT_ARGS['phys_gradual'], help='Use gradual introduction for physical loss')
    parser.add_argument('--data-fraction', type=float, default=DEFAULT_ARGS['data_fraction'], help='Fraction of Train+Val data to use')
    parser.add_argument('--lr', type=float, default=DEFAULT_ARGS['lr'], help='Learning rate')
    
    args = parser.parse_args()
    
    set_seed(42)
    wandb.login(key="c45f78d1fb5c9023cf8d9787e2d3828bd0f891e1")
    
    print("📂 Loading datasets...")
    
    train_set = TuringDataset(config.TRAIN_PATH)
    val_set = TuringDataset(config.VAL_PATH)

    dataset = train_set 
    dataset.scaler.to_device(config.DEVICE)

    if args.data_fraction < 1.0:
        total_train = len(train_set)
        used_size = int(total_train * args.data_fraction)
        unused_size = total_train - used_size
        
        train_set, _ = random_split(
            train_set, [used_size, unused_size], 
            generator=torch.Generator().manual_seed(42)
        )
        print(f"📉 Data Ablation: Using {args.data_fraction:.1%} of Training data ({used_size} samples)")

    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"📦 Data Loaded: Train={len(train_set)}, Val={len(val_set)}")
    
    train_mlp(train_loader, val_loader, dataset, args)

if __name__ == "__main__":
    main()
