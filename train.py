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
from model import config  # Unified config
from model.dataset import TuringDataset
from model.model import ParameterNet, MLPNet, ForwardSurrogate
from model.loss import GradNorm, PhysicsLoss, FourierLoss, HistogramLoss, TopKPhysicsLoss
from model.utils import calculate_multidim_nrmse, get_next_version, find_model_path

# ============================================================
# Default Configuration (可在此修改預設值)
# ============================================================
DEFAULT_ARGS = {
    "pretrain": False,            # 是否訓練 Phase 1 (Surrogate)
    "pretrain_physloss": False,   # Phase 1 是否使用 Physical Loss
    "train_model": "CNN",         # Phase 2 模型架構: 'CNN' or 'MLP'
    "use_loss": "physical",           # Phase 2 Loss 類型: 'pure', 'physical', 'surrogate'
    "phys_gradual": False,         # 是否使用漸進式 Loss 引入
    "data_fraction": 0.1,         # 數據消融測試: 使用 Train+Val 數據的比例 (0.0 ~ 1.0)
    "surrogate_num": None         # 指定使用的 Surrogate 模型編號 (None = 最新)
}

# ============================================================
# 0. Setup & Utils
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

def validate_surrogate(model, loader, criterion_four, criterion_hist, criterion_physics=None):
    model.eval()
    total_four = 0.0
    total_hist = 0.0
    total_phy = 0.0
    
    with torch.no_grad():
        for u_batch, v_batch, p_batch in loader:
            u_batch = u_batch.to(config.DEVICE)
            v_batch = v_batch.to(config.DEVICE)
            p_batch = p_batch.to(config.DEVICE)
            
            recon = model(p_batch)
            loss_f = criterion_four(recon, u_batch)
            loss_h = criterion_hist(recon, u_batch)
            
            loss_p = torch.tensor(0.0).to(config.DEVICE)
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
    sum_squared = torch.zeros(4).to(config.DEVICE)
    count = 0
    phy_total = 0.0
    
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
            
            if surrogate and criterion_phy:
                recon = surrogate(p_pred)
                loss_phy = criterion_phy(recon, u_batch)
                phy_total += loss_phy.item()
                
    mse = sum_squared / count
    rmse = torch.sqrt(mse)
    
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    
    nrmse_multi_dim = calculate_multidim_nrmse(all_targets, all_preds)
    
    return {
        "rmse_avg": rmse.mean().item(),
        "rmse_delta": rmse[3].item(),
        "nrmse_multi_dim": nrmse_multi_dim,
        "val_phy": phy_total / len(loader) if surrogate else 0.0
    }

# ============================================================
# 1. Phase 1 Training
# ============================================================
def train_phase1(train_loader, val_loader, use_phys_loss, data_fraction=1.0):
    print("🚀 [Phase 1] Training Surrogate (Fourier + Histogram)...")
    wandb.init(project=config.WANDB_PROJECT, name=f"Phase1-Surrogate", reinit=True)
    
    # Determine Version Number
    surrogate_dir = os.path.join("checkpoint", "surrogate")
    if not os.path.exists(surrogate_dir): os.makedirs(surrogate_dir)
    
    # Pattern: Surrogate_{pure/phys}_{num}_{wandb_name}.pth
    loss_tag = "phys" if use_phys_loss else "pure"
    pattern_regex = r"Surrogate_" + loss_tag + r"_(?P<num>\d+)_.*\.pth"
    current_num = get_next_version(surrogate_dir, pattern_regex)
    
    wandb_name = wandb.run.name if wandb.run.name else "unknown"
    
    # --- 修改點 1: 在檔名加入 frac 資訊 ---
    model_filename = f"Surrogate_{loss_tag}_{current_num}_frac{data_fraction}_{wandb_name}.pth"
    final_model_path = os.path.join(surrogate_dir, model_filename)
    
    print(f"📝 Model will be saved as: {model_filename}")
    
    surrogate = ForwardSurrogate().to(config.DEVICE)
    optimizer_S = optim.AdamW(surrogate.parameters(), lr=1e-3, weight_decay=0.01)
    
    total_epochs = 60
    warmup_epochs = int(0.1 * total_epochs)
    cosine_epochs = total_epochs - warmup_epochs
    
    warmup_scheduler_S = optim.lr_scheduler.LinearLR(
        optimizer_S, start_factor=1e-6 / 1e-3, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler_S = optim.lr_scheduler.CosineAnnealingLR(
        optimizer_S, T_max=cosine_epochs, eta_min=1e-6
    )
    scheduler_S = optim.lr_scheduler.SequentialLR(
        optimizer_S, schedulers=[warmup_scheduler_S, cosine_scheduler_S], milestones=[warmup_epochs]
    )
    
    criterion_four = FourierLoss(radius=0.7).to(config.DEVICE)
    criterion_hist = HistogramLoss().to(config.DEVICE)
    
    criterion_physics = None
    if use_phys_loss:
        criterion_physics = PhysicsLoss(dx=config.DX, s_diffusion=config.S_DIFFUSION).to(config.DEVICE)
        print("🔬 Phase 1 Physical Loss enabled")
    
    # GradNorm Setup for Phase 1
    gradnorm = None
    if config.USE_GRADNORM:
        num_tasks = 3 if use_phys_loss else 2
        gradnorm = GradNorm(num_tasks=num_tasks, alpha=1.5, device=config.DEVICE)
        # Initial weights: Fourier=1.0, Hist=1.0, Physics=0.1
        if use_phys_loss:
            initial_weights = torch.tensor([1.0, 1.0, 0.1], device=config.DEVICE)
        else:
            initial_weights = torch.tensor([1.0, 1.0], device=config.DEVICE)
        print(f"🔧 [Phase 1] Using GradNorm for {num_tasks} tasks")

    best_hist_loss = float('inf')
    
    for epoch in range(total_epochs):
        surrogate.train()
        loss_sum_four = 0
        loss_sum_hist = 0
        loss_sum_phy = 0
        
        for u_batch, v_batch, p_batch in tqdm(train_loader, desc=f"Surrogate Ep{epoch+1}"):
            u_batch = u_batch.to(config.DEVICE)
            v_batch = v_batch.to(config.DEVICE)
            p_batch = p_batch.to(config.DEVICE)
            
            optimizer_S.zero_grad()
            recon = surrogate(p_batch)
            
            loss_f = criterion_four(recon, u_batch)
            loss_h = criterion_hist(recon, u_batch)
            
            loss_p = torch.tensor(0.0).to(config.DEVICE)
            if criterion_physics is not None:
                loss_p = criterion_physics(recon, v_batch, p_batch)
            
            if gradnorm is not None:
                losses = [loss_f, loss_h]
                if criterion_physics is not None:
                    losses.append(loss_p)
                
                current_weights = gradnorm.adjust_weights(surrogate, losses, initial_weights)
                
                loss = 0
                for i, l in enumerate(losses):
                    loss += current_weights[i] * l
            else:
                loss = loss_f + loss_h
                if criterion_physics is not None:
                    loss = loss + 0.1 * loss_p
            
            loss.backward()
            optimizer_S.step()
            
            loss_sum_four += loss_f.item()
            loss_sum_hist += loss_h.item()
            loss_sum_phy += loss_p.item()
        
        val_metrics = validate_surrogate(surrogate, val_loader, criterion_four, criterion_hist, criterion_physics)
        scheduler_S.step()
        
        log_dict = {
            "train/hist_loss": loss_sum_hist/len(train_loader),
            "val/hist_loss": val_metrics['val_hist'],
            "val/four_loss": val_metrics['val_fourier'],
            "lr": optimizer_S.param_groups[0]['lr']
        }
        if criterion_physics is not None:
            log_dict["train/phy_loss"] = loss_sum_phy/len(train_loader)
            log_dict["val/phy_loss"] = val_metrics['val_physics']
        
        print(f"[Ep {epoch+1}] Train Hist={loss_sum_hist/len(train_loader):.4f} | Val Hist={val_metrics['val_hist']:.4f}", end="")
        if criterion_physics is not None:
            print(f" | Train Phy={loss_sum_phy/len(train_loader):.4f} | Val Phy={val_metrics['val_physics']:.4f}", end="")
        print()
        
        wandb.log(log_dict)
        
        if val_metrics['val_hist'] < best_hist_loss:
            best_hist_loss = val_metrics['val_hist']
            torch.save(surrogate.state_dict(), final_model_path)
            print(f"🏆 New best surrogate saved! Hist Loss: {best_hist_loss:.4f}")
            
    print(f"✅ Final surrogate saved to {final_model_path}")
    wandb.finish()
    return current_num # Return the version number for Phase 2 to use

# ============================================================
# 2. Phase 2 Training
# ============================================================
def train_phase2(train_loader, val_loader, dataset, args, pretrain_num=None):
    print(f"🚀 [Phase 2] Training Inverse Model ({args.train_model}) with {args.use_loss} loss...")
    wandb.init(project=config.WANDB_PROJECT, name=f"Phase2-{args.train_model}-{args.use_loss}", reinit=True)
    
    # Load Surrogate if needed
    surrogate = None
    surrogate_num_str = ""
    
    if args.use_loss == 'surrogate':
        ckpt_dir = os.path.join("checkpoint", "surrogate")
        
        # Determine which surrogate to use
        target_num = None
        if pretrain_num is not None:
            target_num = pretrain_num
            print(f"🔗 Using just-trained Surrogate model (Version {target_num})")
        elif args.surrogate_num is not None:
            target_num = int(args.surrogate_num)
            print(f"🔗 Using specified Surrogate model (Version {target_num})")
        else:
            print("🔗 Using latest Surrogate model")
            
        pattern_regex = r"Surrogate_.*_(?P<num>\d+)_.*\.pth"
        surrogate_path = find_model_path(ckpt_dir, pattern_regex, version=target_num)
        
        if surrogate_path is None:
            raise FileNotFoundError(f"❌ Surrogate model not found (Version: {target_num if target_num else 'Latest'}) in {ckpt_dir}")
            
        print(f"📂 Loading Surrogate from: {surrogate_path}")
        
        # Extract num for naming if we found latest
        if target_num is None:
            match = re.search(r"_(?P<num>\d+)_", os.path.basename(surrogate_path))
            if match:
                target_num = int(match.group('num'))
        
        surrogate_num_str = f"-{target_num}"
        
        surrogate = ForwardSurrogate().to(config.DEVICE)
        surrogate.load_state_dict(torch.load(surrogate_path))
        surrogate.eval()
        for p in surrogate.parameters(): p.requires_grad = False
        print("🔒 Surrogate loaded & frozen.")

    # Initialize Model
    if args.train_model == 'CNN':
        model = ParameterNet().to(config.DEVICE)
    elif args.train_model == 'MLP':
        model = MLPNet().to(config.DEVICE)
    else:
        raise ValueError(f"Unknown model type: {args.train_model}")

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    total_epochs = config.EPOCHS
    warmup_epochs = int(0.1 * total_epochs)
    cosine_epochs = total_epochs - warmup_epochs
    
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6 / 1e-4, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cosine_epochs, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    
    # Loss Functions
    loss_weights = config.LOSS_WEIGHTS
    criterion_phy_surrogate = FourierLoss(radius=0.7).to(config.DEVICE)
    criterion_phy_pinn = PhysicsLoss(dx=config.DX, s_diffusion=config.S_DIFFUSION).to(config.DEVICE)
    
    # GradNorm Setup
    gradnorm = None
    if config.USE_GRADNORM and args.use_loss in ['physical', 'surrogate']:
        gradnorm = GradNorm(num_tasks=2, alpha=1.5, device=config.DEVICE)
        init_lambda = 0.01 if args.use_loss == 'surrogate' else 0.1
        initial_weights = torch.tensor([1.0, init_lambda], device=config.DEVICE)
        print("🔧 Using GradNorm for dynamic loss weighting")

    best_nrmse = float('inf')
    ckpt_dir = "checkpoint"
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    
    # Naming
    loss_type_str = args.use_loss
    if args.use_loss == 'surrogate':
        loss_type_str = f"surrogate{surrogate_num_str}"
        
    pattern_regex = f"{args.train_model}_{loss_type_str}_(?P<num>\\d+)_.*\\.pth"
    current_num = get_next_version(ckpt_dir, pattern_regex)
    
    wandb_name = wandb.run.name if wandb.run.name else "unknown"
    
    # --- 修改點 2: 在檔名加入 frac 資訊 ---
    # 將 frac 放在 num 的後面，避免破壞 get_next_version 的 regex 匹配
    base_filename = f"{args.train_model}_{loss_type_str}_{current_num}_frac{args.data_fraction}_{wandb_name}"
    
    best_model_path = os.path.join(ckpt_dir, f"{base_filename}_best.pth")
    final_model_path = os.path.join(ckpt_dir, f"{base_filename}_last.pth")
    
    print(f"📝 Models will be saved as: {base_filename}_{{best/last}}.pth")

    for epoch in range(total_epochs):
        model.train()
        total_sup = 0
        total_phy = 0
        
        lambda_val = 0.0
        max_lambda = 0.01 if args.use_loss == 'surrogate' else 0.01
        
        if args.use_loss in ['physical', 'surrogate']:
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
        
        progress_bar = tqdm(train_loader, desc=f"Inverse Ep{epoch+1} (λ={lambda_val:.4f})", leave=False)
        
        for u_batch, v_batch, params_target in progress_bar:
            u_batch = u_batch.to(config.DEVICE)
            v_batch = v_batch.to(config.DEVICE)
            params_target = params_target.to(config.DEVICE)
            
            optimizer.zero_grad()
            preds_norm = model(u_batch)
            
            # Supervised Loss
            abs_diff = torch.abs(preds_norm - params_target)
            loss_sup = torch.mean(loss_weights * abs_diff)
            
            # Auxiliary Loss
            loss_aux = torch.tensor(0.0).to(config.DEVICE)
            if lambda_val > 0 or (gradnorm is not None):
                if args.use_loss == 'surrogate':
                    recon_img = surrogate(preds_norm)
                    loss_aux = criterion_phy_surrogate(recon_img, u_batch)
                elif args.use_loss == 'physical':
                    params_pred_real = dataset.scaler.inverse_transform_tensor(preds_norm)
                    loss_aux = criterion_phy_pinn(u_batch, v_batch, params_pred_real)
            
            if gradnorm is not None and args.use_loss in ['physical', 'surrogate']:
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
            
        val_metrics = validate_inverse(
            model, val_loader, 
            surrogate if (args.use_loss == 'surrogate' and lambda_val > 0) else None, 
            criterion_phy_surrogate
        )
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
    parser = argparse.ArgumentParser(description="Unified Training Pipeline")
    parser.add_argument('--pretrain', action='store_true', default=DEFAULT_ARGS['pretrain'], help='Train Phase 1 (Surrogate)')
    parser.add_argument('--pretrain-physloss', action='store_true', default=DEFAULT_ARGS['pretrain_physloss'], help='Use physical loss in Phase 1')
    parser.add_argument('--train-model', type=str, default=DEFAULT_ARGS['train_model'], choices=['CNN', 'MLP'], help='Model architecture for Phase 2')
    parser.add_argument('--use-loss', type=str, default=DEFAULT_ARGS['use_loss'], choices=['pure', 'physical', 'surrogate'], help='Loss type for Phase 2')
    parser.add_argument('--phys-gradual', action='store_true', default=DEFAULT_ARGS['phys_gradual'], help='Use gradual introduction for auxiliary loss')
    parser.add_argument('--data-fraction', type=float, default=DEFAULT_ARGS['data_fraction'], help='Fraction of Train+Val data to use (for ablation)')
    parser.add_argument('--surrogate-num', type=int, default=DEFAULT_ARGS['surrogate_num'], help='Surrogate model version to use (Phase 2)')
    
    args = parser.parse_args()
    
    set_seed(42)
    wandb.login(key="c45f78d1fb5c9023cf8d9787e2d3828bd0f891e1")
    
    # ====================================================
    # 📂 Load Data from Separate Files (已分好的資料)
    # ====================================================
    print("📂 Loading datasets from separate files...")
    
    # 1. 分別讀取 Train 和 Val
    train_set = TuringDataset(config.TRAIN_PATH)
    val_set = TuringDataset(config.VAL_PATH)

    # 2. 設定全域 Scaler
    dataset = train_set 
    dataset.scaler.to_device(config.DEVICE)

    # 3. Data Ablation (資料消融測試)
    if args.data_fraction < 1.0:
        total_train = len(train_set)
        used_size = int(total_train * args.data_fraction)
        unused_size = total_train - used_size
        
        train_set, _ = random_split(
            train_set, [used_size, unused_size], 
            generator=torch.Generator().manual_seed(42)
        )
        print(f"📉 Data Ablation: Using {args.data_fraction:.1%} of Training data ({used_size} samples)")

    # 4. 建立 DataLoader
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"📦 Data Loaded: Train={len(train_set)}, Val={len(val_set)}")
    
    pretrain_num = None
    if args.pretrain:
        # --- 修改點 3: 傳入 args.data_fraction ---
        pretrain_num = train_phase1(train_loader, val_loader, args.pretrain_physloss, args.data_fraction)
        
    train_phase2(train_loader, val_loader, dataset, args, pretrain_num)

if __name__ == "__main__":
    main()