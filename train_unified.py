import os
import argparse
import random
import re
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import wandb

# Import modules
from model import config
from model.dataset import TuringDataset
from model.model import (
    MLPNet, 
    PaperMinimalCNN, PaperMinimalCNN2, PaperMinimalCNN2MaxPool, PaperMinimalCNN2Stride,
    PaperFlexibleCNN
)
from model.loss import PhysicsLoss, GradNorm
from model.utils import calculate_multidim_nrmse

def get_next_version(directory, pattern_regex):
    """
    Scans directory for files matching regex and returns next version number.
    Regex must contain a named group (?P<num>\\d+).
    """
    max_num = 0
    if not os.path.exists(directory):
        return 1
        
    for filename in os.listdir(directory):
        match = re.match(pattern_regex, filename)
        if match:
            num = int(match.group('num'))
            if num > max_num:
                max_num = num
    return max_num + 1

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

def validate_model(model, loader, is_cnn=False):
    model.eval()
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for u_batch, _, p_batch in loader:
            u_batch = u_batch.to(config.DEVICE)
            if is_cnn:
                u_input = u_batch[:, 0:1, :, :] if u_batch.shape[1] > 1 else u_batch
            else:
                u_input = u_batch
            
            p_pred = model(u_input)
            all_targets.append(p_batch.cpu())
            all_preds.append(p_pred.cpu())
    
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    nrmse = calculate_multidim_nrmse(all_targets, all_preds)
    return {"nrmse_multi_dim": nrmse}

def train(args):
    # Name Construction
    if args.model_arch == 'mlp':
        model_name = "MLP"
    elif args.model_arch == 'flexible_cnn':
        model_name = f"CNN_Flex-L{args.conv_layers}-{args.sampling}"
    else:
        model_name = f"CNN-{args.model_arch}"
        
    num_samples_str = f"{args.num_samples}" if args.num_samples else "16000"
    
    # Loss Tag Construction (includes lambda if physical)
    loss_tag = args.use_loss
    if args.use_loss == 'physical':
        loss_tag = f"{args.use_loss}_lam{args.lambda_phy}"
    
    wandb_name = f"{model_name}-{loss_tag}-n{num_samples_str}"
    
    if args.model_arch not in ['mlp']:
        wandb_name += f"-nk{args.nk}"

    wandb.init(project=config.WANDB_PROJECT, name=wandb_name, reinit=True)
    
    print(f"🚀 Training {model_name} with {loss_tag}")
    
    # 1. Dataset
    dataset = TuringDataset(
        npz_path=config.TRAIN_PATH,
        num_samples=args.num_samples,
        val_split=0.25,
        seed=42
    )
    dataset.scaler.to_device(config.DEVICE)
    train_set, val_set = dataset.get_train_val_datasets()
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 2. Model
    is_cnn = args.model_arch != 'mlp'
    
    if args.model_arch == 'mlp':
        model = MLPNet().to(config.DEVICE)
    elif args.model_arch == 'flexible_cnn':
        print(f"   Building Flexible CNN: layers={args.conv_layers}, sampling={args.sampling}")
        model = PaperFlexibleCNN(nk=args.nk, np_size=args.np, nf=args.nf, layers=args.conv_layers, sampling=args.sampling).to(config.DEVICE)
    elif args.model_arch == 'cnn1':
        model = PaperMinimalCNN(nk=args.nk, np_size=args.np, nf=args.nf).to(config.DEVICE)
    elif args.model_arch == 'cnn2':
        model = PaperMinimalCNN2(nk=args.nk, np_size=args.np, nf=args.nf).to(config.DEVICE)
    elif args.model_arch == 'cnn2pool':
        model = PaperMinimalCNN2MaxPool(nk=args.nk, np_size=args.np, nf=args.nf).to(config.DEVICE)
    elif args.model_arch == 'cnn2stride':
        model = PaperMinimalCNN2Stride(nk=args.nk, np_size=args.np, nf=args.nf).to(config.DEVICE)
    
    # 3. Optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 4. Scheduler
    total_epochs = config.EPOCHS
    warmup_epochs = int(0.1 * total_epochs)
    warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6/args.lr, end_factor=1.0, total_iters=warmup_epochs)
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs-warmup_epochs, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs])
    
    # 5. Loss
    criterion_mse = nn.MSELoss()
    criterion_physics = PhysicsLoss(dx=config.DX, s_diffusion=config.S_DIFFUSION).to(config.DEVICE)
    
    # 6. Save Path Construction
    ckpt_dir = "paper_checkpoints"
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    
    # Regex designed to differentiate architectures
    if args.model_arch == 'flexible_cnn':
        base_id = f"PaperPINN_CNN_Flex_L{args.conv_layers}_{args.sampling}_{loss_tag}_n{num_samples_str}"
        pattern = f"PaperPINN_CNN_Flex_L{args.conv_layers}_{args.sampling}_{loss_tag}_n{num_samples_str}_(?P<num>\\d+)_.*\\.pth"
    elif args.model_arch == 'mlp':
        base_id = f"PaperPINN_MLP_{loss_tag}_n{num_samples_str}"
        pattern = f"PaperPINN_MLP_{loss_tag}_n{num_samples_str}_(?P<num>\\d+)_.*\\.pth"
    else:
        base_id = f"PaperPINN_{args.model_arch.upper()}_{loss_tag}_nk{args.nk}_n{num_samples_str}"
        pattern = f"PaperPINN_{args.model_arch.upper()}_{loss_tag}_nk{args.nk}_n{num_samples_str}_(?P<num>\\d+)_.*\\.pth"
        
    current_num = get_next_version(ckpt_dir, pattern)
    best_model_path = os.path.join(ckpt_dir, f"{base_id}_{current_num}_best.pth")
    final_model_path = os.path.join(ckpt_dir, f"{base_id}_{current_num}_last.pth")
    
    print(f"📝 Saving to: {best_model_path}")
    
    best_nrmse = float('inf')
    
    # Training
    for epoch in range(total_epochs):
        model.train()
        total_sup, total_phy = 0, 0
        
        # Physics Schedule
        lambda_val = args.lambda_phy if args.use_loss == 'physical' else 0.0
        if args.phys_gradual and args.use_loss == 'physical' and epoch < 20:
            lambda_val = 0.0
            
        pbar = tqdm(train_loader, desc=f"Ep{epoch+1}", leave=False)
        for u_batch, v_batch, params_target in pbar:
            u_batch = u_batch.to(config.DEVICE)
            v_batch = v_batch.to(config.DEVICE)
            params_target = params_target.to(config.DEVICE)
            
            u_input = u_batch[:, 0:1] if is_cnn else u_batch
            
            optimizer.zero_grad()
            preds = model(u_input)
            
            loss_sup = criterion_mse(preds, params_target)
            loss_phy = torch.tensor(0.0, device=config.DEVICE)
            
            if lambda_val > 0:
                params_real = dataset.scaler.inverse_transform_tensor(preds)
                u_real = u_batch[:, 0:1] if is_cnn else u_batch
                v_real = v_batch[:, 0:1] if is_cnn else v_batch
                loss_phy = criterion_physics(u_real, v_real, params_real)
                
            loss = loss_sup + lambda_val * loss_phy
            loss.backward()
            optimizer.step()
            
            total_sup += loss_sup.item()
            total_phy += loss_phy.item()
        
        scheduler.step()
        
        # Val
        metrics = validate_model(model, val_loader, is_cnn=is_cnn)
        val_nrmse = metrics['nrmse_multi_dim']
        
        avg_sup = total_sup / len(train_loader)
        
        print(f"[Ep {epoch+1}] Sup={avg_sup:.5f} | Val={val_nrmse:.2%} | Saved={val_nrmse < best_nrmse}")
        wandb.log({"train/sup": avg_sup, "val/nrmse": val_nrmse})
        
        if val_nrmse < best_nrmse:
            best_nrmse = val_nrmse
            torch.save(model.state_dict(), best_model_path)
            
        if epoch % 10 == 0:
            torch.save(model.state_dict(), final_model_path)

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Updated choices
    parser.add_argument('--model-arch', type=str, default='flexible_cnn', 
                        choices=['mlp', 'cnn1', 'cnn2', 'cnn2pool', 'cnn2stride', 'flexible_cnn'])
    
    parser.add_argument('--conv-layers', type=int, default=4, help='Number of layers for Flexible CNN')
    parser.add_argument('--sampling', type=str, default='none', choices=['none', 'maxpool', 'avgpool', 'stride', 'dilated'],
                        help='Sampling method for Flexible CNN')
    
    # Old CNN Params
    parser.add_argument('--nk', type=int, default=5)
    parser.add_argument('--np', type=int, default=5)
    parser.add_argument('--nf', type=int, default=5)
    
    # General
    parser.add_argument('--use-loss', type=str, default='physical',
                        choices=['pure', 'physical'], help='Loss type: pure (supervised) or physical (PINN)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Total number of samples to use for train+val (None = use all 16000)')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--lambda-phy', type=float, default=0.01,
                        help='Weight for physics loss')
    parser.add_argument('--phys-gradual', action='store_true', default=False,
                        help='Use gradual introduction for physics loss')
    
    args = parser.parse_args()
    # Set default learning rate based on model architecture
    if args.lr is None:
        args.lr = 1e-3
    
    set_seed(42)
    train(args)