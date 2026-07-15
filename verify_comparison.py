import os
import sys
import re
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import config
from model.dataset import TuringDataset
from model.model import (
    MLPNet, PaperMinimalCNN, PaperMinimalCNN2, 
    PaperMinimalCNN2MaxPool, PaperMinimalCNN2Stride,
    PaperFlexibleCNN
)

# Copied from calculate_test_residuals.py
def compute_residual_batch(u, v, params_real, dx=1.0, s=0.4, device='cpu'):
    """
    Batch computation of residuals.
    u, v: (B, 1, H, W)
    params_real: (B, 4)
    Returns: scalar mean residual for the batch
    """
    # Kernel
    laplace_k = torch.tensor([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
    
    def laplacian(tensor):
        padded = F.pad(tensor, (1, 1, 1, 1), mode='circular')
        return F.conv2d(padded, laplace_k) / (dx**2)

    a = params_real[:, 0].view(-1, 1, 1, 1)
    b = params_real[:, 1].view(-1, 1, 1, 1)
    c = params_real[:, 2].view(-1, 1, 1, 1)
    delta = params_real[:, 3].view(-1, 1, 1, 1)

    # 1. Diffusion
    lap_u = laplacian(u)
    lap_v = laplacian(v)

    # 2. Reaction
    v_safe = torch.clamp(v, min=1e-6)
    # Avoid div by zero just in case
    denom = v_safe * (1 + c * u**2)
    Ru = a - (b * u) + (u**2 / denom)
    Rv = (u**2) - v

    # 3. Residual
    du_dt = s * lap_u + Ru
    dv_dt = s * delta * lap_v + Rv

    # Mean Squared Residual per pixel
    res_map = du_dt**2 + dv_dt**2
    # Mean over all pixels and batch
    return res_map.mean().item()

def main():
    # Update this path if needed - using verified dataset without pattern-less samples
    TEST_PATH = "turing_patterns_dataset_clean_test.npz"
    CKPT_DIR = "paper_checkpoints"
    DX = 1.0
    S_DIFFUSION = 0.4
    
    if not os.path.exists(CKPT_DIR):
        print("Checkpoint directory not found.")
        return

    # Auto-scan all files in checkpoint dir
    model_files = [f for f in os.listdir(CKPT_DIR) if f.endswith('.pth') and 'best' in f]
    
    # Set val_split=1 to avoid confusing "Train/Val" log message since this is purely testing
    dataset = TuringDataset(TEST_PATH, val_split=1.0)
    # Ensure scaler is on the correct device for inverse transforms
    dataset.scaler.to_device(config.DEVICE)
    
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    results = []

    print(f"🚀 Found {len(model_files)} models. Starting Evaluation...")
    
    for filename in sorted(model_files):
        filepath = os.path.join(CKPT_DIR, filename)
        
        # Parse Filename
        # 1. MLP: PaperPINN_MLP_{loss}_n{num}...
        match_mlp = re.search(r'PaperPINN_MLP_(?P<loss>[a-z]+)(_lam(?P<lam>[\d\.e\-]+))?_n(?P<num>\d+)', filename)
        
        # 2. Old CNN: PaperPINN_CNN{ver}_{loss}_nk{nk}_n{num}...
        match_cnn = re.search(r'PaperPINN_CNN(?P<ver>\d*[a-z]*)_(?P<loss>[a-z]+)(_lam(?P<lam>[\d\.e\-]+))?_nk(?P<nk>\d+)_n(?P<num>\d+)', filename)

        # 4. New Flexible CNN: PaperPINN_CNN_Flex_L{L}_{samp}_{loss}_n{num}...
        match_flex = re.search(r'PaperPINN_CNN_Flex_L(?P<L>\d+)_(?P<samp>\w+)_(?P<loss>[a-z]+)(_lam(?P<lam>[\d\.e\-]+))?_n(?P<num>\d+)', filename)

        lambda_val = "N/A"

        if match_flex:
            arch = 'flexible_cnn'
            layers = int(match_flex.group('L'))
            sampling = match_flex.group('samp')
            loss = match_flex.group('loss')
            nsamp = int(match_flex.group('num'))
            if match_flex.group('lam'): lambda_val = match_flex.group('lam')
            
            model = PaperFlexibleCNN(nk=5, np_size=5, nf=5, layers=layers, sampling=sampling).to(config.DEVICE)
            desc = f"FlexCNN (L={layers}, {sampling})"
            
        elif match_mlp:
            arch = 'mlp'
            loss = match_mlp.group('loss')
            nsamp = int(match_mlp.group('num'))
            if match_mlp.group('lam'): lambda_val = match_mlp.group('lam')
            model = MLPNet().to(config.DEVICE)
            desc = "MLP"
            
        elif match_cnn:
            ver = match_cnn.group('ver') # '1', '2', '2pool'
            loss = match_cnn.group('loss')
            nk = int(match_cnn.group('nk'))
            nsamp = int(match_cnn.group('num'))
            if match_cnn.group('lam'): lambda_val = match_cnn.group('lam')
            
            # Instantiate correct old CNN
            if ver == '1' or ver == '': cls = PaperMinimalCNN
            elif ver == '2': cls = PaperMinimalCNN2
            elif 'pool' in ver: cls = PaperMinimalCNN2MaxPool
            else: cls = PaperMinimalCNN2Stride
            
            model = cls(nk=nk, np_size=5, nf=5).to(config.DEVICE)
            desc = f"CNN{ver} (nk={nk})"
        else:
            continue

        # Load Weights
        try:
            model.load_state_dict(torch.load(filepath, map_location=config.DEVICE))
            model.eval()
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
            
        # Infer
        all_preds, all_targets = [], []
        total_pde_res = 0.0
        total_samples = 0
        
        with torch.no_grad():
            for u, v, p in tqdm(loader, desc=desc, leave=False):
                u = u.to(config.DEVICE)
                v = v.to(config.DEVICE)
                
                u_in = u[:, 0:1] if arch != 'mlp' else u
                preds = model(u_in)
                
                # Compute PDE Residual (using inversed params)
                params_pred = dataset.scaler.inverse_transform_tensor(preds)
                res_val = compute_residual_batch(u, v, params_pred, dx=DX, s=S_DIFFUSION, device=config.DEVICE)
                
                bs = u.size(0)
                total_pde_res += res_val * bs
                total_samples += bs
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(p.numpy())
                
        # Metric
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        avg_res = total_pde_res / total_samples
        
        # Joint NRMSE
        diff = preds - targets
        rmse = np.sqrt(np.mean(diff**2))
        norm = np.linalg.norm(np.mean(targets, axis=0))
        nrmse = rmse / norm if norm > 0 else 0
        
        # R2 Score
        r2_val = r2_score(targets, preds)

        if lambda_val != "N/A":
             desc += f" (λ={lambda_val})"
        
        results.append({
            "Architecture": desc,
            "BaseArch": desc.split(' (λ')[0], # For grouping pure/physical regardless of lambda
            "Loss": loss,
            "Lambda": lambda_val,
            "Samples": nsamp,
            "NRMSE": nrmse,
            "R2": r2_val,
            "PDE_Residual": avg_res,
            "File": filename
        })
        print(f"  -> {desc} | {loss} (λ={lambda_val}) | n={nsamp} | NRMSE={nrmse:.2%} | R2={r2_val:.4f} | Res={avg_res:.2e}")

    # Print Table
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(['Samples', 'Architecture', 'Loss'])
        print("\n" + "="*100)
        print("SUMMARY RESULTS")
        print("="*100)
        print(df[["Architecture", "Loss", "Samples", "NRMSE", "R2", "PDE_Residual"]].to_string(index=False))
        
        # Save CSV
        df.to_csv("comparison_summary.csv", index=False)
        print("\nSaved to comparison_summary.csv")
        
        # Calculate Improvements
        print("\n" + "="*100)
        print("IMPROVEMENT ANALYSIS (Physical vs Pure)")
        print("="*100)
        
        # Group by BaseArch and Samples
        grouped = df.groupby(['BaseArch', 'Samples'])
        
        best_rows = []
        
        for (arch, n_samples), group in grouped:
            # Find Pure
            pure_row = group[group['Loss'] == 'pure']
            if pure_row.empty:
                continue
                
            pure_nrmse = pure_row.iloc[0]['NRMSE']
            pure_r2 = pure_row.iloc[0]['R2']
            pure_res = pure_row.iloc[0]['PDE_Residual']
            
            # Find Physicals
            phys_rows = group[group['Loss'] == 'physical']
            
            # Determine Best Lambda (Lowest NRMSE)
            best_lambda = None
            if not phys_rows.empty:
                best_idx = phys_rows['NRMSE'].idxmin()
                best_phys_row = phys_rows.loc[best_idx]
                best_lambda = best_phys_row['Lambda']
                best_nrmse = best_phys_row['NRMSE']
                best_r2 = best_phys_row['R2']
                best_res = best_phys_row['PDE_Residual']

                # Calculate metrics for Best Physical
                best_imp_nrmse = (pure_nrmse - best_nrmse) / pure_nrmse * 100 if pure_nrmse != 0 else 0
                best_imp_res = (pure_res - best_res) / pure_res * 100 if pure_res != 0 else 0

                best_rows.append({
                    "Architecture": arch,
                    "Samples": n_samples,
                    "Pure_NRMSE": pure_nrmse,
                    "Pure_Residual": pure_res,
                    "Pure_R2": pure_r2,
                    "Best_Physical_Lambda": best_lambda,
                    "Best_Physical_NRMSE": best_nrmse,
                    "Best_Physical_Residual": best_res,
                    "Best_Physical_R2": best_r2,
                    "NRMSE_Imp_Pct": best_imp_nrmse,
                    "Residual_Imp_Pct": best_imp_res
                })

            for _, phys_row in phys_rows.iterrows():
                lam = phys_row['Lambda']
                phy_nrmse = phys_row['NRMSE']
                phy_r2 = phys_row['R2']
                phy_res = phys_row['PDE_Residual']
                
                # Improvements
                # NRMSE: Lower is better. (Pure - Phy)/Pure * 100
                imp_nrmse = (pure_nrmse - phy_nrmse) / pure_nrmse * 100 if pure_nrmse != 0 else 0
                
                # Residual: Lower is better.
                imp_res = (pure_res - phy_res) / pure_res * 100 if pure_res != 0 else 0
                
                is_best = (lam == best_lambda)
                marker = " <<< \u2605 BEST LAMBDA" if is_best else ""

                print(f"[{arch}] (n={n_samples}) @ \u03bb={lam}{marker}")
                print(f"  NRMSE    : {pure_nrmse:.4f} -> {phy_nrmse:.4f} | Imp: {imp_nrmse:+.2f}%")
                print(f"  PDE Res  : {pure_res:.2e} -> {phy_res:.2e} | Imp: {imp_res:+.2f}%")
                print(f"  R2       : {pure_r2:.4f} -> {phy_r2:.4f}")
                print("-" * 50)

        # Save Best Summary CSV
        if best_rows:
            best_df = pd.DataFrame(best_rows)
            # Reorder columns for readability
            cols = [
                "Architecture", "Samples", 
                "Pure_NRMSE", "Best_Physical_NRMSE", "NRMSE_Imp_Pct", 
                "Pure_Residual", "Best_Physical_Residual", "Residual_Imp_Pct", 
                "Pure_R2", "Best_Physical_R2",
                "Best_Physical_Lambda"
            ]
            best_df = best_df[cols]
            best_df.to_csv("best_physical_comparison.csv", index=False)
            print("\nSaved best physical comparison to 'best_physical_comparison.csv'")


if __name__ == "__main__":
    main()