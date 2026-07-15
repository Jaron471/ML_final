import torch
import torch.nn.functional as F
import numpy as np
import os
import sys
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader, TensorDataset

# Check if model module is available, if not append path
sys.path.append(os.getcwd())

from model.model import PaperFlexibleCNN
from model.utils import MinMaxScaler

# Constants
MERGED_NPZ_PATH = "turing_patterns_dataset_merged.npz"
TEST_NPZ_PATH = "turing_patterns_dataset_clean_test.npz"
CHECKPOINT_DIR = "paper_checkpoints"

# Defaults
BATCH_SIZE = 32
DX = 1.0
S_DIFFUSION = 0.4

# Default Models (same as verify script)
MODEL_PHY_NAME = "PaperPINN_CNN_Flex_L2_maxpool_physical_lam0.001_n4000_1_best.pth"
MODEL_PURE_NAME = "PaperPINN_CNN_Flex_L2_maxpool_pure_n4000_1_best.pth"

def get_scaler(merged_path):
    """Load merged dataset to fit the scaler globally."""
    print(f"Loading merged data from {merged_path} to fit Scaler...")
    data = np.load(merged_path)
    if 'params' in data:
        params = data['params']
    else:
        # Fallback if params keys are separate
        a = data['a']
        b = data['b']
        c = data['c']
        delta = data['delta']
        params = np.stack([a, b, c, delta], axis=1)
        
    scaler = MinMaxScaler(
        data_min=np.min(params, axis=0), 
        data_max=np.max(params, axis=0)
    )
    return scaler

def load_test_data(test_path):
    print(f"Loading test data from {test_path}...")
    data = np.load(test_path)
    u = data['u']
    v = data['v']
    
    # Ensure (N, 1, 128, 128)
    if u.ndim == 3: u = u[:, np.newaxis, :, :]
    if v.ndim == 3: v = v[:, np.newaxis, :, :]
    
    if 'params' in data:
        params = data['params']
    else:
        a = data['a']
        b = data['b']
        c = data['c']
        delta = data['delta']
        params = np.stack([a, b, c, delta], axis=1)
        
    return u, v, params

import re
def parse_model_config(filename):
    match = re.search(r"PaperPINN_CNN_Flex_L(\d+)_([a-zA-Z0-9]+)_", filename)
    if match:
        return int(match.group(1)), match.group(2)
    return 2, 'maxpool'

def load_model(path, nk=5, nf=5):
    print(f"Loading model from {path}...")
    filename = os.path.basename(path)
    layers, sampling = parse_model_config(filename)
    print(f"  -> Config: layers={layers}, sampling={sampling}")

    model = PaperFlexibleCNN(nk=nk, np_size=5, nf=nf, layers=layers, sampling=sampling, input_size=128)
    try:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    model.eval()
    return model

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
    parser = argparse.ArgumentParser(description="Calculate average residuals on Test Set")
    parser.add_argument("--model_phy", type=str, default=MODEL_PHY_NAME)
    parser.add_argument("--model_pure", type=str, default=MODEL_PURE_NAME)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Scaler
    if os.path.exists(MERGED_NPZ_PATH):
        scaler = get_scaler(MERGED_NPZ_PATH)
    else:
        print(f"Warning: {MERGED_NPZ_PATH} not found. Fitting scaler on test data directly (might be slightly off).")
        # Placeholder scaler
        scaler = None 

    # 2. Load Test Data
    if not os.path.exists(TEST_NPZ_PATH):
        print(f"Error: {TEST_NPZ_PATH} not found.")
        return
        
    u_test, v_test, params_test = load_test_data(TEST_NPZ_PATH)
    
    # If scaler wasn't loaded from merged, fit on test
    if scaler is None:
        scaler = MinMaxScaler(
            data_min=np.min(params_test, axis=0), 
            data_max=np.max(params_test, axis=0)
        )

    # Create DataLoader
    dataset = TensorDataset(
        torch.tensor(u_test, dtype=torch.float32), 
        torch.tensor(v_test, dtype=torch.float32), 
        torch.tensor(params_test, dtype=torch.float32)
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Test samples: {len(dataset)}")

    # 3. Load Models
    model_phy_path = os.path.join(CHECKPOINT_DIR, args.model_phy)
    model_pure_path = os.path.join(CHECKPOINT_DIR, args.model_pure)

    model_phy = load_model(model_phy_path)
    model_pure = load_model(model_pure_path)
    
    if model_phy: model_phy.to(device)
    if model_pure: model_pure.to(device)
    
    # Transfer scaler to device for potential tensor ops if needed, 
    # but here we do inverse transform via numpy mostly or we can add tensor support to scaler
    # The existing scaler class has to_device, let's use it if we want full gpu pipeline
    # But inverse_transform_numpy expects numpy. 
    # Let's stick to CPU for scalar inverse transform step to match previous logic, or optimize.
    # Optimization: Model -> GPU -> Param Pred (GPU) -> Inverse Transform (GPU?? Scaler has `inverse_transform_tensor`)
    
    scaler.to_device(device) # Prepare scaler for GPU ops

    # 4. Loop
    total_res_gt = 0.0
    total_res_pure = 0.0
    total_res_phy = 0.0
    count = 0

    print("Computing residuals...")
    with torch.no_grad():
        for u_b, v_b, params_gt_b in tqdm(loader):
            batch_len = u_b.size(0)
            u_b = u_b.to(device)
            v_b = v_b.to(device)
            params_gt_b = params_gt_b.to(device)

            # A. GT Residual
            res_val = compute_residual_batch(u_b, v_b, params_gt_b, dx=DX, s=S_DIFFUSION, device=device)
            total_res_gt += res_val * batch_len

            # B. Pure Model
            if model_pure:
                pred_norm = model_pure(u_b) # (B, 4) in [0,1]
                # Inverse transform on GPU
                params_pure_b = scaler.inverse_transform_tensor(pred_norm)
                res_val = compute_residual_batch(u_b, v_b, params_pure_b, dx=DX, s=S_DIFFUSION, device=device)
                total_res_pure += res_val * batch_len
            
            # C. Physical Model
            if model_phy:
                pred_norm = model_phy(u_b)
                params_phy_b = scaler.inverse_transform_tensor(pred_norm)
                res_val = compute_residual_batch(u_b, v_b, params_phy_b, dx=DX, s=S_DIFFUSION, device=device)
                total_res_phy += res_val * batch_len
            
            count += batch_len

    # 5. Report
    avg_gt = total_res_gt / count
    print(f"\nResults on Test Set ({count} samples):")
    print("-" * 40)
    print(f"Avg GT Residual:       {avg_gt:.6e}")
    
    if model_pure:
        avg_pure = total_res_pure / count
        print(f"Avg Pure Model Res:    {avg_pure:.6e}")
    
    if model_phy:
        avg_phy = total_res_phy / count
        print(f"Avg Physical Model Res:{avg_phy:.6e}")
    print("-" * 40)

if __name__ == "__main__":
    main()
