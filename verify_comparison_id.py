import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import argparse

# Check if model module is available, if not append path
sys.path.append(os.getcwd())

from model.model import PaperFlexibleCNN
from model.utils import MinMaxScaler
from model.loss import PhysicsLoss

# Paths
NPZ_PATH = "turing_patterns_dataset_clean_test.npz"
CHECKPOINT_DIR = "paper_checkpoints"

# specific models requested
MODEL_PHY_NAME = "PaperPINN_CNN_Flex_L4_maxpool_physical_lam0.001_n8000_1_best.pth"
MODEL_PURE_NAME = "PaperPINN_CNN_Flex_L4_maxpool_pure_n8000_1_best.pth"

DX = 1.0
S_DIFFUSION = 0.4

def load_data(npz_path):
    print(f"Loading data from {npz_path}...")
    data = np.load(npz_path)
    # Ensure correct shape (N, 1, 128, 128)
    u = data['u']
    v = data['v']
    if u.ndim == 3:
        u = u[:, np.newaxis, :, :]
    if v.ndim == 3:
        v = v[:, np.newaxis, :, :]
        
    ids = data['ids']
    
    # Load params [a, b, c, delta]
    # Check if a, b, c, delta exist individually or as params
    if 'params' in data:
        params = data['params']
    else:
        # Assuming a, b, c, delta are keys
        # If some are scalar/missing, handle it. Based on inspect_turing_npz, 'a', 'delta' are arrays.
        # inspect_turing_npz didn't show b, c, but dataset.py assumes they exist.
        # Let's try to load them, if fail, assume they are constants (should assume dataset.py is correct)
        a = data['a']
        b = data['b']
        c = data['c']
        delta = data['delta']
        params = np.stack([a, b, c, delta], axis=1)
        
    # Create Scaler (Fit on ALL data as in training)
    scaler = MinMaxScaler(
        data_min=np.min(params, axis=0), 
        data_max=np.max(params, axis=0)
    )
    
    return u, v, ids, params, scaler

import re

def parse_model_config(filename):
    """Parses layers and sampling from filename"""
    # Regex for "PaperPINN_CNN_Flex_L{layers}_{sampling}_..."
    match = re.search(r"PaperPINN_CNN_Flex_L(\d+)_([a-zA-Z0-9]+)_", filename)
    if match:
        layers = int(match.group(1))
        sampling = match.group(2)
        return layers, sampling
    return 2, 'maxpool' # Default fallback

def load_model(path, layers=None, sampling=None, nk=5, nf=5):
    print(f"Loading model from {path}...")
    
    # If not provided, try to infer from filename
    if layers is None or sampling is None:
        filename = os.path.basename(path)
        f_layers, f_sampling = parse_model_config(filename)
        if layers is None: layers = f_layers
        if sampling is None: sampling = f_sampling
        print(f"  -> Inferred config: layers={layers}, sampling={sampling}")

    model = PaperFlexibleCNN(nk=nk, np_size=5, nf=nf, layers=layers, sampling=sampling, input_size=128)
    # Load state dict
    try:
        state_dict = torch.load(path, map_location='cpu')
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading model {path}: {e}")
        return None
    model.eval()
    return model

def compute_residual_map(u, v, params_real, dx=1.0, s=0.4):
    """
    Computes the residual map (Physics Loss per pixel)
    u, v: (1, 1, H, W) torch tensors
    params_real: (1, 4) torch tensor, denormalized [a, b, c, delta]
    Returns: Residual Map (H, W)
    """
    # Laplacian kernel
    laplace_k = torch.tensor([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    def laplacian(tensor):
        padded = F.pad(tensor, (1, 1, 1, 1), mode='circular')
        return F.conv2d(padded, laplace_k) / (dx**2)

    # Params
    a = params_real[:, 0].view(1, 1, 1, 1)
    b = params_real[:, 1].view(1, 1, 1, 1)
    c = params_real[:, 2].view(1, 1, 1, 1)
    delta = params_real[:, 3].view(1, 1, 1, 1)

    # 1. Diffusion
    lap_u = laplacian(u)
    lap_v = laplacian(v)

    # 2. Reaction
    v_safe = torch.clamp(v, min=1e-6)
    denom = v_safe * (1 + c * u**2)
    Ru = a - (b * u) + (u**2 / denom)
    Rv = (u**2) - v

    # 3. Residual
    du_dt = s * lap_u + Ru
    dv_dt = s * delta * lap_v + Rv

    # Combine (Sum of squares)
    residual_map = du_dt**2 + dv_dt**2
    return residual_map.squeeze().detach().numpy()

def main():
    parser = argparse.ArgumentParser(description="Visualize Turing Patterns and Residual Maps for a specific ID")
    parser.add_argument("--id", type=int, help="ID of the pattern to visualize", required=False)
    parser.add_argument("--model_phy", type=str, default=MODEL_PHY_NAME, help="Filename of the physical model")
    parser.add_argument("--model_pure", type=str, default=MODEL_PURE_NAME, help="Filename of the pure model")
    args = parser.parse_args()

    # 1. Load Data
    u_all, v_all, ids_all, params_all, scaler = load_data(NPZ_PATH)

    # 2. Select ID
    target_id = args.id
    if target_id is None:
        try:
            target_id = int(input("Please enter the ID you want to inspect: "))
        except ValueError:
            print("Invalid ID.")
            return

    indices = np.where(ids_all == target_id)[0]
    if len(indices) == 0:
        print(f"ID {target_id} not found in dataset.")
        return
    idx = indices[0]
    
    # Extract sample
    u_np = u_all[idx] # (1, 128, 128)
    v_np = v_all[idx]
    params_gt_np = params_all[idx] # (4,)

    print(f"\nProcessing ID: {target_id}")
    print(f"GT Params: a={params_gt_np[0]:.4f}, b={params_gt_np[1]:.4f}, c={params_gt_np[2]:.4f}, delta={params_gt_np[3]:.4f}")

    # Prepare tensors
    u_tensor = torch.tensor(u_np).unsqueeze(0) # (1, 1, 128, 128)
    v_tensor = torch.tensor(v_np).unsqueeze(0)
    params_gt_tensor = torch.tensor(params_gt_np).unsqueeze(0) # (1, 4)

    # 3. Load Models
    model_phy_path = os.path.join(CHECKPOINT_DIR, args.model_phy)
    model_pure_path = os.path.join(CHECKPOINT_DIR, args.model_pure)

    # Let dynamic parser handle layers/sampling
    model_phy = load_model(model_phy_path)
    model_pure = load_model(model_pure_path)

    # 4. Predict
    # Normalizer expects (N, 4), models output (N, 4) in [0,1]
    
    if model_pure:
        with torch.no_grad():
            pred_norm_pure = model_pure(u_tensor)
            params_pred_pure = scaler.inverse_transform_numpy(pred_norm_pure.numpy())[0]
            params_pred_pure_tensor = torch.tensor(params_pred_pure).unsqueeze(0)
            print(f"Pure Pred: a={params_pred_pure[0]:.4f}, b={params_pred_pure[1]:.4f}, c={params_pred_pure[2]:.4f}, delta={params_pred_pure[3]:.4f}")
    
    if model_phy:
        with torch.no_grad():
            pred_norm_phy = model_phy(u_tensor)
            params_pred_phy = scaler.inverse_transform_numpy(pred_norm_phy.numpy())[0]
            params_pred_phy_tensor = torch.tensor(params_pred_phy).unsqueeze(0)
            print(f"Phy Pred : a={params_pred_phy[0]:.4f}, b={params_pred_phy[1]:.4f}, c={params_pred_phy[2]:.4f}, delta={params_pred_phy[3]:.4f}")

    # 5. Compute Residuals
    # GT Residual (should be close to 0)
    res_gt = compute_residual_map(u_tensor, v_tensor, params_gt_tensor, dx=DX, s=S_DIFFUSION)
    
    res_pure = None
    if model_pure:
        res_pure = compute_residual_map(u_tensor, v_tensor, params_pred_pure_tensor, dx=DX, s=S_DIFFUSION)
        
    res_phy = None
    if model_phy:
        res_phy = compute_residual_map(u_tensor, v_tensor, params_pred_phy_tensor, dx=DX, s=S_DIFFUSION)

    # 6. Plotting
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Row 1: U, V, GT Residual
    im0 = axes[0, 0].imshow(u_np[0], cmap='RdBu_r')
    axes[0, 0].set_title(f"Field u (ID: {target_id})")
    plt.colorbar(im0, ax=axes[0, 0])
    
    im1 = axes[0, 1].imshow(v_np[0], cmap='RdBu_r')
    axes[0, 1].set_title(f"Field v (ID: {target_id})")
    plt.colorbar(im1, ax=axes[0, 1])

    # Row 2: Prediction Resids (Pure vs Phy)

    if res_pure is not None:
        im3 = axes[1, 0].imshow(res_pure, cmap='viridis')
        axes[1, 0].set_title(f"Pure Model Residual\nErr: {np.mean(res_pure):.2e}")
        plt.colorbar(im3, ax=axes[1, 0])

    if res_phy is not None:
        im4 = axes[1, 1].imshow(res_phy, cmap='viridis')
        axes[1, 1].set_title(f"Phy Model Residual\nErr: {np.mean(res_phy):.2e}")
        plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
