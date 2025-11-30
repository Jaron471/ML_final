import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tqdm import tqdm

# Environment Setup
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Import modules
from model import config
from model.dataset import TuringDataset
# We need to redefine the model class here or import it if you moved it to model.py
# Assuming it's defined in train_paper_pinn.py, but for verification it's safer to have the definition available.
# Ideally, move PaperMinimalCNN to model/model.py. 
# For now, I will include the class definition here to make this script standalone.

import torch.nn as nn

# ==========================================
# Model Definition (Must match training)
# ==========================================
class PaperMinimalCNN(nn.Module):
    """
    Minimal CNN architecture from the paper (nk/np/nf)
    """
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN, self).__init__()
        
        # Input is fixed to 1 Channel (only u)
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
        return torch.sigmoid(x) # Output normalized [0, 1]

# ==========================================
# Metric Calculation
# ==========================================
def calculate_metrics(targets_norm, preds_norm, targets_real, preds_real):
    """
    Calculate evaluation metrics.
    targets_norm, preds_norm: Normalized data [0,1] (For NRMSE)
    targets_real, preds_real: Physical values (For MAE)
    """
    metrics = {}
    param_names = ['a', 'b', 'c', 'delta']
    
    # 1. Per-parameter metrics (Normalized space)
    # Since data is MinMax scaled to [0, 1], Range = 1
    # RMSE in this space is equivalent to NRMSE (Normalized by Range)
    for i, name in enumerate(param_names):
        mse = mean_squared_error(targets_norm[:, i], preds_norm[:, i])
        rmse = np.sqrt(mse)
        r2 = r2_score(targets_norm[:, i], preds_norm[:, i])
        
        metrics[f'NRMSE_{name}'] = rmse 
        metrics[f'R2_{name}'] = r2
        
        # Calculate Real MAE
        mae_real = mean_absolute_error(targets_real[:, i], preds_real[:, i])
        metrics[f'MAE_real_{name}'] = mae_real

    # 2. Joint NRMSE (Paper Definition)
    # Formula: RMSE_total / || mean(y_target) ||
    num_elements = targets_norm.size
    total_sse = np.sum((targets_norm - preds_norm)**2)
    global_rmse = np.sqrt(total_sse / num_elements)
    
    # Denominator: Euclidean norm of the mean vector of targets
    y_mean_vec = np.mean(targets_norm, axis=0)
    norm_y_mean = np.linalg.norm(y_mean_vec)
    
    metrics['NRMSE_Joint'] = global_rmse / norm_y_mean if norm_y_mean != 0 else 0.0
    
    return metrics

# ==========================================
# Main Verification Logic
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="Verify PaperMinimalCNN Model")
    
    # Model Arguments (Must match what you trained with!)
    parser.add_argument('--model-path', type=str, required=True, help='Path to the .pth model file')
    parser.add_argument('--nk', type=int, default=5, help='Number of kernels')
    parser.add_argument('--np', type=int, default=5, help='Kernel size')
    parser.add_argument('--nf', type=int, default=5, help='Hidden neurons')
    
    # Data Arguments
    parser.add_argument('--test-path', type=str, default=None, help='Path to test .npz file (default: config.TEST_PATH)')
    
    args = parser.parse_args()

    # 1. Data Loading
    # If config doesn't have TEST_PATH, fall back to NPZ_PATH or define it here
    test_data_path = args.test_path if args.test_path else getattr(config, 'TEST_PATH', config.NPZ_PATH)
    
    print(f"📦 Loading Test Data from: {test_data_path}")
    if not os.path.exists(test_data_path):
        print(f"❌ Error: Data file not found at {test_data_path}")
        return

    test_dataset = TuringDataset(test_data_path)
    loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"📊 Dataset size: {len(test_dataset)}")

    # 2. Model Loading
    print(f"🏗️  Initializing PaperMinimalCNN (nk={args.nk}, np={args.np}, nf={args.nf})")
    model = PaperMinimalCNN(nk=args.nk, np_size=args.np, nf=args.nf).to(config.DEVICE)

    print(f"🔍 Loading Weights: {args.model_path}")
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=config.DEVICE))
        print("✅ Model weights loaded successfully.")
    except Exception as e:
        print(f"❌ Failed to load model weights: {e}")
        print("💡 Tip: Check if 'nk', 'np', 'nf' match the trained model.")
        return

    model.eval()

    # 3. Inference
    all_preds_norm = []
    all_targets_norm = []
    all_preds_real = []
    all_targets_real = []

    print("🚀 Running Inference...")
    with torch.no_grad():
        for u_batch, _, params_target in tqdm(loader):
            u_batch = u_batch.to(config.DEVICE)
            
            # The model expects (B, 1, 128, 128)
            # If dataset returns (B, 2, ...), slice it
            if u_batch.shape[1] > 1:
                u_input = u_batch[:, 0:1, :, :]
            else:
                u_input = u_batch

            # Predict (Normalized Space [0,1])
            preds_norm = model(u_input)
            
            # Convert to CPU/Numpy
            preds_norm_np = preds_norm.cpu().numpy()
            targets_norm_np = params_target.numpy()
            
            all_preds_norm.append(preds_norm_np)
            all_targets_norm.append(targets_norm_np)
            
            # Inverse Transform to Physics Space (using dataset's scaler)
            preds_real = test_dataset.scaler.inverse_transform_numpy(preds_norm_np)
            targets_real = test_dataset.scaler.inverse_transform_numpy(targets_norm_np)
            
            all_preds_real.append(preds_real)
            all_targets_real.append(targets_real)

    # Stack results
    all_preds_norm = np.vstack(all_preds_norm)
    all_targets_norm = np.vstack(all_targets_norm)
    all_preds_real = np.vstack(all_preds_real)
    all_targets_real = np.vstack(all_targets_real)

    # 4. Metrics
    metrics = calculate_metrics(all_targets_norm, all_preds_norm, all_targets_real, all_preds_real)

    # 5. Report
    print("\n" + "="*70)
    print(f"📊 Evaluation Report: {os.path.basename(args.model_path)}")
    print("="*70)

    print(f"\n🏆 Joint NRMSE (Paper Metric):  {metrics['NRMSE_Joint']:.2%}")
    print("-" * 46)

    print(f"\nDetailed Normalized RMSE (Lower is Better)")
    print(f"{'Parameter':<10} | {'NRMSE (%)':<15} | {'R2 Score':<15}")
    print("-" * 46)
    for name in ['a', 'b', 'c', 'delta']:
        nrmse_val = metrics[f'NRMSE_{name}']
        r2_val = metrics[f'R2_{name}']
        print(f"{name:<10} | {nrmse_val:>6.2%}          | {r2_val:>8.4f}")
    
    print("-" * 46)
    print(f"\n📏 Real-World Mean Absolute Error (MAE)")
    print(f"{'Parameter':<10} | {'MAE':<15} | {'Range (Approx)'}")
    print("-" * 46)
    
    # Calculate approx range from targets for reference
    ranges = {
        'a': np.ptp(all_targets_real[:, 0]),
        'b': np.ptp(all_targets_real[:, 1]),
        'c': np.ptp(all_targets_real[:, 2]),
        'delta': np.ptp(all_targets_real[:, 3])
    }
    
    for name in ['a', 'b', 'c', 'delta']:
        mae = metrics[f'MAE_real_{name}']
        print(f"{name:<10} | {mae:<15.5f} | {ranges[name]:.2f}")
    print("="*70)

if __name__ == "__main__":
    main()