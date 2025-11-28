import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

# Setup environment
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# Import modules
from model import config
from model.dataset import TuringDataset
from model.model import ParameterNet, MLPNet
from model.utils import calculate_multidim_nrmse, find_model_path

# ==========================================
# Default Configuration
# ==========================================
DEFAULT_MODEL_TYPE = "CNN"           # CNN / MLP
DEFAULT_LOSS_TYPE = "surrogate-1"           # pure / physical / surrogate-{num}
DEFAULT_NUM = None                   # None = Latest, or integer
DEFAULT_SUFFIX = "best"              # best / last

def calculate_metrics(targets, preds):
    """Calculate R2, MAE, RMSE, NRMSE for each parameter (in normalized space)"""
    metrics = {}
    param_names = ['a', 'b', 'c', 'delta']
    
    # Per-parameter metrics (計算於歸一化空間 [0,1])
    for i, name in enumerate(param_names):
        metrics[f'R2_{name}'] = r2_score(targets[:, i], preds[:, i])
        metrics[f'MAE_{name}'] = mean_absolute_error(targets[:, i], preds[:, i])
        metrics[f'RMSE_{name}'] = np.sqrt(mean_squared_error(targets[:, i], preds[:, i]))
    
    # Aggregate metrics
    metrics['R2_avg'] = np.mean([metrics[f'R2_{n}'] for n in param_names])
    metrics['MAE_avg'] = np.mean([metrics[f'MAE_{n}'] for n in param_names])
    metrics['RMSE_avg'] = np.mean([metrics[f'RMSE_{n}'] for n in param_names])
    
    # Multi-dim NRMSE (Paper definition)
    metrics['NRMSE_multi'] = calculate_multidim_nrmse(targets, preds)
    
    # Per-parameter NRMSE - 使用與 Multi-dim 相同的歸一化因子
    y_mean_vector = np.mean(targets, axis=0)
    norm_y_mean = np.linalg.norm(y_mean_vector)
    
    for i, name in enumerate(param_names):
        metrics[f'NRMSE_{name}'] = metrics[f'RMSE_{name}'] / norm_y_mean if norm_y_mean != 0 else 0.0
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Verify Model Performance")
    parser.add_argument('--model-type', type=str, default=DEFAULT_MODEL_TYPE, choices=['CNN', 'MLP'], help='Model architecture')
    parser.add_argument('--loss-type', type=str, default=DEFAULT_LOSS_TYPE, help='Loss type (pure, physical, surrogate-{num})')
    parser.add_argument('--num', type=int, default=DEFAULT_NUM, help='Training run number (default: latest)')
    parser.add_argument('--suffix', type=str, default=DEFAULT_SUFFIX, choices=['best', 'last'], help='Model suffix')
    parser.add_argument('--model-path', type=str, default=None, help='Direct path to model (overrides other args)')
    
    args = parser.parse_args()

    # Resolve Model Path
    if args.model_path:
        model_path = args.model_path
    else:
        ckpt_dir = "checkpoint"
        # Pattern: {model_type}_{loss_type}_{num}_{wandb_name}_{suffix}.pth
        # Regex: {args.model_type}_{args.loss_type}_(?P<num>\d+)_.*_{args.suffix}\.pth
        pattern_regex = f"{args.model_type}_{args.loss_type}_(?P<num>\\d+)_.*_{args.suffix}\\.pth"
        
        model_path = find_model_path(ckpt_dir, pattern_regex, version=args.num)
        
        if model_path is None:
            print(f"❌ Could not find model matching: {pattern_regex} (Num: {args.num if args.num else 'Latest'})")
            sys.exit(1)

    print(f"🔍 Verifying model: {model_path}")
    print(f"🏗️  Architecture: {args.model_type}")

    # 1. Load Data
    print(f"📦 Loading dataset from {config.NPZ_PATH}...")
    dataset = TuringDataset(config.NPZ_PATH)
    
    # Split 1: Train+Valid (80%) vs Test (20%)
    # 確保與 train.py 使用完全相同的分割邏輯與種子
    total_size = len(dataset)
    test_size = int(total_size * config.TEST_RATIO)
    train_val_size = total_size - test_size
    
    # 我們只需要 Test Set，前面的 Train+Valid 部分直接忽略 (_)
    _, test_set = random_split(
        dataset, [train_val_size, test_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    loader = DataLoader(test_set, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"📊 Test set size: {len(test_set)}")

    # 2. Load Model
    if args.model_type == "CNN":
        model = ParameterNet().to(config.DEVICE)
    elif args.model_type == "MLP":
        model = MLPNet().to(config.DEVICE)
    
    if not os.path.exists(model_path):
        print(f"❌ Model file not found: {model_path}")
        sys.exit(1)
        
    try:
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        sys.exit(1)

    model.eval()

    # 3. Inference
    all_preds_norm = []
    all_targets_norm = []
    all_preds_real = []
    all_targets_real = []
    
    print("🚀 Running inference...")
    with torch.no_grad():
        for u_batch, _, params_target in tqdm(loader):
            u_batch = u_batch.to(config.DEVICE)
            
            # Predict
            preds_norm = model(u_batch)
            
            # Convert to numpy
            preds_norm_np = preds_norm.cpu().numpy()
            targets_norm_np = params_target.numpy()
            
            # Store normalized values (for metrics calculation)
            all_preds_norm.append(preds_norm_np)
            all_targets_norm.append(targets_norm_np)
            
            # Inverse transform for display
            preds_real = dataset.scaler.inverse_transform_numpy(preds_norm_np)
            targets_real = dataset.scaler.inverse_transform_numpy(targets_norm_np)
            
            all_preds_real.append(preds_real)
            all_targets_real.append(targets_real)

    all_preds_norm = np.vstack(all_preds_norm)
    all_targets_norm = np.vstack(all_targets_norm)
    all_preds_real = np.vstack(all_preds_real)
    all_targets_real = np.vstack(all_targets_real)

    # 4. Calculate Metrics (在歸一化空間計算，與訓練時一致)
    metrics = calculate_metrics(all_targets_norm, all_preds_norm)

    # 5. Report
    print("\n" + "="*50)
    print("📊 Evaluation Results (Normalized Space [0,1])")
    print("="*50)
    
    print(f"{'Metric':<15} {'Value':<10}")
    print("-" * 30)
    print(f"{'NRMSE (Multi)':<15} {metrics['NRMSE_multi']:.2%}")
    print(f"{'R2 (Avg)':<15} {metrics['R2_avg']:.4f}")
    print(f"{'MAE (Avg)':<15} {metrics['MAE_avg']:.4f}")
    print(f"{'RMSE (Avg)':<15} {metrics['RMSE_avg']:.4f}")
    print("-" * 30)
    
    print("\nDetailed Metrics per Parameter (Normalized):")
    print(f"{'Param':<10} {'R2':<10} {'MAE':<10} {'RMSE':<10} {'NRMSE':<10}")
    print("-" * 55)
    for name in ['a', 'b', 'c', 'delta']:
        print(f"{name:<10} {metrics[f'R2_{name}']:<10.4f} {metrics[f'MAE_{name}']:<10.4f} {metrics[f'RMSE_{name}']:<10.4f} {metrics[f'NRMSE_{name}']:<10.2%}")
    print("="*50)

if __name__ == "__main__":
    main()
