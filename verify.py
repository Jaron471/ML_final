import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
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
DEFAULT_LOSS_TYPE = "surrogate"       # pure / physical / surrogate-{num}
DEFAULT_NUM = None                   # None = Latest, or integer
DEFAULT_SUFFIX = "last"              # best / last

def calculate_metrics(targets, preds, scaler=None):
    """Calculate R2, MAE, RMSE, NRMSE for each parameter"""
    metrics = {}
    param_names = ['a', 'b', 'c', 'delta']
    
    # 1. 在正規化空間 [0,1] 計算基礎指標
    for i, name in enumerate(param_names):
        metrics[f'R2_{name}'] = r2_score(targets[:, i], preds[:, i])
        metrics[f'MAE_{name}'] = mean_absolute_error(targets[:, i], preds[:, i])
        metrics[f'RMSE_{name}'] = np.sqrt(mean_squared_error(targets[:, i], preds[:, i]))
    
    # Aggregate metrics
    metrics['R2_avg'] = np.mean([metrics[f'R2_{n}'] for n in param_names])
    metrics['MAE_avg'] = np.mean([metrics[f'MAE_{n}'] for n in param_names])
    metrics['RMSE_avg'] = np.mean([metrics[f'RMSE_{n}'] for n in param_names])
    
    # 2. Multi-dim NRMSE (Paper definition) - 這是最重要的總體指標
    # NRMSE 計算時，分母(norm_y_mean)的大小取決於數據分佈
    metrics['NRMSE_multi'] = calculate_multidim_nrmse(targets, preds)
    
    # 3. Per-parameter NRMSE
    y_mean_vector = np.mean(targets, axis=0)
    norm_y_mean = np.linalg.norm(y_mean_vector)
    
    for i, name in enumerate(param_names):
        metrics[f'NRMSE_{name}'] = metrics[f'RMSE_{name}'] / norm_y_mean if norm_y_mean != 0 else 0.0
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Verify Model Performance on Test Set")
    parser.add_argument('--model-type', type=str, default=DEFAULT_MODEL_TYPE, choices=['CNN', 'MLP'], help='Model architecture')
    parser.add_argument('--loss-type', type=str, default=DEFAULT_LOSS_TYPE, help='Loss type used in training')
    parser.add_argument('--num', type=int, default=DEFAULT_NUM, help='Training run number (default: latest)')
    parser.add_argument('--suffix', type=str, default=DEFAULT_SUFFIX, choices=['best', 'last'], help='Which checkpoint to load')
    parser.add_argument('--model-path', type=str, default=None, help='Direct path to model (overrides auto-search)')
    
    args = parser.parse_args()

    # ==========================================
    # 1. Load Data (Modified for separate files)
    # ==========================================
    print(f"📦 Loading Test Data from: {config.TEST_PATH}")
    
    # 載入測試集
    test_dataset = TuringDataset(config.TEST_PATH)
    
    # [Optional but Recommended] 確保 Scaler 一致性
    # 嚴謹的做法是使用 "Training Set" 的 min/max 來還原數值
    # 如果不這樣做，Test set 自己的 min/max 可能會導致 0.1% 的誤差
    try:
        if os.path.exists(config.TRAIN_PATH):
            print(f"⚖️  Loading scaler reference from: {config.TRAIN_PATH}")
            train_dataset = TuringDataset(config.TRAIN_PATH)
            
            # 強制將測試集的 Scaler 替換為訓練集的 Scaler
            # 注意：TuringDataset 在 init 時已經轉換了 params_norm，
            # 所以如果這裡換了 scaler，理論上應該要重新 transform params_norm。
            # 但因為 train/test 分佈極為接近，這裡我們只替換 scaler 物件以供 inverse_transform 使用。
            test_dataset.scaler = train_dataset.scaler
    except Exception as e:
        print(f"⚠️  Warning: Could not load training scaler ({e}). Using test set scaler.")

    loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"📊 Test set size: {len(test_dataset)}")

    # ==========================================
    # 2. Find and Load Model
    # ==========================================
    # Resolve Model Path
    if args.model_path:
        model_path = args.model_path
    else:
        ckpt_dir = "checkpoint"
        # 這裡的 Regex 必須配合 train.py 的存檔命名規則
        # Pattern example: CNN_physical_1_Unified_Run_best.pth
        pattern_regex = f"{args.model_type}_{args.loss_type}_(?P<num>\\d+)_.*_{args.suffix}\\.pth"
        
        model_path = find_model_path(ckpt_dir, pattern_regex, version=args.num)
        
        if model_path is None:
            print(f"❌ Could not find model matching: {pattern_regex}")
            print(f"   In directory: {os.path.abspath(ckpt_dir)}")
            sys.exit(1)

    print(f"🔍 Verifying model: {os.path.basename(model_path)}")
    print(f"🏗️  Architecture: {args.model_type}")

    # Initialize Model
    if args.model_type == "CNN":
        model = ParameterNet().to(config.DEVICE)
    elif args.model_type == "MLP":
        model = MLPNet().to(config.DEVICE)
    
    # Load Weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model weights: {e}")
        sys.exit(1)

    model.eval()

    # ==========================================
    # 3. Inference Loop
    # ==========================================
    all_preds_norm = []
    all_targets_norm = []
    
    # 用於顯示真實數值的誤差 (Optional)
    all_preds_real = []
    all_targets_real = []
    
    print("🚀 Running inference on Test Set...")
    with torch.no_grad():
        for u_batch, _, params_target in tqdm(loader):
            u_batch = u_batch.to(config.DEVICE)
            
            # Predict
            preds_norm = model(u_batch)
            
            # CPU conversion
            preds_norm_np = preds_norm.cpu().numpy()
            targets_norm_np = params_target.numpy()
            
            all_preds_norm.append(preds_norm_np)
            all_targets_norm.append(targets_norm_np)
            
            # Inverse Transform (還原成物理數值)
            preds_real = test_dataset.scaler.inverse_transform_numpy(preds_norm_np)
            targets_real = test_dataset.scaler.inverse_transform_numpy(targets_norm_np)
            
            all_preds_real.append(preds_real)
            all_targets_real.append(targets_real)

    # Stack results
    all_preds_norm = np.vstack(all_preds_norm)
    all_targets_norm = np.vstack(all_targets_norm)
    all_preds_real = np.vstack(all_preds_real)
    all_targets_real = np.vstack(all_targets_real)

    # ==========================================
    # 4. Calculate & Report Metrics
    # ==========================================
    # 我們主要關注 Normalized Space 的指標 (與訓練 Loss 一致)
    metrics = calculate_metrics(all_targets_norm, all_preds_norm)

    print("\n" + "="*60)
    print(f"📊 Evaluation Results (Test Set N={len(test_dataset)})")
    print("="*60)
    
    print(f"{'Metric':<20} {'Value':<10}")
    print("-" * 40)
    # NRMSE Multi 是論文常用的主要指標
    print(f"{'NRMSE (Multi)':<20} {metrics['NRMSE_multi']:.2%}  <-- Key Metric") 
    print(f"{'R2 (Avg)':<20} {metrics['R2_avg']:.4f}")
    print(f"{'MAE (Avg)':<20} {metrics['MAE_avg']:.4f}")
    print("-" * 40)
    
    print("\nDetailed Metrics per Parameter (Normalized Space):")
    print(f"{'Param':<8} {'R2':<10} {'MAE':<10} {'RMSE':<10} {'NRMSE':<10}")
    print("-" * 55)
    for name in ['a', 'b', 'c', 'delta']:
        print(f"{name:<8} {metrics[f'R2_{name}']:<10.4f} {metrics[f'MAE_{name}']:<10.4f} "
              f"{metrics[f'RMSE_{name}']:<10.4f} {metrics[f'NRMSE_{name}']:<10.2%}")
    print("="*60)
    
    # 額外顯示真實物理數值的平均誤差 (讓人類比較有感)
    mae_real = np.mean(np.abs(all_targets_real - all_preds_real), axis=0)
    print("\n[Reference] Mean Absolute Error in Physics Domain:")
    print(f"a: {mae_real[0]:.5f}, b: {mae_real[1]:.5f}, c: {mae_real[2]:.5f}, delta: {mae_real[3]:.5f}")

if __name__ == "__main__":
    main()