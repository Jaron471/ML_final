import subprocess
import os
import sys
import time
import torch
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import r2_score

from model import config
from model.dataset import TuringDataset
from model.model import PaperFlexibleCNN, MLPNet

# Configurations
SAMPLES = [1000]
LOSS_CONFIGS = [
    # (use_loss, lambda_phy)
    ('pure', 0.0),          # Pure Supervised
    ('physical', 0.1),      # Physical with lambda=0.1
    ('physical', 0.01),    # Physical with lambda=0.01
    ('physical', 0.001),   # Physical with lambda=0.001
    ('physical', 0.0001),  # Physical with lambda=0.0001
    ('physical', 0.00001)  # Physical with lambda=0.00001
]

# Architectures to test
# (layers, sampling)
ARCHITECTURES = [
    #(0, 'mlp'),       # MLP
    (1, 'none'),      # Like CNN1
    (2, 'none'),      # Like CNN2
    (2, 'maxpool'),   # Like CNN2Pool
    (2, 'dilated'),   # Like CNN2 but with dilated conv (New)
    (4, 'none'),      # Deeper without sampling
    (4, 'maxpool'),   # Deeper with MaxPool
    (4, 'dilated'),   # Deeper with Dilated Conv (New)
]

def get_latest_checkpoint(ckpt_dir, pattern_regex):
    """
    Scans directory for files matching regex and returns the path to the one with largest version number.
    Assumption: version group is named 'num'
    """
    max_num = -1
    best_file = None
    
    if not os.path.exists(ckpt_dir):
        return None
        
    for filename in os.listdir(ckpt_dir):
        match = re.match(pattern_regex, filename)
        if match and 'best' in filename:
            num = int(match.group('num'))
            if num > max_num:
                max_num = num
                best_file = filename
                
    if best_file:
        return os.path.join(ckpt_dir, best_file)
    return None

def evaluate_model(config_dict, model_path, dataset_loader):
    """
    Evaluates the model and appends results to CSV.
    """
    print(f"📊 Evaluating: {os.path.basename(model_path)}")
    
    # 1. Build Model
    if config_dict['layers'] == 0:
        model = MLPNet().to(config.DEVICE)
    else:
        model = PaperFlexibleCNN(
            nk=5, np_size=5, nf=5,
            layers=config_dict['layers'],
            sampling=config_dict['sampling']
        ).to(config.DEVICE)
    
    # 2. Load Weights
    try:
        model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
        model.eval()
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # 3. Inference
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for u, _, p in dataset_loader:
            u_input = u.to(config.DEVICE)
            # Ensure input shape [B, 1, 128, 128]
            if u_input.dim() == 3:
                u_input = u_input.unsqueeze(1)
            elif u_input.shape[1] > 1: # If [B, 2, 128, 128]
                u_input = u_input[:, 0:1]
                
            preds = model(u_input)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(p.numpy())

    preds = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    # 4. Metrics
    # NRMSE
    mse = np.mean((preds - targets) ** 2)
    rmse = np.sqrt(mse)
    norm = np.linalg.norm(np.mean(targets, axis=0))
    nrmse = rmse / norm if norm > 0 else 0
    
    # R^2 Score
    r2 = r2_score(targets, preds, multioutput='uniform_average')

    # 5. Save Results
    results_file = "flexible_experiments_results.csv"
    
    # Architecture description
    arch_desc = f"L{config_dict['layers']}_{config_dict['sampling']}"
    
    result_row = {
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "Architecture": arch_desc,
        "Layers": config_dict['layers'],
        "Sampling": config_dict['sampling'],
        "Samples": config_dict['samples'],
        "LossType": config_dict['loss_type'],
        "Lambda": config_dict['lambda'],
        "NRMSE": nrmse,
        "R2": r2,
        "ModelFile": os.path.basename(model_path)
    }
    
    df = pd.DataFrame([result_row])
    
    # Append to CSV
    if not os.path.isfile(results_file):
        df.to_csv(results_file, index=False)
    else:
        df.to_csv(results_file, mode='a', header=False, index=False)
        
    print(f"   Done. NRMSE={nrmse:.4f}, R2={r2:.4f}")

def run_training():
    total_runs = len(SAMPLES) * len(LOSS_CONFIGS) * len(ARCHITECTURES)
    current_run = 0
    
    # Prepare Evaluation Dataset Once
    print("⏳ Loading Test Dataset for Evaluation...")
    # Use path from config
    test_path = config.TEST_PATH
    dataset = TuringDataset(test_path, val_split=0) # Set val_split=0 for full test set
    test_loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"✅ Dataset Loaded from {test_path}")
    
    print(f"🚀 Starting Batch Training: {total_runs} total configurations")
    print("=" * 60)

    for nsamp in SAMPLES:
        for layers, sampling in ARCHITECTURES:
            for use_loss, lam in LOSS_CONFIGS:
                current_run += 1
                
                arch_name = "MLP" if layers == 0 else f"Flexible CNN (L={layers}, {sampling})"
                print(f"\n[{current_run}/{total_runs}] Training Config:")
                print(f"  Architecture: {arch_name}")
                print(f"  Samples:      {nsamp}")
                print(f"  Loss:         {use_loss} (lambda={lam})")
                
                if layers == 0:
                    cmd = [
                        sys.executable, "train_unified.py",
                        "--model-arch", "mlp",
                        "--num-samples", str(nsamp),
                        "--use-loss", use_loss,
                        "--lambda-phy", str(lam)
                    ]
                else:
                    cmd = [
                        sys.executable, "train_unified.py",
                        "--model-arch", "flexible_cnn",
                        "--conv-layers", str(layers),
                        "--sampling", sampling,
                        "--num-samples", str(nsamp),
                        "--use-loss", use_loss,
                        "--lambda-phy", str(lam)
                    ]
                
                # Run the command
                start_time = time.time()
                try:
                    subprocess.run(cmd, check=True)
                except subprocess.CalledProcessError as e:
                    print(f"❌ Error training configuration!")
                    # Continue to next
                    continue
                
                duration = time.time() - start_time
                print(f"✅ Finished in {duration:.1f} seconds")
                
                # --- Evaluation Step ---
                # Reconstruct the expected filename pattern to find the file
                # Pattern Logic from train_unified.py:
                # PaperPINN_CNN_Flex_L{layers}_{sampling}_{loss_tag}_n{nsamp}_{num}_*.pth
                
                loss_tag = use_loss
                if use_loss == 'physical':
                    loss_tag = f"{use_loss}_lam{lam}"
                
                # Note: Regex requires escaping special chars if necessary, but here strings are alphanumeric mostly
                if layers == 0:
                     pattern = f"PaperPINN_MLP_{loss_tag}_n{nsamp}_(?P<num>\\d+)_.*\\.pth"
                else:
                     pattern = f"PaperPINN_CNN_Flex_L{layers}_{sampling}_{loss_tag}_n{nsamp}_(?P<num>\\d+)_.*\\.pth"
                
                ckpt_dir = "paper_checkpoints"
                latest_model = get_latest_checkpoint(ckpt_dir, pattern)
                
                if latest_model:
                    eval_config = {
                        'layers': layers,
                        'sampling': sampling,
                        'samples': nsamp,
                        'loss_type': use_loss,
                        'lambda': lam
                    }
                    evaluate_model(eval_config, latest_model, test_loader)
                else:
                    print(f"⚠️ Could not find checkpoint matching: {pattern}")
                
                # Optional: Small cooldown
                time.sleep(1)

    print("\n" + "=" * 60)
    print("🎉 All experiments completed! Results saved to flexible_experiments_results.csv")

if __name__ == "__main__":
    run_training()
