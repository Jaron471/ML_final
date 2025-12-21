import os
import sys
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

# ==========================
# 0. 環境與路徑設定
# ==========================
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

try:
    from model import config
    from model.dataset import TuringDataset
except ImportError:
    print("❌ 錯誤: 找不到 'model' 模組。請確保你的專案目錄結構正確。")
    sys.exit(1)

# ==========================================
# 1. 模型定義
# ==========================================
class PaperMinimalCNN(nn.Module):
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        out_dim = input_size - np_size + 1
        self.flat_features = nk * out_dim * out_dim
        self.fc1 = nn.Linear(self.flat_features, nf)
        self.fc_out = nn.Linear(nf, 4)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1) 
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return torch.sigmoid(x)

class PaperMinimalCNN2(nn.Module):
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN2, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=nk, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        out_dim = input_size - 2 * np_size + 2
        self.flat_features = nk * out_dim * out_dim
        self.fc1 = nn.Linear(self.flat_features, nf)
        self.fc_out = nn.Linear(nf, 4) 

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return torch.sigmoid(x)

class PaperMinimalCNN2MaxPool(nn.Module):
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN2MaxPool, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=nk, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Calculate output dimension
        d1 = input_size - np_size + 1
        d2 = d1 // 2
        d3 = d2 - np_size + 1
        d4 = d3 // 2
        self.flat_features = nk * d4 * d4
        
        self.fc1 = nn.Linear(self.flat_features, nf)
        self.fc_out = nn.Linear(nf, 4) 

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return torch.sigmoid(x)

class PaperMinimalCNN2Stride(nn.Module):
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN2Stride, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=np_size, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=nk, out_channels=nk, kernel_size=np_size, stride=2, padding=0)
        
        # Calculate output dimension
        # After Conv1 (stride 2)
        d1 = (input_size - np_size) // 2 + 1
        # After Conv2 (stride 2)
        d2 = (d1 - np_size) // 2 + 1
        
        self.flat_features = nk * d2 * d2
        
        self.fc1 = nn.Linear(self.flat_features, nf)
        self.fc_out = nn.Linear(nf, 4) 

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return torch.sigmoid(x)

# ==========================================
# 2. 多元評估指標計算函式
# ==========================================
def calculate_metrics(targets_norm, preds_norm, scaler=None):
    metrics = {}
    param_names = ['a', 'b', 'c', 'delta']
    epsilon = 1e-8
    
    # 準備物理數值用於計算 MAPE (如果 scaler 存在)
    if scaler:
        targets_real = scaler.inverse_transform_numpy(targets_norm)
        preds_real = scaler.inverse_transform_numpy(preds_norm)
    else:
        # 如果沒有 scaler，只好用 normalized 數值 (但不建議用於 MAPE)
        targets_real = targets_norm
        preds_real = preds_norm

    # 個別參數計算
    for i, name in enumerate(param_names):
        y_true = targets_norm[:, i]
        y_pred = preds_norm[:, i]
        
        # NRMSE (維持在 Normalized Space，符合論文定義)
        metrics[f'NRMSE_{name}'] = np.sqrt(np.mean((y_true - y_pred)**2))
        
        # R2 Score (維持在 Normalized Space，R2 是相對指標沒差)
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        if ss_tot == 0: ss_tot = epsilon
        metrics[f'R2_{name}'] = 1 - (ss_res / ss_tot)
        
        # MAPE (改用 Physical Space)
        # 使用真實物理數值計算，避免 0 值問題並具有物理意義
        y_true_real = targets_real[:, i]
        y_pred_real = preds_real[:, i]
        metrics[f'MAPE_{name}'] = np.mean(np.abs((y_true_real - y_pred_real) / (y_true_real + epsilon))) * 100

    # 整體指標
    # 1. Joint NRMSE
    num_elements = targets_norm.size
    total_sse = np.sum((targets_norm - preds_norm)**2)
    global_rmse = np.sqrt(total_sse / num_elements)
    y_mean_vec = np.mean(targets_norm, axis=0)
    norm_y_mean = np.linalg.norm(y_mean_vec)
    metrics['NRMSE_Joint'] = global_rmse / norm_y_mean if norm_y_mean != 0 else 0.0

    # 2. Mean R2
    metrics['R2_Mean'] = np.mean([metrics[f'R2_{n}'] for n in param_names])
    
    # 3. Mean MAPE
    metrics['MAPE_Mean'] = np.mean([metrics[f'MAPE_{n}'] for n in param_names])
    
    # 4. Max Error
    metrics['Max_Error'] = np.max(np.abs(targets_norm - preds_norm))

    return metrics

# ==========================================
# 3. 輔助函式：生成單一指標報表
# ==========================================
def generate_metric_report(comparison_results, metric_key, title, unit="", higher_is_better=False):
    """
    通用報表生成器
    """
    print("\n" + "="*80)
    print(f"📊 {title}")
    print("="*80)
    
    table_data = []
    
    for frac in sorted(comparison_results.keys(), reverse=True):
        res = comparison_results[frac]
        pure_val = res.get('pure', {}).get(metric_key, np.nan)
        phys_val = res.get('physical', {}).get(metric_key, np.nan)
        
        row = {'Data %': f"{frac:.1%}"}
        
        # 格式化數值
        if unit == "%":
            # NRMSE 等本來就是小數，需要轉百分比顯示
            # 但如果 metric 本身已经是百分比(如MAPE)，則直接顯示
            # 這裡假設 NRMSE_Joint 是小數，需要 :.2%
            # MAPE_Mean 是數值(0-100)，需要 :.2f
            if "NRMSE" in metric_key:
                row['Pure'] = f"{pure_val:.2%}" if not np.isnan(pure_val) else "-"
                row['Phys'] = f"{phys_val:.2%}" if not np.isnan(phys_val) else "-"
            else:
                row['Pure'] = f"{pure_val:.2f}" if not np.isnan(pure_val) else "-"
                row['Phys'] = f"{phys_val:.2f}" if not np.isnan(phys_val) else "-"
        else:
            row['Pure'] = f"{pure_val:.4f}" if not np.isnan(pure_val) else "-"
            row['Phys'] = f"{phys_val:.4f}" if not np.isnan(phys_val) else "-"
            
        # 計算差異與勝負
        if not np.isnan(pure_val) and not np.isnan(phys_val):
            if higher_is_better:
                # 如 R2: 越高越好，差異 = Phys - Pure
                diff = phys_val - pure_val
                row['Diff'] = f"{diff:+.4f}"
                row['Winner'] = "Physical 🟢" if diff > 0 else "Pure 🔴"
            else:
                # 如 Error: 越低越好，改善率 = (Pure - Phys) / Pure
                diff = pure_val - phys_val
                imp = (diff / pure_val) * 100
                row['Imp(%)'] = f"{imp:+.2f}%"
                row['Winner'] = "Physical 🟢" if diff > 0 else "Pure 🔴"
        else:
            # Use consistent column names based on metric type
            if higher_is_better:
                row['Diff'] = "-"
            else:
                row['Imp(%)'] = "-"
            row['Winner'] = "-"
            
        table_data.append(row)
        
    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))

# ==========================================
# 4. 主程式流程
# ==========================================
def main():
    TEST_PATH = "test_data.npz"
    CKPT_DIR = "paper_checkpoints"
    
    model_files = [
        "PaperPINN_physical_nk5_frac1.0_7_best_11.72.pth",
        "PaperPINN_physical_nk5_frac0.6_1_best_12.88.pth",
        "PaperPINN_physical_nk5_frac0.375_28_best.pth",
        "PaperPINN_physical_nk5_frac0.25_3_best_16.91.pth",
        "PaperPINN_physical_nk5_frac0.1875_16_best_16.62.pth",
        "PaperPINN_physical_nk5_frac0.125_5_best_18.49.pth",
        "PaperPINN_physical_nk5_frac0.0125_9_best_19.50.pth",
        "PaperPINN_pure_nk5_frac1.0_8_best_11.89.pth",
        "PaperPINN_pure_nk5_frac0.6_2_best_13.06.pth",
        "PaperPINN_pure_nk5_frac0.375_21_best_14.01.pth",
        "PaperPINN_pure_nk5_frac0.25_4_best_17.92.pth",
        "PaperPINN_pure_nk5_frac0.1875_17_best_18.55.pth",
        "PaperPINN_pure_nk5_frac0.125_6_best_18.68.pth",
        "PaperPINN_pure_nk5_frac0.0125_10_best_19.62.pth",
        "PaperPINN_cnn2_physical_nk5_frac0.25_53_best.pth",
        "PaperPINN_cnn2_pure_nk5_frac0.25_58_best.pth",
        "PaperPINN_cnn2pool_physical_nk5_frac0.25_54_best.pth",
        "PaperPINN_cnn2pool_pure_nk5_frac0.25_57_best.pth",
        "PaperPINN_cnn2stride_physical_nk5_frac0.25_55_best.pth",
        "PaperPINN_cnn2stride_pure_nk5_frac0.25_56_best.pth",
        "PaperPINN_physical_nk5_frac0.375_28_best.pth",
        "PaperPINN_pure_nk5_frac0.375_29_best.pth",
        "PaperPINN_physical_nk5_frac1.0_25_best_11.77_Rmask.pth",
    ]

    print(f"📦 Loading Test Data from: {TEST_PATH}")
    if not os.path.exists(TEST_PATH):
        if hasattr(config, 'NPZ_PATH') and os.path.exists(config.NPZ_PATH):
            TEST_PATH = config.NPZ_PATH
        else:
            print(f"❌ Error: Cannot find test data file.")
            return
            
    dataset = TuringDataset(TEST_PATH)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Structure: results[model_arch][frac][loss_type] = metrics
    comparison_results = {}

    print("\n🚀 Starting Evaluation...")
    
    # Special tracking for Rmask comparison
    rmask_comparison = {}
    
    for filename in sorted(model_files):
        filepath = os.path.join(CKPT_DIR, filename)
        
        # Check for Rmask variant
        has_rmask = '_Rmask' in filename
        
        # Try new format first: PaperPINN_{model_type}_{loss_type}_nk...
        match_new = re.search(r'PaperPINN_(?P<model>[a-zA-Z0-9]+)_(?P<loss>pure|physical)_nk(?P<nk>\d+)_frac(?P<frac>[\d\.]+)_', filename)
        
        # Try old format: PaperPINN_{loss_type}_nk...
        match_old = re.search(r'PaperPINN_(?P<loss>pure|physical)_nk(?P<nk>\d+)_frac(?P<frac>[\d\.]+)_', filename)
        
        if match_new:
            model_arch = match_new.group('model')
            loss_type = match_new.group('loss')
            frac = float(match_new.group('frac'))
        elif match_old:
            model_arch = 'cnn1' # Default to original
            loss_type = match_old.group('loss')
            frac = float(match_old.group('frac'))
        else:
            continue
            
        print(f"Processing: {filename} -> Arch: {model_arch}, Loss: {loss_type}, Frac: {frac}")
        
        # Check file existence
        if not os.path.exists(filepath):
            print(f"⚠️ File not found: {filepath}, skipping.")
            continue

        # Instantiate correct model
        if model_arch == 'cnn1':
            model = PaperMinimalCNN(nk=5, np_size=5, nf=5).to(config.DEVICE)
        elif model_arch == 'cnn2':
            model = PaperMinimalCNN2(nk=5, np_size=5, nf=5).to(config.DEVICE)
        elif model_arch == 'cnn2pool':
            model = PaperMinimalCNN2MaxPool(nk=5, np_size=5, nf=5).to(config.DEVICE)
        elif model_arch == 'cnn2stride':
            model = PaperMinimalCNN2Stride(nk=5, np_size=5, nf=5).to(config.DEVICE)
        else:
            print(f"⚠️ Unknown model architecture: {model_arch}, skipping.")
            continue

        try:
            model.load_state_dict(torch.load(filepath, map_location=config.DEVICE, weights_only=False))
        except Exception as e:
            print(f"❌ Failed to load {filename}: {e}")
            continue
            
        model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for u_batch, _, params_target in loader:
                u_batch = u_batch.to(config.DEVICE)
                u_input = u_batch[:, 0:1, :, :] if u_batch.shape[1] > 1 else u_batch
                preds = model(u_input)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(params_target.numpy())
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        metrics = calculate_metrics(all_targets, all_preds, scaler=dataset.scaler)
        
        if model_arch not in comparison_results: comparison_results[model_arch] = {}
        if frac not in comparison_results[model_arch]: comparison_results[model_arch][frac] = {}
        comparison_results[model_arch][frac][loss_type] = metrics
        
        # Track Rmask comparison for frac=1.0 physical models
        # Note: Rmask version = No mask, Original = With mask
        if has_rmask and loss_type == 'physical' and frac == 1.0:
            rmask_comparison['no_mask'] = metrics
        elif not has_rmask and loss_type == 'physical' and frac == 1.0 and model_arch == 'cnn1':
            rmask_comparison['with_mask'] = metrics

    # Generate reports for each model architecture
    for model_arch, results in comparison_results.items():
        print("\n" + "#"*80)
        print(f"🏆 FINAL COMPARISON REPORT: {model_arch}")
        print("#"*80)

        # 1. NRMSE Table
        generate_metric_report(
            results, 
            metric_key='NRMSE_Joint', 
            title=f"[{model_arch}] Joint NRMSE Comparison (Lower is Better)", 
            unit="%", 
            higher_is_better=False
        )

        # 2. R2 Score Table
        generate_metric_report(
            results, 
            metric_key='R2_Mean', 
            title=f"[{model_arch}] Mean R² Score Comparison (Higher is Better)", 
            unit="", 
            higher_is_better=True
        )

    # Special Rmask Comparison Report
    if len(rmask_comparison) == 2:
        print("\n" + "#"*80)
        print("🔬 SPECIAL COMPARISON: Mask Effect (frac=1.0, Physical Loss)")
        print("#"*80)
        
        metrics_to_compare = [
            ('NRMSE_Joint', 'Joint NRMSE', '%', False),
        ]
        
        print("\n" + "="*80)
        print("📊 With-Mask vs No-Mask Comparison")
        print("="*80)
        
        table_data = []
        for metric_key, metric_name, unit, higher_is_better in metrics_to_compare:
            with_mask = rmask_comparison['with_mask'].get(metric_key, np.nan)
            no_mask = rmask_comparison['no_mask'].get(metric_key, np.nan)
            
            row = {'Metric': metric_name}
            
            # Format values
            if unit == '%' and 'NRMSE' in metric_key:
                row['With-Mask'] = f"{with_mask:.2%}" if not np.isnan(with_mask) else "-"
                row['No-Mask'] = f"{no_mask:.2%}" if not np.isnan(no_mask) else "-"
            elif unit == '%':
                row['With-Mask'] = f"{with_mask:.2f}" if not np.isnan(with_mask) else "-"
                row['No-Mask'] = f"{no_mask:.2f}" if not np.isnan(no_mask) else "-"
            else:
                row['With-Mask'] = f"{with_mask:.4f}" if not np.isnan(with_mask) else "-"
                row['No-Mask'] = f"{no_mask:.4f}" if not np.isnan(no_mask) else "-"
            
            # Calculate improvement
            if not np.isnan(with_mask) and not np.isnan(no_mask):
                if higher_is_better:
                    diff = no_mask - with_mask
                    row['Diff'] = f"{diff:+.4f}"
                    row['Winner'] = "No-Mask 🔴" if diff > 0 else "With-Mask 🟢"
                else:
                    diff = with_mask - no_mask
                    imp = (diff / with_mask) * 100
                    row['Imp(%)'] = f"{imp:+.2f}%"
                    row['Winner'] = "No-Mask 🔴" if diff > 0 else "With-Mask 🟢"
            else:
                if higher_is_better:
                    row['Diff'] = "-"
                else:
                    row['Imp(%)'] = "-"
                row['Winner'] = "-"
            
            table_data.append(row)
        
        df = pd.DataFrame(table_data)
        print(df.to_string(index=False))
    
    print("\n" + "#"*80)
    print("Done.")

if __name__ == "__main__":
    main()