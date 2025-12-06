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
            row['Diff/Imp'] = "-"
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
        "PaperPINN_physical_nk5_frac0.25_3_best_16.91.pth",
        "PaperPINN_physical_nk5_frac0.1875_16_best_16.62.pth",
        "PaperPINN_physical_nk5_frac0.125_5_best_18.49.pth",
        "PaperPINN_physical_nk5_frac0.0125_9_best_19.50.pth",
        "PaperPINN_pure_nk5_frac1.0_8_best_11.89.pth",
        "PaperPINN_pure_nk5_frac0.6_2_best_13.06.pth",
        "PaperPINN_pure_nk5_frac0.25_4_best_17.92.pth",
        "PaperPINN_pure_nk5_frac0.1875_17_best_18.55.pth",
        "PaperPINN_pure_nk5_frac0.125_6_best_18.68.pth",
        "PaperPINN_pure_nk5_frac0.0125_10_best_19.62.pth",
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
    
    comparison_results = {}

    print("\n🚀 Starting Evaluation...")
    
    # --- 推論迴圈 ---
    for filename in model_files:
        filepath = os.path.join(CKPT_DIR, filename)
        if not os.path.exists(filepath): 
            if os.path.exists(filename): filepath = filename
            else: continue

        match = re.search(r'PaperPINN_(?P<type>[a-zA-Z]+)_nk5_frac(?P<frac>[\d\.]+)_', filename)
        if not match: continue
            
        model_type = match.group('type')
        frac = float(match.group('frac'))
        
        # print(f"Processing {model_type} | Frac: {frac}") # 除錯用

        model = PaperMinimalCNN(nk=5, np_size=5, nf=5).to(config.DEVICE)
        try:
            model.load_state_dict(torch.load(filepath, map_location=config.DEVICE))
        except: continue
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
        
        # 傳入 scaler 以便在內部還原物理數值計算 MAPE
        metrics = calculate_metrics(all_targets, all_preds, scaler=dataset.scaler)
        
        if frac not in comparison_results: comparison_results[frac] = {}
        comparison_results[frac][model_type] = metrics

    # --- 4. 生成多個獨立報表 ---
    
    print("\n" + "#"*80)
    print("🏆 FINAL COMPARISON REPORT (Per Metric)")
    print("#"*80)

    # 1. NRMSE Table (越低越好)
    generate_metric_report(
        comparison_results, 
        metric_key='NRMSE_Joint', 
        title="Joint NRMSE Comparison (Lower is Better)", 
        unit="%", 
        higher_is_better=False
    )

    # 2. R2 Score Table (越高越好)
    generate_metric_report(
        comparison_results, 
        metric_key='R2_Mean', 
        title="Mean R² Score Comparison (Higher is Better)", 
        unit="", 
        higher_is_better=True
    )

    # 3. MAPE Table (越低越好)
    generate_metric_report(
        comparison_results, 
        metric_key='MAPE_Mean', 
        title="Mean MAPE Comparison (Lower is Better)", 
        unit="%", 
        higher_is_better=False
    )

    # 4. Max Error Table (越低越好)
    generate_metric_report(
        comparison_results, 
        metric_key='Max_Error', 
        title="Max Error Comparison (Lower is Better - Worst Case)", 
        unit="", 
        higher_is_better=False
    )

    print("\n" + "#"*80)
    print("Done.")

if __name__ == "__main__":
    main()