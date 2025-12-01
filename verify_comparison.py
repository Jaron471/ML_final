import os
import sys
import re
import torch
import torch.nn as nn
import numpy as np
import pandas as pd  # 用於漂亮的表格輸出
from torch.utils.data import DataLoader
from tqdm import tqdm

# 環境設定
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

# 引用你的模組
from model import config
from model.dataset import TuringDataset

# ==========================================
# 1. 定義論文中的極簡 CNN 架構 (必須與訓練時一致)
# ==========================================
class PaperMinimalCNN(nn.Module):
    """
    實作論文中的極簡 CNN (nk/np/nf)
    """
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
# 2. 評估指標計算 (符合論文定義)
# ==========================================
def calculate_metrics(targets_norm, preds_norm):
    """
    計算個別參數 NRMSE 與 論文定義的 Joint NRMSE
    """
    metrics = {}
    param_names = ['a', 'b', 'c', 'delta']
    
    # 1. 個別參數 NRMSE (基於正規化數據 [0,1], Range=1, 所以 RMSE=NRMSE)
    for i, name in enumerate(param_names):
        mse = np.mean((targets_norm[:, i] - preds_norm[:, i])**2)
        rmse = np.sqrt(mse)
        metrics[f'NRMSE_{name}'] = rmse

    # 2. Joint NRMSE (Paper Definition )
    # 公式: RMSE_total / || mean(y_target) ||
    num_elements = targets_norm.size
    total_sse = np.sum((targets_norm - preds_norm)**2)
    global_rmse = np.sqrt(total_sse / num_elements)
    
    # 分母：目標向量平均值的歐幾里得範數
    y_mean_vec = np.mean(targets_norm, axis=0)
    norm_y_mean = np.linalg.norm(y_mean_vec)
    
    metrics['NRMSE_Joint'] = global_rmse / norm_y_mean if norm_y_mean != 0 else 0.0
    
    return metrics

def main():
    # ==========================
    # 1. 設定檔案清單與路徑
    # ==========================
    TEST_PATH = "test_data.npz"  # 請確認你的測試檔案名稱
    CKPT_DIR = "paper_checkpoints" # 模型存放目錄
    
    # 你提供的模型清單
    model_files = [
        "PaperPINN_physical_nk5_frac0.6_1_best_12.88.pth",
        "PaperPINN_physical_nk5_frac0.25_3_best_16.91.pth",
        "PaperPINN_physical_nk5_frac0.125_5_best_18.49.pth",
        "PaperPINN_physical_nk5_frac1.0_7_best_11.72.pth",
        "PaperPINN_physical_nk5_frac0.0125_9_best_19.50.pth",
        "PaperPINN_pure_nk5_frac0.6_2_best_13.06.pth",
        "PaperPINN_pure_nk5_frac0.25_4_best_17.92.pth",
        "PaperPINN_pure_nk5_frac0.125_6_best_18.68.pth",
        "PaperPINN_pure_nk5_frac1.0_8_best_11.89.pth",
        "PaperPINN_pure_nk5_frac0.0125_10_best_19.62.pth"
    ]

    # ==========================
    # 2. 載入測試資料
    # ==========================
    print(f"📦 Loading Test Data from: {TEST_PATH}")
    if not os.path.exists(TEST_PATH):
        print(f"❌ Error: {TEST_PATH} not found. Trying config.NPZ_PATH...")
        TEST_PATH = config.NPZ_PATH
        
    dataset = TuringDataset(TEST_PATH)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    print(f"📊 Test Set Size: {len(dataset)}")

    # 用於儲存結果的字典: results[frac][type] = metrics
    comparison_results = {}

    # ==========================
    # 3. 批量評估迴圈
    # ==========================
    print("\n🚀 Starting Batch Evaluation...")
    
    for filename in model_files:
        filepath = os.path.join(CKPT_DIR, filename)
        if not os.path.exists(filepath):
            # 嘗試在當前目錄尋找
            filepath = filename
            if not os.path.exists(filepath):
                print(f"⚠️  Skipping missing file: {filename}")
                continue

        # 解析檔名資訊 (Regex)
        # 格式: PaperPINN_{type}_nk5_frac{frac}_...
        match = re.search(r'PaperPINN_(?P<type>[a-zA-Z]+)_nk5_frac(?P<frac>[\d\.]+)_', filename)
        if not match:
            print(f"⚠️  Cannot parse filename info: {filename}")
            continue
            
        model_type = match.group('type') # physical / pure
        frac = float(match.group('frac'))
        
        print(f"\n🔍 Evaluating: {filename}")
        print(f"   Type: {model_type}, Fraction: {frac}")

        # 初始化模型 (nk=5, np=5, nf=5)
        model = PaperMinimalCNN(nk=5, np_size=5, nf=5).to(config.DEVICE)
        
        # 載入權重
        try:
            model.load_state_dict(torch.load(filepath, map_location=config.DEVICE))
        except Exception as e:
            print(f"❌ Load failed: {e}")
            continue
            
        model.eval()

        # 推論
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for u_batch, _, params_target in loader:
                u_batch = u_batch.to(config.DEVICE)
                
                # 處理輸入 Channel
                if u_batch.shape[1] > 1:
                    u_input = u_batch[:, 0:1, :, :]
                else:
                    u_input = u_batch
                
                preds = model(u_input)
                
                all_preds.append(preds.cpu().numpy())
                all_targets.append(params_target.numpy())
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)

        # 計算指標
        metrics = calculate_metrics(all_targets, all_preds)
        
        # 儲存結果
        if frac not in comparison_results:
            comparison_results[frac] = {}
        comparison_results[frac][model_type] = metrics
        
        # 印出單一模型結果
        print(f"   Joint NRMSE: {metrics['NRMSE_Joint']:.2%}")
        print(f"   Per-param NRMSE: a={metrics['NRMSE_a']:.2%}, b={metrics['NRMSE_b']:.2%}, c={metrics['NRMSE_c']:.2%}, δ={metrics['NRMSE_delta']:.2%}")

    # ==========================
    # 4. 生成比較報表
    # ==========================
    print("\n" + "="*80)
    print("🏆 FINAL COMPARISON REPORT: Pure vs. Physical (PINN)")
    print("="*80)
    
    # 準備 DataFrame 資料
    table_data = []
    
    # 依資料比例排序 (大 -> 小)
    for frac in sorted(comparison_results.keys(), reverse=True):
        row = {'Data Fraction': f"{frac:.1%}"}
        
        res = comparison_results[frac]
        
        # 取得 Joint NRMSE
        pure_score = res.get('pure', {}).get('NRMSE_Joint', np.nan)
        phys_score = res.get('physical', {}).get('NRMSE_Joint', np.nan)
        
        row['Pure NRMSE'] = f"{pure_score:.2%}" if not np.isnan(pure_score) else "N/A"
        row['Physical NRMSE'] = f"{phys_score:.2%}" if not np.isnan(phys_score) else "N/A"
        
        # 計算差異
        if not np.isnan(pure_score) and not np.isnan(phys_score):
            diff = pure_score - phys_score
            # 正值代表 Physical 比較好 (Pure 誤差較大)
            improvement = (diff / pure_score) * 100
            row['Improvement'] = f"{improvement:+.2f}%"
            row['Winner'] = "Physical 🟢" if diff > 0 else "Pure 🔴"
        else:
            row['Improvement'] = "-"
            row['Winner'] = "-"
            
        table_data.append(row)

    # 輸出表格
    df = pd.DataFrame(table_data)
    print(df.to_string(index=False))
    print("="*80)
    print("註: 'Improvement' 為正值代表 Physical (PINN) 誤差較小，效能較好。")

if __name__ == "__main__":
    main()