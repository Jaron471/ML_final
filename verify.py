import os
import sys

# --- 1. 解決 OMP 錯誤 (必須放在最上面) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 解決 Windows 中文路徑輸出編碼問題
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from tqdm import tqdm

# 引用你的模組
from model import config_new
from model.dataset import TuringDataset
# --- 修改這裡：同時引入 MLPNet ---
from model.model import ParameterNet, MLPNet 

def verify():
    # 1. 載入資料
    print(f"Loading model from {config_new.MODEL_SAVE_PATH}...")
    dataset = TuringDataset(config_new.NPZ_PATH)
    
    # 隨機取樣 1000 筆資料來驗證
    indices = np.random.choice(len(dataset), size=min(len(dataset), 1000), replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=config_new.BATCH_SIZE, shuffle=False)
    
    # --- 修改這裡：根據 Config 選擇模型架構 ---
    print(f"Using Model Architecture: {config_new.MODEL_TYPE}")
    if config_new.MODEL_TYPE == "CNN":
        model = ParameterNet().to(config_new.DEVICE)
    elif config_new.MODEL_TYPE == "MLP":
        model = MLPNet().to(config_new.DEVICE)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {config_new.MODEL_TYPE}")
    # ----------------------------------------

    # 載入訓練好的權重
    # 注意：如果權重檔名有變 (例如 MLP_PINN.pth)，請確保 config.MODEL_SAVE_PATH 指向正確檔案
    # 或者手動指定路徑: model.load_state_dict(torch.load("MLP_PINN.pth"))
    try:
        model.load_state_dict(torch.load("checkpoint/" + config_new.MODEL_SAVE_PATH))
    except FileNotFoundError:
        print(f"❌ 找不到權重檔: {config_new.MODEL_SAVE_PATH}")
        print("請確認 config.MODEL_TYPE 是否與訓練時一致，或手動修改路徑。")
        return

    model.eval()
    
    # 2. 進行預測
    all_preds = []
    all_targets = []
    
    print("Running inference...")
    with torch.no_grad():
        for u_batch, _, params_target in tqdm(loader):
            u_batch = u_batch.to(config_new.DEVICE)
            
            # 預測 (正規化後的)
            preds_norm = model(u_batch)
            
            # 反正規化回真實數值
            preds_norm = preds_norm.cpu().numpy()
            targets_norm = params_target.numpy()
            
            # 手動反正規化
            preds_real = dataset.scaler.inverse_transform_numpy(preds_norm)
            targets_real = dataset.scaler.inverse_transform_numpy(targets_norm)
            
            all_preds.append(preds_real)
            all_targets.append(targets_real)
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 3. 數值評估 (Metrics: R2, MAE, NRMSE)
    param_names = ['a', 'b', 'c', 'delta']
    print("\n" + "="*55)
    print(f"   Evaluation Metrics ({config_new.MODEL_TYPE})")
    print("="*55)
    print(f"{'Metric':<10} | {'a':<10} | {'b':<10} | {'c':<10} | {'delta':<10}")
    print("-" * 60)
    
    metrics = {'R2': [], 'MAE': [], 'NRMSE': []}
    
    for i, name in enumerate(param_names):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        # 計算 NRMSE
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        data_range = np.max(y_true) - np.min(y_true)
        nrmse = (rmse / data_range) if data_range != 0 else 0.0
        
        metrics['R2'].append(r2)
        metrics['MAE'].append(mae)
        metrics['NRMSE'].append(nrmse)

    # 顯示表格
    for name, vals in metrics.items():
        if name == 'NRMSE':
            row_str = f"{name:<10} | {vals[0]:.2%}   | {vals[1]:.2%}   | {vals[2]:.2%}   | {vals[3]:.2%}"
        else:
            row_str = f"{name:<10} | {vals[0]:.4f}   | {vals[1]:.4f}   | {vals[2]:.4f}   | {vals[3]:.4f}"
        print(row_str)
        
    print("="*60)

    # 4. 繪製散點圖 (Scatter Plots)
    plot_scatter(all_targets, all_preds, param_names)

def plot_scatter(y_true, y_pred, param_names):
    """
    繪製 2x2 的散點圖
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=10, c='blue', label='Samples')
        
        lims = [
            np.min([ax.get_xlim(), ax.get_ylim()]),
            np.max([ax.get_xlim(), ax.get_ylim()]),
        ]
        ax.plot(lims, lims, 'r-', alpha=0.75, linewidth=2, label='Ideal (y=x)')
        
        ax.set_xlabel(f"True {param_names[i]}")
        ax.set_ylabel(f"Predicted {param_names[i]}")
        ax.set_title(f"Parameter: {param_names[i]}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    filename = f"verify_{config_new.MODEL_TYPE}.png"
    plt.savefig(filename)
    print(f"\n✅ Scatter plots saved to '{filename}'")
    plt.show()

if __name__ == "__main__":
    verify()