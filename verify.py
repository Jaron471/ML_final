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
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm

# 引用你的模組
from model import config
from model.dataset import TuringDataset
from model.model import ParameterNet

# 如果你有 GPU 模擬程式碼，可以嘗試 import 進來做「物理重生成」驗證
# from GPU_generate import simulate_gm_cupy 

def verify():
    # 1. 載入模型與資料
    print(f"Loading model from {config.MODEL_SAVE_PATH}...")
    dataset = TuringDataset(config.NPZ_PATH)
    
    # 為了驗證，我們通常看整個資料集，或者你可以只看沒參與訓練的後 20%
    # 這裡我們為了展示方便，隨機取樣 1000 筆資料來驗證
    indices = np.random.choice(len(dataset), size=min(len(dataset), 1000), replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    model = ParameterNet().to(config.DEVICE)
    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH))
    model.eval()
    
    # 2. 進行預測
    all_preds = []
    all_targets = []
    
    print("Running inference...")
    with torch.no_grad():
        for u_batch, _, params_target in tqdm(loader):
            u_batch = u_batch.to(config.DEVICE)
            
            # 預測 (正規化後的)
            preds_norm = model(u_batch)
            
            # 反正規化回真實數值
            # 注意：dataset.scaler 需要先 to_device 才能處理 Tensor，或者我們取出 scaler 自己算
            preds_norm = preds_norm.cpu().numpy()
            targets_norm = params_target.numpy()
            
            # 手動反正規化 (使用 dataset.scaler 的 numpy 功能)
            preds_real = dataset.scaler.inverse_transform_numpy(preds_norm)
            targets_real = dataset.scaler.inverse_transform_numpy(targets_norm)
            
            all_preds.append(preds_real)
            all_targets.append(targets_real)
            
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # 3. 數值評估 (Metrics)
    param_names = ['a', 'b', 'c', 'delta']
    print("\n" + "="*40)
    print("   Evaluation Metrics (R2 & MAE)")
    print("="*40)
    
    for i, name in enumerate(param_names):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        # 計算相對誤差百分比 (MAPE)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        print(f"Parameter {name}:")
        print(f"  R2 Score : {r2:.4f} (越接近 1 越好)")
        print(f"  MAE      : {mae:.4f}")
        print(f"  MAPE     : {mape:.2f}%")
        print("-" * 20)

    # 4. 繪製散點圖 (Scatter Plots)
    plot_scatter(all_targets, all_preds, param_names)
    
    # 5. 找出預測最好與最差的案例 (Optional)
    # 計算每個樣本的平均相對誤差
    errors = np.mean(np.abs(all_targets - all_preds) / all_targets, axis=1)
    best_idx = np.argmin(errors)
    worst_idx = np.argmax(errors)
    
    print(f"\nBest Prediction Index: {best_idx}, Error: {errors[best_idx]:.4f}")
    print(f"Worst Prediction Index: {worst_idx}, Error: {errors[worst_idx]:.4f}")

def plot_scatter(y_true, y_pred, param_names):
    """
    繪製 2x2 的散點圖
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, ax in enumerate(axes):
        # 繪製散點
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.5, s=10, c='blue', label='Samples')
        
        # 繪製理想對角線 (y=x)
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
    plt.savefig("verification_scatter_plots.png")
    print("\n✅ Scatter plots saved to 'verification_scatter_plots.png'")
    plt.show()

if __name__ == "__main__":
    verify()