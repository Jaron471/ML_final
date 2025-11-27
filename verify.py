import os
import sys
import argparse

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

def main():
    parser = argparse.ArgumentParser(description="驗證模型性能")
    parser.add_argument('--method', choices=['pinn', 'hybrid'], default='hybrid',
                       help='選擇要驗證的方法 (預設: hybrid)')
    args = parser.parse_args()

    # 根據method動態選擇config
    if args.method == 'pinn':
        from model import config_pinn as config
        print("🔍 驗證原始PINN方法")
    else:  # hybrid
        from model import config_hybrid as config
        print("🔍 驗證混合循環方法")

    verify(config)

# 引用你的模組
from model.dataset import TuringDataset
# --- 修改這裡：同時引入 MLPNet ---
from model.model import ParameterNet, MLPNet

def verify(config):
    # 1. 載入資料並使用與訓練時相同的分割
    print(f"Loading model from {config.MODEL_SAVE_PATH}...")
    dataset = TuringDataset(config.NPZ_PATH)
    
    # 使用與訓練時完全相同的分割邏輯，確保驗證集與訓練時一致
    import torch
    train_size = int(config.TRAIN_VAL_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = torch.utils.data.random_split(
        dataset, 
        [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    # 使用訓練時的驗證集進行評估
    loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    print(f"Using validation set from training split: {len(val_dataset)} samples")
    
    # --- 修改這裡：根據 Config 選擇模型架構 ---
    print(f"Using Model Architecture: {config.MODEL_TYPE}")
    if config.MODEL_TYPE == "CNN":
        model = ParameterNet().to(config.DEVICE)
    elif config.MODEL_TYPE == "MLP":
        model = MLPNet().to(config.DEVICE)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {config.MODEL_TYPE}")
    # ----------------------------------------

    # 載入訓練好的權重
    # 注意：如果權重檔名有變 (例如 MLP_PINN.pth)，請確保 config.MODEL_SAVE_PATH 指向正確檔案
    # 或者手動指定路徑: model.load_state_dict(torch.load("MLP_PINN.pth"))
    try:
        model.load_state_dict(torch.load("checkpoint/" + config.MODEL_SAVE_PATH))
    except FileNotFoundError:
        print(f"❌ 找不到權重檔: {config.MODEL_SAVE_PATH}")
        print("請確認 config.MODEL_TYPE 是否與訓練時一致，或手動修改路徑。")
        return

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
    print("\n" + "="*75)
    print(f"   Evaluation Metrics ({config.MODEL_TYPE})")
    print("="*75)
    
    # =============== 多維 NRMSE (論文定義) ===============
    # RMSE = sqrt( 1/(m*d) * sum_i sum_j (y_{i,j} - f_j(x_i))^2 )
    # NRMSE = RMSE / ||y_bar||
    num_samples = all_targets.shape[0]        # m (樣本數量)
    num_parameters = all_targets.shape[1]     # d (參數數量，此處為 4)
    num_items = num_samples * num_parameters  # m * d (所有單一預測總數)
    
    total_squared_error = np.sum((all_targets - all_preds)**2)
    rmse_multi_dim = np.sqrt(total_squared_error / num_items)
    
    # 歸一化因子：真實參數平均向量的歐幾里得範數
    y_mean_vector = np.mean(all_targets, axis=0)
    norm_y_mean = np.linalg.norm(y_mean_vector)
    
    if norm_y_mean != 0:
        nrmse_multi_dim = rmse_multi_dim / norm_y_mean
    else:
        nrmse_multi_dim = 0.0
    
    print(f"\n🎯 多維 NRMSE (論文定義):")
    print(f"   RMSE (multi-dim) = {rmse_multi_dim:.6f}")
    print(f"   Normalization Factor (||y_bar||) = {norm_y_mean:.6f}")
    print(f"   ➤ NRMSE = {nrmse_multi_dim:.4f} ({nrmse_multi_dim:.2%})")
    
    # =============== 單參數評估 (按範圍標準化) ===============
    print(f"\n{'Metric':<15} | {'a':<10} | {'b':<10} | {'c':<10} | {'delta':<10}")
    print("-" * 75)
    
    metrics = {'R2': [], 'MAE': [], 'NRMSE (by Range)': []}
    
    for i, name in enumerate(param_names):
        y_true = all_targets[:, i]
        y_pred = all_preds[:, i]
        
        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        
        # 計算 NRMSE (按範圍標準化)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        data_range = np.max(y_true) - np.min(y_true)
        nrmse = (rmse / data_range) if data_range != 0 else 0.0
        
        metrics['R2'].append(r2)
        metrics['MAE'].append(mae)
        metrics['NRMSE (by Range)'].append(nrmse)

    # 顯示表格
    for name, vals in metrics.items():
        if name == 'NRMSE (by Range)':
            row_str = f"{name:<15} | {vals[0]:.2%}   | {vals[1]:.2%}   | {vals[2]:.2%}   | {vals[3]:.2%}"
        else:
            row_str = f"{name:<15} | {vals[0]:.4f}   | {vals[1]:.4f}   | {vals[2]:.4f}   | {vals[3]:.4f}"
        print(row_str)
        
    print("="*75)

    # 4. 繪製散點圖 (Scatter Plots)
    plot_scatter(all_targets, all_preds, param_names, config)

def plot_scatter(y_true, y_pred, param_names, config):
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
    filename = f"verify_{config.MODEL_TYPE}.png"
    plt.savefig(filename)
    print(f"\n✅ Scatter plots saved to '{filename}'")
    plt.show()

if __name__ == "__main__":
    main()