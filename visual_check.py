import os
import sys

# --- 1. 解決 OMP 錯誤 (必須放在最上面) ---
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 解決 Windows 中文路徑輸出編碼問題
if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# 引用你的模組
from model import config
from model.dataset import TuringDataset
from model.model import ParameterNet

# 引入 GPU 模擬器 (自動偵測 CuPy 或 MPS)
try:
    from gm_cupy import simulate_gm_cupy as simulate_gm, calculate_gierer_meinhardt_steady_state
    print("✅ Using CuPy (NVIDIA GPU) for re-simulation.")
except ImportError:
    try:
        from gm_mps import simulate_gm_mps as simulate_gm, calculate_gierer_meinhardt_steady_state
        print("✅ Using MPS (Apple Silicon) for re-simulation.")
    except ImportError:
        print("❌ Could not import simulation engine (gm_cupy or gm_mps).")
        sys.exit(1)

def visual_validation():
    # 1. 載入模型
    print("Loading model...")
    model = ParameterNet().to(config.DEVICE)
    model.load_state_dict(torch.load("checkpoint/" + config.MODEL_SAVE_PATH))
    model.eval()
    
    # 2. 載入數據 (隨機取 3 筆)
    dataset = TuringDataset(config.NPZ_PATH)
    indices = np.random.choice(len(dataset), size=3, replace=False)
    
    # 3. 準備畫布
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    plt.subplots_adjust(hspace=0.4)
    
    print("Running re-simulation check...")
    
    for row_idx, data_idx in enumerate(indices):
        # 取得真實數據
        u_tensor, _, params_norm = dataset[data_idx]
        u_true_img = u_tensor[0] # (128, 128)
        
        # 取得真實參數 (反正規化)
        params_true = dataset.scaler.inverse_transform_numpy(params_norm)
        
        # --- 模型預測 ---
        u_input = torch.tensor(u_tensor).unsqueeze(0).to(config.DEVICE) # (1, 1, 128, 128)
        with torch.no_grad():
            pred_norm = model(u_input).cpu().numpy()[0]
        
        # 預測參數 (反正規化)
        params_pred = dataset.scaler.inverse_transform_numpy(pred_norm)
        
        # --- GPU 重生成 (Re-simulation) ---
        # 解包預測參數: a, b, c, delta
        p_a, p_b, p_c, p_d = params_pred
        
        print(f"Sample {row_idx+1}:")
        print(f"  True: a={params_true[0]:.2f}, b={params_true[1]:.2f}, c={params_true[2]:.2f}, d={params_true[3]:.1f}")
        print(f"  Pred: a={p_a:.2f}, b={p_b:.2f}, c={p_c:.2f}, d={p_d:.1f}")
        
        # 計算穩態
        u_star, v_star = calculate_gierer_meinhardt_steady_state(p_a, p_b, p_c)
        
        u_sim = np.zeros_like(u_true_img)
        
        if u_star is not None:
            # 呼叫模擬器 (ID 設為 0, seed 設為固定值以便比較)
            # 注意：這裡假設你的 simulate_gm 接口如下，如果不同請自行調整
            try:
                u_res, _, _ = simulate_gm(
                    id=0, a=p_a, b=p_b, c=p_c, delta=p_d,
                    u_star=u_star, v_star=v_star,
                    seed=42 # 固定種子看結構
                )
                u_sim = u_res
            except Exception as e:
                print(f"  Simulation failed: {e}")
        else:
            print("  Steady state calculation failed for predicted params.")

        # --- 繪圖 ---
        # 1. 真實圖
        ax1 = axes[row_idx, 0]
        ax1.imshow(u_true_img, cmap='magma', origin='lower')
        ax1.set_title(f"Ground Truth\n(d={params_true[3]:.1f})")
        ax1.axis('off')
        
        # 2. 重生成圖
        ax2 = axes[row_idx, 1]
        ax2.imshow(u_sim, cmap='magma', origin='lower')
        ax2.set_title(f"Re-simulated (Prediction)\n(d={p_d:.1f})")
        ax2.axis('off')
        
        # 3. 差異圖 (Difference)
        ax3 = axes[row_idx, 2]
        # 簡單計算絕對差 (注意可能會有些微位移導致誤差大，主要看結構)
        diff = np.abs(u_true_img - u_sim)
        im3 = ax3.imshow(diff, cmap='viridis', origin='lower')
        ax3.set_title(f"Difference\n(Structure Check)")
        ax3.axis('off')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)

    plt.savefig("visual_validation_resimulation.png")
    print("\n✅ Check 'visual_validation_resimulation.png'. If the middle column looks like the left column, your model is working!")
    plt.show()

if __name__ == "__main__":
    visual_validation()