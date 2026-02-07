# -*- coding: utf-8 -*-
"""
批次生成 Gierer-Meinhardt Turing pattern（純 CuPy GPU 極速版）
功能：
1. 強制使用 CuPy 進行 GPU 加速。
2. 針對 GPU 優化的 FFT 運算。
3. 支援指定 ID 範圍 (Range) 與動態檔名。
"""

import os
import time
import sys
import glob
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from multiprocessing import Pool, set_start_method
import matplotlib.pyplot as plt

# ==========================================
# 自動修正：加入 NVIDIA 套件 DLL 路徑 (Windows)
# 解決 cupy-cuda12x 找不到 nvrtc64_120_0.dll 的問題
# ==========================================
if os.name == 'nt':
    try:
        # 搜尋 site-packages 中的 nvidia 資料夾
        site_packages = [p for p in sys.path if 'site-packages' in p]
        for sp in site_packages:
            nvidia_path = os.path.join(sp, 'nvidia')
            if os.path.isdir(nvidia_path):
                # 遞歸尋找 bin 目錄
                for root, dirs, files in os.walk(nvidia_path):
                    if 'bin' in dirs:
                        bin_path = os.path.join(root, 'bin')
                        # 加入 DLL 搜尋路徑 (Python 3.8+)
                        if hasattr(os, 'add_dll_directory'):
                            try:
                                os.add_dll_directory(bin_path)
                            except Exception:
                                pass
                        # 同步加入環境變數 PATH (供子進程或其他工具使用)
                        if bin_path not in os.environ['PATH']:
                            os.environ['PATH'] = bin_path + os.pathsep + os.environ['PATH']
    except Exception as e:
        print(f"⚠️ Warning: Auto-setup of NVIDIA paths failed: {e}")

# 強制引入 CuPy，如果沒裝會直接報錯
try:
   import cupy as cp
   # 測試一下 NVRTC 是否可用
   cp.cuda.Device(0).use()
   try:
       # 觸發一次編譯確保 DLL 載入成功
       cp.ElementwiseKernel('T x', 'T y', 'y = x', 'test_kernel')(cp.array([1.]))
   except Exception as e:
       print(f"⚠️ CuPy 載入成功但 NVRTC 可能失敗: {e}")
       raise e

   print(f"✅ 成功載入 CuPy (Device: {cp.cuda.Device(0).compute_capability})")
except ImportError as e:
    print(f"❌ CuPy Import Error: {e}")
    raise ImportError("❌ 錯誤：此腳本需要安裝 CuPy (pip install cupy-cuda12x) 且需正確配置 CUDA 環境。")

# 嘗試引入 tqdm
try:
   from tqdm import tqdm
except ImportError:
   tqdm = lambda x, **kwargs: x

# =============== 全局模擬參數設定 ===============

# 模擬參數
N_GRID = 128           # GPU 可以輕鬆跑 128x128 或更大
DT = 0.2
DX = 1.0

# 擴散係數
S_DIFFUSION = 0.4

# 模擬時間
MAX_T       = 5000.0
MAX_T_STEPS = int(MAX_T / DT)

# 雜訊與種子
NOISE_STD = 0.001
SEED      = 41

# 收斂參數
PRACTICAL_TOLERANCE = 1e-6
MAX_REL_CHANGE      = 1e-6
CHECK_INTERVAL      = 200

# 檔案路徑
PARAM_CSV   = "qualified_turing_params_20500.csv"
OUTPUT_FILENAME_BASE = "turing_patterns_dataset_cupy"

# =============== 範圍設定 (手動調整這裡) ===============
# 設定要執行的 CSV 行數範圍
RANGE_START = 20000
RANGE_END   = 20500   # 設為 None 代表跑到最後

# 設定 GPU 平行工人的數量 (建議 1~4)
# ⚠️ 注意：每個 Worker 都會佔用約 500MB 顯存。如果你的顯存小於 8GB，建議設為 1 或 2。
GPU_WORKERS = 1


# =============== 數值方法 (純 CuPy) ===============

def calculate_gierer_meinhardt_steady_state(a, b, c):
   """ 穩態計算 (這部分用 CPU/Scipy 算比較快且穩) """
   def steady_state_eq(u):
       if u <= 0: return 1e6
       return b * u - (a + 1.0 / (1.0 + c * u**2))

   u_initial_guess = (a + 1.0) / b
   try:
       u_star = fsolve(steady_state_eq, u_initial_guess, maxfev=200)[0]
   except:
       return None, None

   if u_star <= 0: return None, None
   if not np.isclose(steady_state_eq(u_star), 0, atol=1e-5): return None, None
   return u_star, u_star**2

def simulate_gm_cupy(id, a, b, c, delta, u_star, v_star,
                    s=S_DIFFUSION, N=N_GRID, T_steps=MAX_T_STEPS,
                    dt=DT, dx=DX, noise_std=NOISE_STD, seed=SEED):
  
   # 確保使用 GPU 0 (多卡環境可在此修改)
   cp.cuda.Device(0).use()
  
   grid_shape = (N, N)
  
   # 使用 cupy 的隨機數生成器
   # 注意：在多進程中重設 seed 很重要
   rng = np.random.default_rng(seed + int(id))
   u_noise = rng.normal(0, noise_std, size=grid_shape).astype(np.float32)
   v_noise = rng.normal(0, noise_std, size=grid_shape).astype(np.float32)
  
   # 丟到 GPU
   u = cp.full(grid_shape, u_star, dtype=cp.float32) + cp.asarray(u_noise)
   v = cp.full(grid_shape, v_star, dtype=cp.float32) + cp.asarray(v_noise)

   # 預計算算子 (GPU)
   kx = 2 * cp.pi * cp.fft.fftfreq(N, d=dx)
   ky = 2 * cp.pi * cp.fft.fftfreq(N, d=dx)
   # 利用廣播機制計算 K^2
   L_eigenvalues = -(kx[:, None]**2 + ky[None, :]**2)
  
   Du = s * 1.0
   Dv = s * delta
  
   # 預計算隱式算子逆矩陣
   D_u_inv = 1.0 / (1.0 - dt * Du * L_eigenvalues)
   D_v_inv = 1.0 / (1.0 - dt * Dv * L_eigenvalues)

   # --- 時間步進迴圈 (純 GPU) ---
   for t in range(T_steps):
       # Explicit Reaction
       # cp.where, cp.multiply 等操作都是在 GPU 上並行
       v_safe = cp.where(v > 1e-6, v, 1e-6)
       denominator = v_safe * (1 + c * u**2)
      
       # 反應項
       Ru = a - (b * u) + (u**2 / denominator)
       Rv = (u**2) - v

       # FFT (CuPy 的 FFT 非常快，因為使用了 cuFFT)
       R_u_hat = cp.fft.fft2(Ru)
       R_v_hat = cp.fft.fft2(Rv)
       U_hat_k = cp.fft.fft2(u)
       V_hat_k = cp.fft.fft2(v)

       # IMEX Update
       U_hat_k_plus_1 = (U_hat_k + dt * R_u_hat) * D_u_inv
       V_hat_k_plus_1 = (V_hat_k + dt * R_v_hat) * D_v_inv

       # Inverse FFT
       u_next = cp.real(cp.fft.ifft2(U_hat_k_plus_1))
       v_next = cp.real(cp.fft.ifft2(V_hat_k_plus_1))

       # Check Convergence (每隔一段時間檢查)
       if (t + 1) % CHECK_INTERVAL == 0:
           # 這裡需要一個同步點，計算 max diff
           abs_diff = float(cp.max(cp.abs(u_next - u))) # float() 會觸發從 GPU 拉回 CPU 的同步
          
           if abs_diff < PRACTICAL_TOLERANCE:
               # 收斂，回傳結果 (轉回 CPU NumPy array)
               return cp.asnumpy(u_next), cp.asnumpy(v_next), t + 1
      
       u = u_next
       v = v_next

   # 達到最大步數，回傳
   return cp.asnumpy(u), cp.asnumpy(v), T_steps


# =============== Worker Function ===============

def run_one_row(row_dict):
   """
   Worker 負責：
   1. 接收參數
   2. 呼叫 GPU 模擬
   3. 將 GPU 結果轉回 CPU 並回傳
   """
   try:
       pid = int(row_dict["id"])
       a_val, b_val, c_val, d_val = float(row_dict["a"]), float(row_dict["b"]), float(row_dict["c"]), float(row_dict["delta"])

       # 穩態計算 (CPU)
       u_star, v_star = calculate_gierer_meinhardt_steady_state(a_val, b_val, c_val)
       if u_star is None: return None

       # GPU 模擬
       u_res, v_res, steps = simulate_gm_cupy(
           id=pid, a=a_val, b=b_val, c=c_val, delta=d_val,
           u_star=u_star, v_star=v_star
       )

       # 這裡的 u_res, v_res 已經是 numpy array 了
       return {
           "id": pid, "a": a_val, "b": b_val, "c": c_val, "delta": d_val,
           "u_star": u_star, "v_star": v_star,
           "u": u_res.astype(np.float32), "v": v_res.astype(np.float32),
           "final_step": steps
       }
   except Exception as e:
       print(f"Error ID {row_dict.get('id')}: {e}")
       return None


# =============== Main ===============

def main():
   # Windows 下使用 multiprocessing 搭配 CUDA 建議使用 spawn (但 Pool 預設就是 spawn)
   # set_start_method('spawn', force=True)
  
   start_time = time.time()
  
   if not os.path.exists(PARAM_CSV):
       print(f"❌ 找不到 {PARAM_CSV}")
       return

   df = pd.read_csv(PARAM_CSV)
  
   # --- 範圍控制 ---
   s_idx = max(0, RANGE_START)
   e_idx = min(len(df), RANGE_END) if RANGE_END is not None else len(df)
   df_to_run = df.iloc[s_idx:e_idx]
  
   output_filename = f"{OUTPUT_FILENAME_BASE}_{s_idx}_{e_idx}.npz"
   print(f"🎯 執行範圍: {s_idx} ~ {e_idx} (共 {len(df_to_run)} 筆)")
   print(f"🚀 [GPU 模式] 啟動 {GPU_WORKERS} 個 Worker (純 CuPy)...")
   print(f"💾 輸出檔名: {output_filename}")

   rows = [row.to_dict() for _, row in df_to_run.iterrows()]

   results = []
  
   # 使用 Pool 進行並行處理
   # 雖然是 GPU，但開 2 個 Process 可以掩蓋數據傳輸的時間
   with Pool(processes=GPU_WORKERS) as pool:
       iterator = pool.imap(run_one_row, rows, chunksize=1)
       for res in tqdm(iterator, total=len(rows), unit="sim"):
           if res is not None:
               results.append(res)

   print(f"\n✅ 成功樣本: {len(results)}")
   if not results: return

   # 打包儲存
   count = len(results)
   U_all = np.zeros((count, N_GRID, N_GRID), dtype=np.float32)
   V_all = np.zeros((count, N_GRID, N_GRID), dtype=np.float32)
   ids = np.zeros(count, dtype=np.int32)
   a_arr = np.zeros(count, dtype=np.float32)
   b_arr = np.zeros(count, dtype=np.float32)
   c_arr = np.zeros(count, dtype=np.float32)
   delta_arr = np.zeros(count, dtype=np.float32)
   u_star_arr = np.zeros(count, dtype=np.float32)
   v_star_arr = np.zeros(count, dtype=np.float32)
   step_arr = np.zeros(count, dtype=np.int32)
  
   for i, r in enumerate(results):
       U_all[i] = r["u"]
       V_all[i] = r["v"]
       ids[i] = r["id"]
       a_arr[i] = r["a"]
       b_arr[i] = r["b"]
       c_arr[i] = r["c"]
       delta_arr[i] = r["delta"]
       u_star_arr[i] = r.get("u_star", 0)
       v_star_arr[i] = r.get("v_star", 0)
       step_arr[i] = r["final_step"]

   np.savez(
       output_filename,
       u=U_all,
       v=V_all,
       ids=ids,
       a=a_arr,
       b=b_arr,
       c=c_arr,
       delta=delta_arr,
       u_star=u_star_arr,
       v_star=v_star_arr,
       final_step=step_arr,
       N_grid=N_GRID,
       dt=DT,
       dx=DX,
       s_diffusion=S_DIFFUSION,
       noise_std=NOISE_STD,
   )
   print(f"🎉 儲存完畢: {output_filename} ({time.time()-start_time:.2f}s)")

   # 預覽前 4 張
   if count > 0:
       import matplotlib.pyplot as plt
       fig, axes = plt.subplots(1, 4, figsize=(16, 4))
       indices = np.linspace(0, count-1, 4, dtype=int)
       for i, idx in enumerate(indices):
           if i >= 4: break
           ax = axes[i] if count > 1 else axes
           ax.imshow(U_all[idx], cmap='magma', origin='lower')
           ax.set_title(f"ID {ids[idx]}")
           ax.axis('off')
       plt.show()

if __name__ == "__main__":
   main()