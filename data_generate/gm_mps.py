# -*- coding: utf-8 -*-
"""
批次生成 Gierer-Meinhardt Turing pattern（PyTorch MPS 加速版）
功能：
1. 使用 PyTorch MPS backend 在 Apple Silicon Mac 上進行 GPU 加速。
2. 針對 MPS 優化的 FFT 運算。
3. 支援指定 ID 範圍 (Range) 與動態檔名。
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from multiprocessing import Pool
import matplotlib.pyplot as plt

# 強制引入 PyTorch，檢查 MPS 是否可用
try:
    import torch
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print(f"✅ 成功載入 PyTorch MPS (Device: {device})")
    else:
        print("⚠️ MPS 不可用，使用 CPU")
        device = torch.device("cpu")
except ImportError:
    raise ImportError("❌ 錯誤：此腳本需要安裝 PyTorch (pip install torch)。")

# 嘗試引入 tqdm
try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x

# =============== 全局模擬參數設定 ===============

# 空間 / 時間離散
N_GRID = 128           # MPS 可以輕鬆跑 128x128 或更大
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
MIN_EVOLUTION_STEPS = 2000  # 最小演化步數，確保 Turing 不穩定性有時間發展
MIN_PATTERN_STD     = 0.02  # 最小空間標準差，確保形成了斑紋

# 檔案路徑
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARAM_CSV   = os.path.join(SCRIPT_DIR, "qualified_turing_params_20000.csv")
OUTPUT_FILENAME_BASE = os.path.join(SCRIPT_DIR, "turing_patterns_dataset_mps")

# =============== 範圍設定 (手動調整這裡) ===============
# 設定要執行的 CSV 行數範圍
RANGE_START = 0
RANGE_END   = 20000   # 設為 None 代表跑到最後

# 設定平行工人的數量
# ⚠️ 注意：MPS 在多進程下可能有限制，建議設為 1-2
# 如果遇到問題，請設為 1
MPS_WORKERS = 4


# =============== 數值方法 (PyTorch MPS) ===============

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

def simulate_gm_mps(id, a, b, c, delta, u_star, v_star,
                    s=S_DIFFUSION, N=N_GRID, T_steps=MAX_T_STEPS,
                    dt=DT, dx=DX, noise_std=NOISE_STD, seed=SEED):
    
    grid_shape = (N, N)
    
    # 使用 numpy 的隨機數生成器
    rng = np.random.default_rng(seed + int(id))
    u_noise = rng.normal(0, noise_std, size=grid_shape).astype(np.float32)
    v_noise = rng.normal(0, noise_std, size=grid_shape).astype(np.float32)
    
    # 轉到 MPS
    u = torch.full(grid_shape, u_star, dtype=torch.float32, device=device) + torch.from_numpy(u_noise).to(device)
    v = torch.full(grid_shape, v_star, dtype=torch.float32, device=device) + torch.from_numpy(v_noise).to(device)

    # 預計算算子 (MPS)
    kx = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    ky = 2 * np.pi * np.fft.fftfreq(N, d=dx)
    # 利用廣播機制計算 K^2
    L_eigenvalues = -(kx[:, None]**2 + ky[None, :]**2)
    L_eigenvalues = torch.from_numpy(L_eigenvalues.astype(np.float32)).to(device)
    
    Du = s * 1.0
    Dv = s * delta
    
    # 預計算隱式算子逆矩陣
    D_u_inv = 1.0 / (1.0 - dt * Du * L_eigenvalues)
    D_v_inv = 1.0 / (1.0 - dt * Dv * L_eigenvalues)

    # --- 時間步進迴圈 (MPS GPU) ---
    current_t = 0
    max_steps_dynamic = T_steps
    has_extended = False

    while current_t < max_steps_dynamic:
        # Explicit Reaction
        v_safe = torch.where(v > 1e-6, v, torch.tensor(1e-6, device=device))
        denominator = v_safe * (1 + c * u**2)
        
        # 反應項
        Ru = a - (b * u) + (u**2 / denominator)
        Rv = (u**2) - v

        # FFT (PyTorch 的 FFT 在 MPS 上運行)
        R_u_hat = torch.fft.fft2(Ru)
        R_v_hat = torch.fft.fft2(Rv)
        U_hat_k = torch.fft.fft2(u)
        V_hat_k = torch.fft.fft2(v)

        # IMEX Update
        U_hat_k_plus_1 = (U_hat_k + dt * R_u_hat) * D_u_inv
        V_hat_k_plus_1 = (V_hat_k + dt * R_v_hat) * D_v_inv

        # Inverse FFT
        u_next = torch.real(torch.fft.ifft2(U_hat_k_plus_1))
        v_next = torch.real(torch.fft.ifft2(V_hat_k_plus_1))

        current_t += 1

        # 檢查收斂
        if current_t % CHECK_INTERVAL == 0:
            abs_diff = float(torch.max(torch.abs(u_next - u)).item())
            
            # 只有在最小演化步數之後，且變化夠小，才視為收斂
            if current_t >= MIN_EVOLUTION_STEPS and abs_diff < PRACTICAL_TOLERANCE:
                return u_next.cpu().numpy(), v_next.cpu().numpy(), current_t
        
            # [新增] 如果到達預定的 max_steps，但還沒形成斑紋，且還沒延長過 -> 延長一倍
            if current_t >= max_steps_dynamic and not has_extended:
                # 檢查斑紋強度 (從 CPU 檢查)
                # 注意：這裡會造成一些 synchronization overhead，但只在 loop 尾端發生一次
                u_cpu = u_next.cpu().numpy()
                u_std = np.std(u_cpu)
                u_range = np.max(u_cpu) - np.min(u_cpu)
                
                # 閾值判斷 (使用與 run_one_row 一致的標準)
                MIN_RANGE_CHK = 0.01
                if u_std < MIN_PATTERN_STD or u_range < MIN_RANGE_CHK:
                    # 決定延長
                    # print(f"  --> ID {id} extending steps from {max_steps_dynamic} to {max_steps_dynamic*2} (std={u_std:.4f})")
                    max_steps_dynamic = T_steps * 2
                    has_extended = True
                    # 繼續 while 迴圈 (因為 max_steps_dynamic 變大了，條件 current_t < max_steps_dynamic 成立)

        u = u_next
        v = v_next

    # 達到最大步數，回傳
    return u.cpu().numpy(), v.cpu().numpy(), current_t


# =============== Worker Function ===============

def run_one_row(row_dict):
    """
    Worker 負責：
    1. 接收參數
    2. 呼叫 MPS 模擬
    3. 將 MPS 結果轉回 CPU 並回傳
    4. 驗證是否真正形成圖靈斑紋
    """
    try:
        pid = int(row_dict["id"])
        a_val, b_val, c_val, d_val = float(row_dict["a"]), float(row_dict["b"]), float(row_dict["c"]), float(row_dict["delta"])

        # 穩態計算 (CPU)
        u_star, v_star = calculate_gierer_meinhardt_steady_state(a_val, b_val, c_val)
        if u_star is None:
            return None
            
        # 呼叫 MPS 模擬
        u_res, v_res, steps = simulate_gm_mps(
            pid, a_val, b_val, c_val, d_val, u_star, v_star,
            seed=SEED+pid
        )

        # 驗證最終結果
        u_std = np.std(u_res)
        u_range = np.max(u_res) - np.min(u_res)
        
        # 閾值：標準差和範圍都要足夠大，才算形成了斑紋
        MIN_RANGE = 0.01
        
        if u_std < MIN_PATTERN_STD or u_range < MIN_RANGE:
            # print(f"⚠️  ID {pid}: 未形成斑紋 (std={u_std:.6f}, range={u_range:.6f}) - 已拒絕")
            return None
        
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

def save_results_to_file(results, filename_base, batch_idx, s_idx, e_idx):
    if not results:
        return
    
    count = len(results)
    # 決定輸出檔名
    output_filename = f"{filename_base}_batch{batch_idx}_{s_idx}_{e_idx}.npz"
    
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
    print(f"💾 已儲存批次 {batch_idx}: {output_filename} (含 {count} 筆)")


def main():
    start_time = time.time()
    
    if not os.path.exists(PARAM_CSV):
        print(f"❌ 找不到 {PARAM_CSV}")
        return

    df = pd.read_csv(PARAM_CSV)
    
    # --- 範圍控制 ---
    s_idx = max(0, RANGE_START)
    e_idx = min(len(df), RANGE_END) if RANGE_END is not None else len(df)
    df_to_run = df.iloc[s_idx:e_idx]
    
    print(f"🎯 執行範圍: {s_idx} ~ {e_idx} (共 {len(df_to_run)} 筆)")
    print(f"🚀 [MPS 模式] 啟動 {MPS_WORKERS} 個 Worker (PyTorch MPS)...")
    print(f"💾 輸出將以 1000 筆為單位儲存至: {OUTPUT_FILENAME_BASE}...")

    rows = [row.to_dict() for _, row in df_to_run.iterrows()]

    results_buffer = []
    batch_counter = 0
    SAVE_INTERVAL = 1000
    
    def flush_results():
        nonlocal batch_counter
        if results_buffer:
            save_results_to_file(results_buffer, OUTPUT_FILENAME_BASE, batch_counter, s_idx, e_idx)
            batch_counter += 1
            results_buffer.clear()

    if MPS_WORKERS == 1:
        # 單進程模式（推薦給 MPS）
        for row in tqdm(rows, unit="sim"):
            res = run_one_row(row)
            if res is not None:
                results_buffer.append(res)
                if len(results_buffer) >= SAVE_INTERVAL:
                    flush_results()
    else:
        # 多進程模式
        with Pool(processes=MPS_WORKERS) as pool:
            iterator = pool.imap(run_one_row, rows, chunksize=1)
            for res in tqdm(iterator, total=len(rows), unit="sim"):
                if res is not None:
                    results_buffer.append(res)
                    if len(results_buffer) >= SAVE_INTERVAL:
                        flush_results()
    
    # 儲存最後剩餘的
    flush_results()
    
    print(f"\n✅ 全部完成。總耗時: {time.time() - start_time:.2f}s")

if __name__ == "__main__":
    main()
