"""
注意事項：
記憶體使用 (VRAM)：
    對於 128 X 128 的網格，總節點數 M=16384
    如果遇到 OOM，可以考慮依照論文建議將數據下採樣到 64 X 64（將 t 設為 2），這會將記憶體需求降低 16 倍
R_max 的設定：
    程式碼中暫時使用單張圖的 99% 分位數作為 R_max
    論文要求：您應該先對訓練集中的一部分圖案跑一遍，統計出一個全局的 $R_{max}$，然後在生成最終特徵時固定使用這個值，這樣不同圖案的直方圖才具有可比性
"""

import numpy as np
import cupy as cp
import time

# ==========================================
# Part 1: 核心運算函式 (Core Functions)
# ==========================================

def calculate_graph_weights_cupy(u, epsilon=0.003):
    """
    計算加權圖的鄰接矩陣 (Adjacency Matrix)。
    """
    N = u.shape[0]
    M = N * N  # 總節點數
    
    mean_u = cp.mean(u)
    u_flat = u.flatten()
    is_high = (u_flat >= mean_u)
    
    # 初始化稠密鄰接矩陣
    Omega = cp.zeros((M, M), dtype=cp.float32)
    
    grid_indices = cp.arange(M).reshape(N, N)
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    for dr, dc in shifts:
        neighbor_indices = cp.roll(grid_indices, shift=(-dr, -dc), axis=(0, 1)).flatten()
        
        curr_high = is_high
        neigh_high = is_high[neighbor_indices]
        
        same_side = (curr_high == neigh_high)
        weights = cp.full(M, epsilon, dtype=cp.float32)
        weights[same_side] = 1.0
        
        current_indices_flat = cp.arange(M)
        Omega[current_indices_flat, neighbor_indices] = weights
        
    return Omega

def calculate_resistance_matrix_cupy(Omega):
    """
    計算電阻距離矩陣 R。
    """
    M = Omega.shape[0]
    degrees = cp.sum(Omega, axis=1)
    
    L_G = -Omega
    L_G[cp.diag_indices(M)] += degrees
    
    J = cp.ones((M, M), dtype=cp.float32)
    Mat_to_inv = J + L_G
    K = cp.linalg.inv(Mat_to_inv)
    
    diag_K = cp.diag(K)
    R = diag_K[:, None] + diag_K[None, :] - 2 * K
    
    return R

def get_r_values_for_radius(R, N, radius):
    """
    輔助函式：根據半徑 r 篩選電阻值
    """
    M = N * N
    # 構造網格坐標並計算距離 (Torus distance)
    rows, cols = cp.indices((N, N))
    coords = cp.stack([rows, cols], axis=-1).reshape(M, 2)
    
    d0 = cp.abs(coords[:, 0:1] - coords[:, 0:1].T)
    d0 = cp.minimum(d0, N - d0)
    d1 = cp.abs(coords[:, 1:2] - coords[:, 1:2].T)
    d1 = cp.minimum(d1, N - d1)
    
    grid_dists = cp.sqrt(d0**2 + d1**2)
    mask = (grid_dists <= radius)
    
    return R[mask]

# ==========================================
# Part 2: 兩階段處理邏輯 (Two-Stage Process)
# ==========================================

def get_global_r_max(dataset_u, radius=8):
    """
    第一階段：掃描數據集以確定全局 R_max
    嚴格遵循論文：計算每個圖案的 99% 分位數，然後取其中的最大值。
    """
    percentiles = []
    total_imgs = len(dataset_u)
    print(f"\n[Stage 1] 正在掃描 {total_imgs} 張圖案以計算全局 R_max...")
    
    start_time = time.time()
    for i, u_cpu in enumerate(dataset_u):
        # 1. 移至 GPU
        u_gpu = cp.asarray(u_cpu, dtype=cp.float32)
        N = u_gpu.shape[0]
        
        # 2. 計算權重與電阻
        Omega = calculate_graph_weights_cupy(u_gpu)
        R = calculate_resistance_matrix_cupy(Omega)
        
        # 3. 篩選半徑內的電阻值
        r_values = get_r_values_for_radius(R, N, radius)
        
        # 4. 計算單張圖的 99% 分位數
        p99 = cp.percentile(r_values, 99)
        percentiles.append(float(p99))
        
        # 進度條
        if (i + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  已掃描 {i + 1}/{total_imgs} 張... ({elapsed:.2f}s)")
            
    # 5. 取所有圖案 99% 分位數中的最大值
    global_r_max = max(percentiles)
    print(f"[Stage 1] 掃描完成。全局 R_max (基於 99% 分位數的最大值): {global_r_max:.4f}")
    return global_r_max

def generate_final_features(dataset_u, global_r_max, radius=8, bins=12):
    """
    第二階段：使用固定的 global_r_max 生成特徵
    """
    features = []
    total_imgs = len(dataset_u)
    print(f"\n[Stage 2] 正在生成最終 RDH 特徵 (Bins={bins}, R_max={global_r_max:.4f})...")
    
    start_time = time.time()
    for i, u_cpu in enumerate(dataset_u):
        u_gpu = cp.asarray(u_cpu, dtype=cp.float32)
        N = u_gpu.shape[0]
        
        # 重複計算 R (如果記憶體足夠，其實可以在 Stage 1 把 R 存下來，但通常 VRAM 不夠存所有 R)
        Omega = calculate_graph_weights_cupy(u_gpu)
        R = calculate_resistance_matrix_cupy(Omega)
        r_values = get_r_values_for_radius(R, N, radius)
        
        # 生成直方圖，固定 range=(0, global_r_max)
        hist, _ = cp.histogram(r_values, bins=bins, range=(0, global_r_max), density=True)
        
        # 歸一化 (Sum = 1)
        hist = hist / cp.sum(hist)
        features.append(cp.asnumpy(hist))

        if (i + 1) % 10 == 0:
             elapsed = time.time() - start_time
             print(f"  已生成 {i + 1}/{total_imgs} 張特徵... ({elapsed:.2f}s)")
        
    return np.array(features)

# ==========================================
# Part 3: 主程式入口 (Main Execution)
# ==========================================

if __name__ == "__main__":
    # 1. 載入數據
    DATA_PATH = 'turing_patterns_dataset_cuda.npz'
    print(f"正在載入數據: {DATA_PATH}")
    data = np.load(DATA_PATH)
    
    # 讀取濃度場 u
    dataset_u = data['u'] 
    
    # 可以只提取前x張做測試
    # dataset_u = dataset_u[:10] 
    print(f"數據集大小: {dataset_u.shape}")

    # 參數設定 (依據論文)
    RADIUS = 8
    BINS = 12
    
    # 2. 執行第一階段：取得全局 R_max
    global_r_max = get_global_r_max(dataset_u, radius=RADIUS)
    
    # 3. 執行第二階段：生成特徵矩陣
    X_features = generate_final_features(dataset_u, global_r_max, radius=RADIUS, bins=BINS)
    
    print("\n================ 結果 ================")
    print(f"特徵矩陣形狀: {X_features.shape}")
    print(f"第一張圖的特徵向量:\n{X_features[0]}")
    
    # 4. (可選) 儲存特徵供後續機器學習使用
    # np.savez('rdh_features.npz', features=X_features, r_max=global_r_max)