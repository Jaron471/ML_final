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
from scipy.signal import find_peaks

# ==========================================
# 1. 核心運算 (Core Functions)
# ==========================================

def calculate_graph_weights_cupy(u, epsilon=0.003):
    """計算加權圖的鄰接矩陣 (論文 3.1.1)。"""
    N = u.shape[0]
    M = N * N
    
    mean_u = cp.mean(u)
    u_flat = u.flatten()
    is_high = (u_flat >= mean_u)
    
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
    """計算電阻距離矩陣 R (論文 3.1.2)。"""
    M = Omega.shape[0]
    degrees = cp.sum(Omega, axis=1)
    
    # L = D - Omega
    L_G = -Omega
    L_G[cp.diag_indices(M)] += degrees
    
    # K = (J + L)^-1
    J = cp.ones((M, M), dtype=cp.float32)
    Mat_to_inv = J + L_G 
    K = cp.linalg.inv(Mat_to_inv)
    
    # R_ij = K_ii + K_jj - 2K_ij
    diag_K = cp.diag(K)
    R = diag_K[:, None] + diag_K[None, :] - 2 * K
    
    return R

def precalculate_distance_mask(N, radius):
    """
    【優化關鍵】
    預先計算距離遮罩。因為網格幾何結構不變，這只需要算一次！
    """
    print(f"正在預計算 {N}x{N} 網格的距離遮罩 (Radius={radius})...")
    M = N * N
    rows, cols = cp.indices((N, N))
    coords = cp.stack([rows, cols], axis=-1).reshape(M, 2)
    
    # 為了節省記憶體，我們分批計算或者使用 float16，
    # 但為了簡單，這裡假設 GPU 夠大 (16384^2 bool mask 約 256MB)
    
    d0 = cp.abs(coords[:, 0:1] - coords[:, 0:1].T)
    d0 = cp.minimum(d0, N - d0) # Torus distance
    d1 = cp.abs(coords[:, 1:2] - coords[:, 1:2].T)
    d1 = cp.minimum(d1, N - d1)
    
    grid_dists = cp.sqrt(d0**2 + d1**2)
    mask = (grid_dists <= radius)
    
    print("遮罩計算完成。")
    return mask

def calculate_max_concentration(u, bins=25):
    """計算最大濃度 c_m (論文 3.2)。"""
    u_cpu = cp.asnumpy(u).flatten()
    counts, bin_edges = np.histogram(u_cpu, bins=bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peaks, _ = find_peaks(counts)
    
    if len(peaks) > 0:
        c_m = bin_centers[peaks[-1]]
    else:
        c_m = bin_centers[np.argmax(counts)]
    return float(c_m)

# ==========================================
# 2. 兩階段流程 (Two-Stage Process)
# ==========================================

def get_global_r_max(dataset_u, mask, sample_size=200):
    """
    第一階段：【優化】只採樣部分數據來估計 R_max
    """
    total_imgs = len(dataset_u)
    # 隨機採樣索引
    if total_imgs > sample_size:
        indices = np.random.choice(total_imgs, sample_size, replace=False)
        print(f"\n[Stage 1] 隨機採樣 {sample_size} 張圖案來估計全局 R_max...")
    else:
        indices = np.arange(total_imgs)
        print(f"\n[Stage 1] 掃描所有 {total_imgs} 張圖案...")

    percentiles = []
    start_time = time.time()
    
    for idx, i in enumerate(indices):
        u_cpu = dataset_u[i]
        u_gpu = cp.asarray(u_cpu, dtype=cp.float32)
        
        # 核心計算
        Omega = calculate_graph_weights_cupy(u_gpu)
        R = calculate_resistance_matrix_cupy(Omega)
        
        # 使用預計算的遮罩直接取值
        r_values = R[mask]
        
        p99 = cp.percentile(r_values, 99)
        percentiles.append(float(p99))
        
        # 釋放記憶體
        del Omega, R, r_values, u_gpu
        cp.get_default_memory_pool().free_all_blocks()
        
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - start_time
            print(f"  Sample {idx + 1}/{len(indices)} | Last p99: {percentiles[-1]:.2f} | Time: {elapsed:.2f}s")
            
    global_r_max = max(percentiles)
    print(f"[Stage 1] 完成。全局 R_max: {global_r_max:.4f}")
    return global_r_max

def generate_final_features(dataset_u, mask, global_r_max, bins=12):
    """
    第二階段：生成所有特徵
    """
    features = []
    total_imgs = len(dataset_u)
    print(f"\n[Stage 2] 生成最終特徵 (Total: {total_imgs})...")
    
    start_time = time.time()
    
    for i, u_cpu in enumerate(dataset_u):
        u_gpu = cp.asarray(u_cpu, dtype=cp.float32)
        
        # 1. 計算矩陣
        Omega = calculate_graph_weights_cupy(u_gpu)
        R = calculate_resistance_matrix_cupy(Omega)
        
        # 2. RDH
        r_values = R[mask]
        hist, _ = cp.histogram(r_values, bins=bins, range=(0, global_r_max), density=True)
        hist = hist / cp.sum(hist)
        
        # 3. c_m
        c_m = calculate_max_concentration(u_gpu, bins=25)
        
        # 4. 合併
        feat_vec = np.concatenate([cp.asnumpy(hist), [c_m]])
        features.append(feat_vec)
        
        # 5. 清理
        del Omega, R, r_values, u_gpu
        cp.get_default_memory_pool().free_all_blocks()

        if (i + 1) % 50 == 0:
             elapsed = time.time() - start_time
             # 估算剩餘時間
             avg_time = elapsed / (i + 1)
             remain_time = avg_time * (total_imgs - i - 1)
             print(f"  Progress: {i + 1}/{total_imgs} | Elapsed: {elapsed/60:.1f}m | ETA: {remain_time/60:.1f}m")
        
    return np.array(features)

# ==========================================
# 3. 主程式
# ==========================================
if __name__ == "__main__":
    # 設定
    DATA_PATH = 'turing_patterns_dataset_merged.npz'
    RADIUS = 8
    BINS = 12
    SAMPLE_SIZE_FOR_RMAX = 200 
    
    print(f"正在載入數據: {DATA_PATH}")
    data = np.load(DATA_PATH)
    dataset_u = data['u'] # (20000, 128, 128)
    
    # ==========================================
    # 下採樣 (128x128 -> 64x64)
    # ==========================================
    # numpy 切片語法 [::2, ::2] 代表每隔 1 個像素取樣一次
    dataset_u_small = dataset_u[:, ::2, ::2] 
    
    # 更新 N 的大小
    N = dataset_u_small.shape[1] # 現在是 64
    print(f"下採樣後數據形狀: {dataset_u_small.shape}")
    
    # 0. 預計算遮罩 (針對 64x64 網格)
    mask_gpu = precalculate_distance_mask(N, RADIUS)
    
    # 1. 快速取得 Global R_max (使用下採樣後的數據)
    global_r_max = get_global_r_max(dataset_u_small, mask_gpu, sample_size=SAMPLE_SIZE_FOR_RMAX)
    
    # 2. 生成特徵 (使用下採樣後的數據)
    X_features = generate_final_features(dataset_u_small, mask_gpu, global_r_max, bins=BINS)
    
    print("\n================ 結果 ================")
    print(f"特徵矩陣形狀: {X_features.shape}")
    
    # 儲存
    np.savez('features_with_cm.npz', X=X_features, ids=data['ids'])
    print("特徵已儲存至 features_with_cm.npz")