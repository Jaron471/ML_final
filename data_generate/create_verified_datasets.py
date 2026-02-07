import numpy as np
import os
import sys

# 引用現有的 split 邏輯 (確保它能從當前目錄或子目錄正確引用)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from split import split_dataset_fixed
except ImportError:
    # Fallback for running from root
    from data_generate.split import split_dataset_fixed

# ==========================================
# 設定
# ==========================================
VARIANCE_THRESHOLD = 0.01
ROUGHNESS_THRESHOLD = 0.20

# ==========================================
# 輔助函數
# ==========================================

def calculate_roughness(image):
    """
    計算圖像的粗糙度 (Total Variation / (Std * Size))
    數值越低表示圖像越平滑、區塊越大 (Low frequency)
    """
    tv = np.sum(np.abs(np.diff(image, axis=0))) + np.sum(np.abs(np.diff(image, axis=1)))
    std = np.std(image)
    if std < 1e-6: return 0 
    return tv / (std * image.size)

def analyze_pattern(u, v, threshold=0.01):
    """
    分析圖靈斑紋是否形成 (Variance check)
    """
    u_std = np.std(u)
    v_std = np.std(v)
    return (u_std > threshold) or (v_std > threshold)

def get_keys(data):
    """分離樣本數據鍵和元數據鍵"""
    keys = data.files
    n_samples = len(data['u'])
    sample_keys = []
    meta_keys = []
    
    for k in keys:
        arr = data[k]
        is_sample_data = False
        if hasattr(arr, 'shape') and len(arr.shape) > 0:
             if arr.shape[0] == n_samples:
                 is_sample_data = True
        
        if is_sample_data and k not in ['N_grid', 'dt', 'dx', 's_diffusion', 'noise_std']:
            sample_keys.append(k)
        else:
            meta_keys.append(k)
    return sample_keys, meta_keys

# ==========================================
# 核心功能模組 (只負責清洗)
# ==========================================

def merge_and_clean_full_dataset(input_paths, output_clean_path):
    """
    讀取多個檔案 -> 合併 -> 清洗 -> 輸出單一乾淨檔案
    """
    print(f"\n[Process] 處理合併與清洗: {input_paths} -> {output_clean_path}")
    
    # 1. Load & Merge in memory
    merged_data = None
    sample_keys = []
    meta_keys = []
    first_data = None
    
    total_raw_samples = 0
    data_list = []

    # 讀取所有檔案
    for p in input_paths:
        if not os.path.exists(p):
            print(f"Warning: Skipping missing file {p}")
            continue
        d = np.load(p)
        print(f"  - Loading {p} ({len(d['u'])} samples)")
        data_list.append(d)
        total_raw_samples += len(d['u'])
        
        if first_data is None:
            first_data = d
            sample_keys, meta_keys = get_keys(d)

    if not data_list:
        print("Error: No valid data loaded.")
        return

    # 合併 array
    print(f"  - Merging {total_raw_samples} samples...")
    full_arrays = {}
    for k in sample_keys:
        arrays = [d[k] for d in data_list]
        full_arrays[k] = np.concatenate(arrays, axis=0)

    # 處理 ids (若合併後重複或缺失，這裡重新生成連續 ID 以確保唯一性，或者沿用)
    # 為了安全，如果發現 ID 重複嚴重，可能需要重置。但原本 dataset 應該是獨立的。
    # 簡單起見，我們沿用 concatenate 的 ids
    if 'ids' not in full_arrays:
        full_arrays['ids'] = np.arange(total_raw_samples)

    # 2. Clean Logic
    u_all = full_arrays['u']
    v_real = full_arrays['v'] # 使用真實 V，假設鍵名一致
    # 注意: 有些 dataset 鍵名為 v_star，這裡依賴 get_keys 自動抓取樣本鍵，但 analyze_pattern 需要明確指定
    # 這裡假設 sample_keys 包含 'u' 和 'v'
    
    keep_mask = np.ones(total_raw_samples, dtype=bool)
    cnt_no_pattern = 0
    cnt_bad_blob = 0
    
    print("  - Cleaning (Variance & Roughness Check)...")
    for i in range(total_raw_samples):
        # Variance Check
        if not analyze_pattern(u_all[i], v_real[i], VARIANCE_THRESHOLD):
            keep_mask[i] = False
            cnt_no_pattern += 1
            continue
            
        # Roughness Check
        if calculate_roughness(u_all[i]) < ROUGHNESS_THRESHOLD:
            keep_mask[i] = False
            cnt_bad_blob += 1
    
    print(f"    * Removed No-Pattern: {cnt_no_pattern}")
    print(f"    * Removed Bad-Blob:   {cnt_bad_blob}")
    
    # 3. Save Clean File
    final_data = {}
    for k in sample_keys:
        final_data[k] = full_arrays[k][keep_mask]
        
    for k in meta_keys:
        final_data[k] = first_data[k] # Metadata from first file
        
    cleaned_count = np.sum(keep_mask)
    print(f"  => Saving Cleaned Dataset: {output_clean_path} ({cleaned_count} samples)")
    np.savez(output_clean_path, **final_data)

# ==========================================
# 主流程
# ==========================================

if __name__ == "__main__":
    # 決定路徑
    cwd = os.getcwd()
    if os.path.basename(cwd) == 'data_generate':
        root_dir = ".."
    else:
        root_dir = "."
        
    # 輸入: 原始合併檔 + 測試檔 (確保覆蓋所有數據)
    # 為了避免重複讀取 (如果 merged 已經包含了 test)，使用者需確認。
    # 根據之前的 context，merged.npz 可能包含也可能不包含 test。
    # 但為了 "全量處理"，我們讀取所有可能的源頭。
    # 如果 merged 已經包含了 train+val+test，那只讀 merged 即可。
    # 假設這是一個 "重做" 流程，我們讀取 merged 和 test。
    
    input_files = [
        os.path.join(root_dir, "turing_patterns_dataset_merged.npz"),
        os.path.join(root_dir, "test_data.npz")
    ]
    
    # 中間暫存檔 (Cleaned Full Dataset)
    all_clean_path = os.path.join(root_dir, "all_data_clean_temp.npz")
    
    # 1. Merge & Clean -> Single File
    if any(os.path.exists(f) for f in input_files):
        merge_and_clean_full_dataset(input_files, all_clean_path)
    else:
        print("Error: Input files not found.")
        sys.exit(1)
    
    # 2. Call existing Split function
    # 使用修改後的 split_dataset_fixed，它現在支持自定義輸出名稱和比例
    print("\n[Split] 調用 split.py 進行分割...")
    
    # 我們希望 80/20 切分，不指定固定數量 (n_train=None, n_test=None -> 使用 ratio)
    split_dataset_fixed(
        all_clean_path, 
        output_dir=root_dir, 
        seed=42, 
        n_train=None, 
        n_test=None, 
        split_ratio=0.8
    )
    
    # Clean up temp
    if os.path.exists(all_clean_path):
        os.remove(all_clean_path)
        print(f"Removed temp file: {all_clean_path}")
        
    print("\n[Done] All Process Finished.")
