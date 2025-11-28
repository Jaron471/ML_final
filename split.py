import numpy as np
import os

def split_dataset_fixed(source_path, output_dir=".", seed=42):
    """
    將 .npz 資料集分割為指定數量的 Train, Val, Test
    """
    if not os.path.exists(source_path):
        print(f"❌ 找不到檔案: {source_path}")
        return

    print(f"📂 正在讀取: {source_path} ...")
    data = np.load(source_path)
    keys = list(data.files)
    
    # 1. 找出主要的資料長度
    total_samples = 0
    for k in keys:
        arr = data[k]
        if arr.ndim > 0 and arr.shape[0] > total_samples:
            total_samples = arr.shape[0]

    if total_samples == 0:
        print("❌ 錯誤：無法偵測到有效的樣本數據")
        return

    print(f"📊 總樣本數基準: {total_samples}")

    # ==========================================
    # 🔢 設定切分數量 (在此修改)
    # ==========================================
    n_train = 16000
    n_val   = 2000
    
    # 選項 A: 把剩下的全部當作 Test (推薦，會變成 2000 筆)
    n_test  = total_samples - n_train - n_val
    
    # 選項 B: 如果你真的只要 200 筆 (會丟棄 1800 筆資料)，請解開下面這行註解
    # n_test = 200 

    # 檢查數量
    if n_train + n_val + n_test > total_samples:
        print(f"❌ 錯誤: 指定數量 ({n_train}+{n_val}+{n_test}) 超過總數 ({total_samples})")
        return

    print(f"✂️  分割計畫: Train={n_train}, Val={n_val}, Test={n_test}")
    if (n_train + n_val + n_test) < total_samples:
        print(f"⚠️  注意: 有 {total_samples - (n_train + n_val + n_test)} 筆資料將被丟棄不使用。")
    
    # 隨機打亂索引
    np.random.seed(seed)
    indices = np.random.permutation(total_samples)
    
    idx_train = indices[:n_train]
    idx_val   = indices[n_train : n_train + n_val]
    idx_test  = indices[n_train + n_val : n_train + n_val + n_test]
    
    # 準備儲存的字典
    data_train = {}
    data_val = {}
    data_test = {}
    
    # 2. 智慧遍歷並切分
    for key in keys:
        array = data[key]
        
        # 判斷是否為主要資料 (長度符合 total_samples)
        if array.ndim > 0 and array.shape[0] == total_samples:
            data_train[key] = array[idx_train]
            data_val[key]   = array[idx_val]
            data_test[key]  = array[idx_test]
        else:
            # Metadata 直接複製
            print(f"   ℹ️  保留 Metadata: '{key}'")
            data_train[key] = array
            data_val[key]   = array
            data_test[key]  = array
    
    # 存檔
    print("💾 正在儲存分割後的檔案...")
    np.savez_compressed(os.path.join(output_dir, "train_data.npz"), **data_train)
    np.savez_compressed(os.path.join(output_dir, "val_data.npz"), **data_val)
    np.savez_compressed(os.path.join(output_dir, "test_data.npz"), **data_test)
    
    print("✅ 完成！")

if __name__ == "__main__":
    SOURCE_FILE = "turing_patterns_dataset_merged.npz"
    split_dataset_fixed(SOURCE_FILE)