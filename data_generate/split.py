import numpy as np
import os

def split_dataset_fixed(source_path, output_dir=".", seed=42):
    """
    將 .npz 資料集分割為固定的 16000 train+val 和 4000 test
    不使用 data_fraction（訓練時動態取樣）
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
    # 🔢 固定切分: 16000 (train+val pool) + 4000 (test)
    # ==========================================
    n_trainval = 16000
    n_test = 4000

    # 檢查數量
    if n_trainval + n_test > total_samples:
        print(f"❌ 錯誤: 指定數量 ({n_trainval}+{n_test}) 超過總數 ({total_samples})")
        return

    print(f"✂️  分割計畫: TrainVal Pool={n_trainval}, Test={n_test}")
    if (n_trainval + n_test) < total_samples:
        print(f"⚠️  注意: 有 {total_samples - (n_trainval + n_test)} 筆資料將被丟棄不使用。")
    
    # 隨機打亂索引（固定 seed）
    np.random.seed(seed)
    indices = np.random.permutation(total_samples)
    
    idx_trainval = indices[:n_trainval]
    idx_test = indices[n_trainval : n_trainval + n_test]
    
    # 準備儲存的字典
    data_train = {}
    data_val = {}
    data_test = {}
    
    # 2. 智慧遍歷並val = {}
    data_test = {}
    
    # 2. 智慧遍歷並切分
    for key in keys:
        array = data[key]
        
        # 判斷是否為主要資料 (長度符合 total_samples)
        if array.ndim > 0 and array.shape[0] == total_samples:
            data_trainval[key] = array[idx_trainval]
            data_test[key] = array[idx_test]
        else:
            # Metadata 直接複製
            print(f"   ℹ️  保留 Metadata: '{key}'")
            data_trainval[key] = array
            data_test[key] = array
    
    # 存檔 (train_data.npz 為 16000 的 pool，訓練時動態取樣)
    print("💾 正在儲存分割後的檔案...")
    np.savez_compressed(os.path.join(output_dir, "train_data.npz"), **data_trainval)
    np.savez_compressed(os.path.join(output_dir, "test_data.npz"), **data_test)
    
    print("✅ 完成！")
    print(f"   📦 train_data.npz: {n_trainval} 樣本 (訓練時根據 data_fraction 取樣)")
    print(f"   📦 test_data.npz: {n_test} 樣本")

if __name__ == "__main__":
    # 對應 gm_mps.py 的輸出
    SOURCE_FILE = "turing_patterns_dataset_mps_0_20000.npz"
    if os.path.exists(SOURCE_FILE):
        split_dataset_fixed(SOURCE_FILE)
    else:
        print(f"⚠️  找不到 {SOURCE_FILE}，請確認檔名")