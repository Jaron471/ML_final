import numpy as np
import os

def split_dataset_fixed(source_path, output_dir=".", seed=42, n_train=None, n_test=None, split_ratio=0.8):
    """
    將 .npz 資料集分割為 train 和 test
    如果指定 n_train/n_test 則使用固定數量
    否則使用 split_ratio (預設 0.8)
    """
    if not os.path.exists(source_path):
        print(f"❌ 找不到檔案: {source_path}")
        return

    print(f"📂 正在讀取: {source_path} ...")
    data = np.load(source_path)
    keys = list(data.files)
    
    # 1. 找出主要的資料長度
    total_samples = 0
    # 先找 'u' 或 'v'
    if 'u' in keys:
        total_samples = data['u'].shape[0]
    elif 'ids' in keys:
        total_samples = data['ids'].shape[0]
    else:
        # Fallback
        for k in keys:
            arr = data[k]
            if arr.ndim > 0 and arr.shape[0] > total_samples:
                total_samples = arr.shape[0]

    if total_samples == 0:
        print("❌ 錯誤：無法偵測到有效的樣本數據")
        return

    print(f"📊 總樣本數基準: {total_samples}")

    # ==========================================
    # 決定切分數量
    # ==========================================
    if n_train is None and n_test is None:
        # 使用比例
        n_trainval = int(total_samples * split_ratio)
        n_test = total_samples - n_trainval
    else:
        # 使用固定數量 (預設為原始設定)
        if n_train is None: n_train = 16000
        if n_test is None: n_test = 4000
        n_trainval = n_train
    
    # 檢查數量
    if n_trainval + n_test > total_samples:
        print(f"⚠️ 指定數量 ({n_trainval}+{n_test}) 超過總數 ({total_samples})")
        print(f"   自動調整為比例分配 ({split_ratio*100:.0f}%/{100-split_ratio*100:.0f}%)")
        n_trainval = int(total_samples * split_ratio)
        n_test = total_samples - n_trainval

    print(f"✂️  分割計畫: TrainVal={n_trainval}, Test={n_test}")
    
    # 隨機打亂索引（固定 seed）
    np.random.seed(seed)
    indices = np.random.permutation(total_samples)
    
    idx_trainval = indices[:n_trainval]
    idx_test = indices[n_trainval : n_trainval + n_test]
    
    # 準備儲存的字典
    data_trainval = {}
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
    # 預設儲存為 generic name，或根據需求修改
    out_train = os.path.join(output_dir, "turing_patterns_dataset_clean_train.npz")
    out_test = os.path.join(output_dir, "turing_patterns_dataset_clean_test.npz")
    
    np.savez_compressed(out_train, **data_trainval)
    np.savez_compressed(out_test, **data_test)
    
    print("✅ 完成！")
    print(f"   📦 {os.path.basename(out_train)}: {n_trainval} 樣本")
    print(f"   📦 {os.path.basename(out_test)}: {n_test} 樣本")

if __name__ == "__main__":
    # 對應 gm_mps.py 的輸出
    SOURCE_FILE = "turing_patterns_dataset_mps_0_20000.npz"
    if os.path.exists(SOURCE_FILE):
        split_dataset_fixed(SOURCE_FILE)
    else:
        print(f"⚠️  找不到 {SOURCE_FILE}，請確認檔名")