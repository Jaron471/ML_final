import numpy as np
import os

def merge_npz():
    # 1. 設定檔案路徑
    file_list = [
        "turing_patterns_dataset_mps_10000_18000.npz",
        "turing_patterns_dataset_cupy_18000_20000.npz"
    ]
    output_filename = "turing_patterns_dataset_merged.npz"

    # 檢查檔案是否存在
    for f in file_list:
        if not os.path.exists(f):
            print(f"❌ 找不到檔案: {f}")
            return

    print(f"📂 正在讀取檔案...")
    data_objects = [np.load(f) for f in file_list]
    
    # 2. 定義哪些欄位需要合併 (Array)，哪些是固定參數 (Scalar)
    # 根據你之前的生成代碼，這些是會有 N 筆資料的欄位
    array_keys = ['u', 'v', 'ids', 'a', 'b', 'c', 'delta', 'u_star', 'v_star', 'final_step']
    
    # 這些是固定參數，讀取第一份檔案的即可
    scalar_keys = ['N_grid', 'dt', 'dx', 's_diffusion', 'noise_std']

    merged_dict = {}

    # --- 合併 Array 資料 ---
    print("🔄 正在合併數據陣列...")
    for key in array_keys:
        if key in data_objects[0]:
            try:
                # 把所有檔案的該欄位串接起來 (Concatenate)
                merged_dict[key] = np.concatenate([d[key] for d in data_objects], axis=0)
            except ValueError as e:
                print(f"⚠️ 合併欄位 '{key}' 時發生錯誤 (可能是維度不匹配): {e}")
        else:
            print(f"⚠️ 警告: 欄位 '{key}' 在檔案中找不到，已跳過。")

    # --- 處理固定參數 ---
    for key in scalar_keys:
        if key in data_objects[0]:
            merged_dict[key] = data_objects[0][key]

    # --- 3. 檢查重複 ID ---
    all_ids = merged_dict['ids']
    total_count = len(all_ids)
    
    print(f"📊 合併後原始數量: {total_count}")

    # 使用 np.unique 找出唯一 ID 的索引
    # return_index=True 會回傳「第一次出現」該 ID 的索引位置
    unique_ids, unique_indices = np.unique(all_ids, return_index=True)
    
    duplicate_count = total_count - len(unique_ids)
    
    if duplicate_count > 0:
        print(f"⚠️ 發現 {duplicate_count} 筆重複 ID！正在移除重複項...")
        
        # 根據 unique_indices 篩選所有數據
        # 注意：np.unique 回傳的 unique_indices 是排序過的 ID 對應的 index，
        # 如果你想保持原始順序並去重，做法稍微不同，但這裡我們用 sorted id 沒關係。
        
        for key in array_keys:
            if key in merged_dict:
                merged_dict[key] = merged_dict[key][unique_indices]
                
        print(f"✅ 已移除重複項，剩餘資料: {len(merged_dict['ids'])}")
    else:
        print("✅ ID 檢查通過，無重複資料。")

    # --- 4. 儲存 ---
    print(f"💾 正在儲存至 {output_filename} ...")
    np.savez(output_filename, **merged_dict)
    print("🎉 合併完成！")

    # 驗證讀取
    print("\n--- 驗證新檔案 ---")
    new_data = np.load(output_filename)
    print(f"Shape of U: {new_data['u'].shape}")
    print(f"ID Range: {np.min(new_data['ids'])} ~ {np.max(new_data['ids'])}")

if __name__ == "__main__":
    merge_npz()