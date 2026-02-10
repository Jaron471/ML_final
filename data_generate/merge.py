print("Starting merge_npz.py...")

import numpy as np
import os

print("Imports complete.")

def merge_npz():
    # --- 1. 設定檔案路徑 (舉例) ---
    file_list = [
        "turing_patterns_dataset_cupy_1_10000.npz",
        "turing_patterns_dataset_mps_10000_15000.npz",
        "turing_patterns_dataset_cupy_15000_20000.npz"
    ]
    
    output_filename = "turing_patterns_dataset_merged.npz"

    # 檢查檔案是否存在
    valid_files = []
    print(f"🔍 開始檢查檔案...")
    for f in file_list:
        if os.path.exists(f):
            print(f"  ✅ 找到: {f}")
            valid_files.append(f)
        else:
            print(f"  ❌ 找不到: {f} (請確認檔名或路徑)")
    
    if not valid_files:
        print("❌ 沒有找到任何有效檔案，終止合併。")
        return

    print(f"\n📂 正在讀取並合併 {len(valid_files)} 個檔案...")
    data_objects = [np.load(f) for f in valid_files]
    
    # 2. 定義哪些欄位需要合併 (Array)，哪些是固定參數 (Scalar)
    array_keys = ['u', 'v', 'ids', 'a', 'b', 'c', 'delta', 'u_star', 'v_star', 'final_step']
    # 固定參數讀取第一份檔案的即可
    scalar_keys = ['N_grid', 'dt', 'dx', 's_diffusion', 'noise_std']

    merged_dict = {}

    # --- 合併 Array 資料 ---
    print("🔄 正在串接數據陣列...")
    for key in array_keys:
        # 檢查第一份資料有沒有這個 key
        if key in data_objects[0]:
            try:
                # 把所有檔案的該欄位串接起來
                merged_dict[key] = np.concatenate([d[key] for d in data_objects if key in d], axis=0)
            except ValueError as e:
                print(f"⚠️ 合併欄位 '{key}' 時發生錯誤 (可能是維度不匹配): {e}")
        else:
            print(f"⚠️ 警告: 欄位 '{key}' 在第一個檔案中找不到，已跳過。")

    # --- 處理固定參數 ---
    for key in scalar_keys:
        if key in data_objects[0]:
            merged_dict[key] = data_objects[0][key]

    # --- 3. 檢查重複 ID 並排序 ---
    all_ids = merged_dict['ids']
    total_count = len(all_ids)
    
    print(f"📊 串接後總數量: {total_count}")

    # 使用 np.unique 找出唯一 ID 的索引
    # return_index=True 回傳的是「排序後」的唯一值在原陣列中的索引
    # 這一步會同時完成「去重」和「根據 ID 排序」
    unique_ids, unique_indices = np.unique(all_ids, return_index=True)
    
    duplicate_count = total_count - len(unique_ids)
    
    if duplicate_count > 0:
        print(f"🧹 發現 {duplicate_count} 筆重複 ID，正在移除並重新排序...")
    else:
        print("✅ 無重複 ID，正在依 ID 順序整理數據...")

    # 根據 unique_indices 篩選所有數據 (這會讓資料依照 ID 從小到大排列)
    for key in array_keys:
        if key in merged_dict:
            merged_dict[key] = merged_dict[key][unique_indices]

    final_count = len(merged_dict['ids'])
    print(f"✨ 最終資料筆數: {final_count}")

    # --- 4. 儲存 ---
    print(f"💾 正在儲存至 {output_filename} ...")
    np.savez(output_filename, **merged_dict)
    print("🎉 合併完成！")

    # 驗證讀取
    print("\n--- 驗證新檔案 ---")
    new_data = np.load(output_filename)
    min_id = np.min(new_data['ids'])
    max_id = np.max(new_data['ids'])
    
    print(f"Shape of U: {new_data['u'].shape}")
    print(f"ID Range: {min_id} ~ {max_id}")
    
    # 簡單檢查是否有缺漏 (如果 ID 是連續的)
    expected_count = max_id - min_id + 1
    if final_count == expected_count:
        print("✅ ID 連續無中斷")
    else:
        print(f"⚠️ ID可能有中斷 (範圍涵蓋 {expected_count} 筆，但實際只有 {final_count} 筆)")

if __name__ == "__main__":
    merge_npz()