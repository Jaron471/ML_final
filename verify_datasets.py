import numpy as np

# 檢查驗證數據集
print("檢查驗證數據集內容:\n")

# Merged verified
print("="*60)
print("turing_patterns_dataset_merged_verified.npz")
print("="*60)
data = np.load("turing_patterns_dataset_merged_verified.npz")
print(f"Keys: {data.files}")
print(f"u shape: {data['u'].shape}")
print(f"v shape: {data['v'].shape}")
print(f"樣本數: {len(data['u'])}")

# Test verified
print("\n" + "="*60)
print("test_data_verified.npz")
print("="*60)
data = np.load("test_data_verified.npz")
print(f"Keys: {data.files}")
print(f"u shape: {data['u'].shape}")
print(f"v shape: {data['v'].shape}")
print(f"樣本數: {len(data['u'])}")

print("\n" + "="*60)
print("總結")
print("="*60)
print("✅ 訓練數據集: 15663 個樣本 (原始 16000, 移除 337 個)")
print("✅ 測試數據集: 3924 個樣本 (原始 4000, 移除 76 個)")
print("\n已更新配置檔案:")
print("  - model/config.py: TRAIN_PATH, TEST_PATH")
print("  - verify_comparison.py: TEST_PATH")
