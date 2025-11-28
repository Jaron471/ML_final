import torch
import os

# ==========================================
# 1. 硬體與路徑設定
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 指向你合併後的大數據集 (確保這檔案在根目錄)
NPZ_PATH = "turing_patterns_dataset_merged.npz"

# 舊版 train.py 用的儲存路徑 (new_train.py 會自動存到 checkpoint/ 資料夾)
MODEL_SAVE_PATH = "checkpoint/inverse_cnn_hybrid.pth"

# ==========================================
# 2. 物理模擬參數 (必須與數據生成時一致)
# ==========================================
N_GRID = 128
DX = 1.0
S_DIFFUSION = 0.4

# ==========================================
# 3. 訓練超參數 (給 Inverse CNN 使用)
# ==========================================
BATCH_SIZE = 32        # 如果顯存(VRAM) > 8GB，建議改為 64 以加快 Surrogate 訓練
EPOCHS = 100           # Phase 2 & 3 的總訓練輪數
TRAIN_VAL_RATIO = 0.8  # 訓練集比例 (0.8 = 80%訓練, 20%驗證)

# 學習率設定
# Phase 1 (Surrogate) 通常用 1e-3 (在 new_train.py 內寫死)
# Phase 2 (Inverse) 建議用慢一點的 1e-4 以求精準
LEARNING_RATE = 1e-4   

# ==========================================
# 4. Loss 權重策略 [a, b, c, delta]
# ==========================================
# 這是 Phase 2 (Supervised Loss) 的核心
# 建議設定：[1.0, 5.0, 1.0, 5.0]
# 原因：
# - a, c (1.0): 給予標準關注，不放棄 c。
# - b, delta (5.0): 強力修正幾何特徵，解決 delta 卡在 5.3% 的問題。
LOSS_WEIGHTS = torch.tensor([1.0, 5.0, 1.0, 5.0]).to(DEVICE)

# 是否使用 GradNorm 動態權重調整 (Phase 2&3)
USE_GRADNORM = False

# 是否在 Phase 1 中使用 Physical Loss
USE_PHYSICS_PHASE1 = False

# ==========================================
# 5. 其他設定 (WandB / 舊版相容)
# ==========================================
WANDB_PROJECT = "Turing-Pattern-Surrogate"
WANDB_RUN_NAME = "Hybrid_Cycle_Run"
