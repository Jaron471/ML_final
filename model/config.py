import torch
import os

# ==========================================
# 1. 硬體與路徑設定
# ==========================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

# 指向你合併後的大數據集 (確保這檔案在根目錄)
NPZ_PATH = "turing_patterns_dataset_merged.npz"

# 儲存路徑 (會由 train.py 動態修改，這裡只是預設值)
MODEL_SAVE_PATH = "checkpoint/model.pth"

# ==========================================
# 2. 物理模擬參數 (必須與數據生成時一致)
# ==========================================
N_GRID = 128
DX = 1.0
S_DIFFUSION = 0.4

# ==========================================
# 3. 訓練超參數
# ==========================================
BATCH_SIZE = 32        # 如果顯存(VRAM) > 8GB，建議改為 64
EPOCHS = 100           # 總訓練輪數
# TRAIN_VAL_RATIO = 0.8  # 訓練集比例 (0.8 = 80%訓練, 20%驗證)
TRAIN_RATIO = 0.6      # 60% 訓練
VAL_RATIO = 0.2        # 20% 驗證
TEST_RATIO = 0.2       # 20% 測試

# 學習率設定
LEARNING_RATE = 1e-4   # Inverse Model 建議用 1e-4
SCHEDULER_ETA_MIN = 1e-6

# ==========================================
# 4. Loss 權重策略 [a, b, c, delta]
# ==========================================
# 建議設定：[1.0, 5.0, 1.0, 5.0]
# 原因：
# - a, c (1.0): 給予標準關注
# - b, delta (5.0): 強力修正幾何特徵
LOSS_WEIGHTS = torch.tensor([1.0, 5.0, 1.0, 5.0]).to(DEVICE)

# ==========================================
# 5. 功能開關 (預設值，可被 train.py 覆蓋)
# ==========================================
USE_GRADNORM = False       # 是否使用 GradNorm
USE_PHYSICS_PHASE1 = False # 是否在 Phase 1 中使用 Physical Loss
LOSS_TYPE = "L1"           # "L1" or "MSE"

# ==========================================
# 6. WandB 設定
# ==========================================
WANDB_PROJECT = "Turing-Pattern-Unified"
WANDB_RUN_NAME = "Unified_Run"
