import torch

# 硬體與路徑 (保持不變)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
NPZ_PATH = "turing_patterns_dataset_merged.npz"

# verify要使用，train部分要改下面的
MODEL_SAVE_PATH = "CNN_PINN.pth" # 檔名會由 train.py 動態修改

# 物理參數 (保持不變)
N_GRID = 128
DX = 1.0
S_DIFFUSION = 0.4

# 訓練超參數
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4 # MLP 可能需要小一點的 LR
TRAIN_VAL_RATIO = 0.8  # 訓練集比例 (0.8 = 80%訓練, 20%驗證)
# Loss_type 有 MSE or L1
LOSS_TYPE = "L1"
LOSS_WEIGHTS = torch.tensor([1.0, 5.0, 1.0, 5.0]).to(DEVICE)
LAMBDA_PHY = 0.1

# ==========================================
# 🧪 實驗控制中心 (修改這裡來切換實驗)
# ==========================================

# 1. 選擇模型架構: "CNN" 或 "MLP"
MODEL_TYPE = "CNN" 

# 2. 是否使用 PINN (物理 Loss): True 或 False
USE_PHYSICS = False

# 3. 是否使用 GradNorm 動態權重調整: True 或 False
USE_GRADNORM = False  # 預設關閉，需要手動開啟測試

# 自動生成 WandB 專案名稱 (不用改)
WANDB_PROJECT = "Turing-Pattern-PINN"

#wandb train name 改這裡
WANDB_RUN_NAME = f"{MODEL_TYPE}_{'PINN' if USE_PHYSICS else 'PureData'}_20000_newloss_L1"


# --- Scheduler 設定 (Cosine Annealing) ---
SCHEDULER_T_MAX = EPOCHS      # Cosine annealing 的週期 (通常設為總epoch數)
SCHEDULER_ETA_MIN = 1e-6      # 最小學習率