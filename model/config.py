import torch

# 硬體與路徑 (保持不變)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NPZ_PATH = "turing_patterns_dataset_merged.npz"
MODEL_SAVE_PATH = "turing_model.pth" # 檔名會由 train.py 動態修改

# 物理參數 (保持不變)
N_GRID = 128
DX = 1.0
S_DIFFUSION = 0.4

# 訓練超參數
BATCH_SIZE = 32
EPOCHS = 100
LEARNING_RATE = 1e-4 # MLP 可能需要小一點的 LR
LOSS_WEIGHTS = torch.tensor([1.0, 5.0, 0.0, 5.0]).to(DEVICE)
LAMBDA_PHY = 0.1

# ==========================================
# 🧪 實驗控制中心 (修改這裡來切換實驗)
# ==========================================

# 1. 選擇模型架構: "CNN" 或 "MLP"
MODEL_TYPE = "CNN" 

# 2. 是否使用 PINN (物理 Loss): True 或 False
USE_PHYSICS = True

# 自動生成 WandB 專案名稱 (不用改)
WANDB_PROJECT = "Turing-Pattern-Comparison"
WANDB_RUN_NAME = f"{MODEL_TYPE}_{'PINN' if USE_PHYSICS else 'PureData'}"