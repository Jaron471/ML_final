import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
import cupy as cp
import os
import wandb

# ==========================================
# 1. 設置與參數 (依據論文實驗設定)
# ==========================================
# 檔案路徑
FEATURE_PATH = 'features_with_cm.npz'       # 您的特徵檔 (由 extract_features_v2.py 生成)
LABEL_PATH = 'turing_patterns_dataset_merged.npz' # 您的原始數據檔 (提供 a, b, c, delta)
SAVE_PATH = 'rdh_ffnn_model.pth'            # 模型存檔路徑

# 硬體設定
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_CUPY = cp.cuda.is_available()
print(f"使用裝置: {DEVICE}")
print(f"CuPy 可用: {USE_CUPY}")
if USE_CUPY:
    print(f"CuPy CUDA 版本: {cp.cuda.runtime.runtimeGetVersion()}")
    print(f"GPU 裝置: {cp.cuda.Device(0).compute_capability}")

# 訓練超參數 (依據論文 5.1.10 與 5.4 節)
# 論文對大數據集使用 (20, 20) 的隱藏層結構 [cite: 1891-1892]
# 這裡我們可以設定為 [64, 64] 以獲得更好的穩定性，或是嚴格遵守論文改為 [20, 20]
HIDDEN_LAYERS = [20, 20] 
BATCH_SIZE = 32
LEARNING_RATE = 0.001    # Adam 預設
EPOCHS = 2000           # 論文訓練步數很多 (10^5 steps) [cite: 1690]
PATIENCE = 50            # 早停機制 (Early Stopping) [cite: 1689]

# WandB 設定
WANDB_PROJECT = "Turing-RDH-FFNN"
WANDB_RUN_NAME = "Paper-Method-RDH"
WANDB_API_KEY = "1e5a317026049b240da499f3aa7affb915f0b6ab"  # 與 new_train.py 相同

# ==========================================
# 2. 定義模型 (FFNN) - 依據論文 4.3.1
# ==========================================
class PaperFFNN(nn.Module):
    def __init__(self, input_dim=13, output_dim=4, hidden_units=[64, 64]):
        super(PaperFFNN, self).__init__()
        
        layers = []
        
        # 輸入層 -> 第一隱藏層 (ReLU) [cite: 1573-1576]
        # input_dim = 12 (RDH) + 1 (c_m) = 13
        layers.append(nn.Linear(input_dim, hidden_units[0]))
        layers.append(nn.ReLU())
        
        # 中間隱藏層 (Fully Connected + ReLU)
        for i in range(len(hidden_units) - 1):
            layers.append(nn.Linear(hidden_units[i], hidden_units[i+1]))
            layers.append(nn.ReLU())
            
        # 輸出層 (Linear, 無激活函數) [cite: 1578-1579]
        # 直接輸出歸一化後的參數值
        layers.append(nn.Linear(hidden_units[-1], output_dim))
        
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ==========================================
# 3. 專用資料集 (RDH Dataset)
# ==========================================
class RDHDataset(Dataset):
    def __init__(self, feature_path, label_path, use_cupy=True):
        # 1. 載入特徵 X (13維: 12 RDH + 1 c_m)
        print(f"正在載入特徵: {feature_path}")
        if not os.path.exists(feature_path):
            raise FileNotFoundError(f"找不到 {feature_path}，請先執行 extract_features_v2.py")
            
        f_data = np.load(feature_path)
        X_np = f_data['X'].astype(np.float32)
        
        # 2. 載入標籤 Y (a, b, c, delta)
        print(f"正在載入標籤: {label_path}")
        l_data = np.load(label_path)
        
        # 使用 CuPy 加速數據處理
        if use_cupy and USE_CUPY:
            print("使用 CuPy 加速數據預處理...")
            # 在 GPU 上堆疊和處理
            Y_cp = cp.stack([
                cp.asarray(l_data['a']), 
                cp.asarray(l_data['b']), 
                cp.asarray(l_data['c']), 
                cp.asarray(l_data['delta'])
            ], axis=1).astype(cp.float32)
            
            # 3. 標籤歸一化 (Target Preprocessing)
            # 依據論文 5.1.7 公式 (80): y'_{ij} = y_{ij} / max_l(y_{lj})
            # 使用 CuPy 在 GPU 上計算
            self.y_max = cp.asnumpy(cp.max(Y_cp, axis=0))
            Y_norm_cp = Y_cp / cp.asarray(self.y_max)
            
            # 轉回 NumPy (PyTorch 需要 NumPy 或 CPU tensor)
            self.X = X_np
            self.Y = cp.asnumpy(Y_cp)
            self.Y_norm = cp.asnumpy(Y_norm_cp)
            print("CuPy 預處理完成")
        else:
            # 使用 NumPy 處理（回退模式）
            print("使用 NumPy 進行數據預處理...")
            self.Y = np.stack([
                l_data['a'], 
                l_data['b'], 
                l_data['c'], 
                l_data['delta']
            ], axis=1).astype(np.float32)
            
            self.y_max = np.max(self.Y, axis=0)
            self.Y_norm = self.Y / self.y_max
            self.X = X_np
        
        print(f"數據準備完成: X shape={self.X.shape}, Y shape={self.Y.shape}")
        print(f"參數最大值 (用於反歸一化): {self.y_max}")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.from_numpy(self.Y_norm[idx])

    def get_y_max(self):
        return self.y_max

# ==========================================
# 4. 訓練與評估流程
# ==========================================
def compute_nrmse(preds, targets, use_cupy=True):
    """計算 NRMSE (用於驗證階段的快速評估)"""
    if use_cupy and USE_CUPY:
        # 使用 CuPy 在 GPU 上計算 NRMSE
        preds_cp = cp.asarray(preds)
        targets_cp = cp.asarray(targets)
        
        mse = cp.mean((preds_cp - targets_cp)**2, axis=0)
        rmse = cp.sqrt(mse)
        y_mean = cp.mean(targets_cp, axis=0)
        nrmse_per_param = rmse / y_mean
        
        # 轉回 NumPy
        return cp.asnumpy(nrmse_per_param), float(cp.mean(nrmse_per_param))
    else:
        # NumPy 版本（回退模式）
        mse = np.mean((preds - targets)**2, axis=0)
        rmse = np.sqrt(mse)
        y_mean = np.mean(targets, axis=0)
        nrmse_per_param = rmse / y_mean
        return nrmse_per_param, np.mean(nrmse_per_param)

def train_and_evaluate():
    # --- 步驟 A: 數據準備 ---
    full_dataset = RDHDataset(FEATURE_PATH, LABEL_PATH)
    y_max = full_dataset.get_y_max() # 之後評估要用
    
    # 依據論文 5.1.6: 60% Train, 20% Validation, 20% Test [cite: 1651-1652]
    total_size = len(full_dataset)
    train_size = int(0.6 * total_size)
    val_size = int(0.2 * total_size)
    test_size = total_size - train_size - val_size
    
    train_ds, val_ds, test_ds = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)
    
    # --- 初始化 WandB ---
    wandb.login(key=WANDB_API_KEY)
    wandb.init(
        project=WANDB_PROJECT,
        name=WANDB_RUN_NAME,
        config={
            "architecture": "FFNN",
            "hidden_layers": HIDDEN_LAYERS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "train_size": train_size,
            "val_size": val_size,
            "test_size": test_size,
            "input_features": "RDH_12 + c_m",
            "output_params": 4
        }
    )
    
    # --- 步驟 B: 模型初始化 ---
    model = PaperFFNN(input_dim=13, output_dim=4, hidden_units=HIDDEN_LAYERS).to(DEVICE)
    
    # 論文 5.1.10: 使用 Adam 優化器與 Mean-Squared Loss [cite: 1687]
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE) 
    criterion = nn.MSELoss() 
    
    # --- 步驟 C: 訓練迴圈 ---
    print(f"\n=== 開始訓練 FFNN (RDH -> Params) ===")
    best_val_loss = float('inf')
    patience_cnt = 0
    
    for epoch in range(EPOCHS):
        # 1. Training
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_ds)
        
        # 2. Validation
        model.eval()
        val_loss = 0.0
        val_preds_list = []
        val_targets_list = []
        
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * x.size(0)
                
                # 收集預測與真實值 (用於計算 NRMSE)
                pred_real = pred.cpu().numpy() * y_max
                target_real = y.cpu().numpy() * y_max
                val_preds_list.append(pred_real)
                val_targets_list.append(target_real)
                
        val_loss /= len(val_ds)
        
        # 計算驗證集 NRMSE
        val_preds = np.concatenate(val_preds_list, axis=0)
        val_targets = np.concatenate(val_targets_list, axis=0)
        nrmse_per_param, nrmse_joint = compute_nrmse(val_preds, val_targets)
        
        # 記錄到 WandB
        wandb.log({
            "epoch": epoch + 1,
            "lr": optimizer.param_groups[0]['lr'],
            "train/sup_loss": train_loss,
            "train/phy_loss": 0.0,  # 此方法無物理Loss
            "val/rmse_delta": nrmse_per_param[3],  # delta 是第4個參數
            "val/phy_loss": 0.0,  # 此方法無物理Loss
            "val/nrmse_joint": nrmse_joint,
            "val/nrmse_a": nrmse_per_param[0],
            "val/nrmse_b": nrmse_per_param[1],
            "val/nrmse_c": nrmse_per_param[2]
        })
        
        if (epoch+1) % 10 == 0:
            print(f"Epoch {epoch+1:4d} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | NRMSE: {nrmse_joint:.4f}")
            
        # 3. 早停機制 (Early Stopping) [cite: 1689]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_cnt = 0
            torch.save(model.state_dict(), SAVE_PATH) # 儲存最佳權重
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    print("訓練結束。")
    
    # 關閉 WandB 運行
    wandb.finish()
    
    # --- 步驟 D: 測試集最終評估 (NRMSE) ---
    print("\n=== 正在評估測試集 (Test Set Evaluation) ===")
    
    # 載入最佳模型
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    
    preds_list = []
    targets_list = []
    
    with torch.no_grad():
        for x, y_norm in test_loader:
            x = x.to(DEVICE)
            
            # 預測
            pred_norm = model(x).cpu().numpy()
            y_norm = y_norm.numpy()
            
            # 反歸一化 (還原成真實物理數值)
            pred_real = pred_norm * y_max
            target_real = y_norm * y_max
            
            preds_list.append(pred_real)
            targets_list.append(target_real)
            
    preds = np.concatenate(preds_list, axis=0)
    targets = np.concatenate(targets_list, axis=0)
    
    # 計算 NRMSE (Normalized Root Mean Square Error) [cite: 1376-1380]
    # 公式 (45): NRMSE = RMSE / mean(y_ij)
    # 使用 CuPy 加速最終評估計算
    
    if USE_CUPY:
        print("使用 CuPy 加速最終 NRMSE 計算...")
        preds_cp = cp.asarray(preds)
        targets_cp = cp.asarray(targets)
        
        mse = cp.mean((preds_cp - targets_cp)**2, axis=0)
        rmse = cp.sqrt(mse)
        y_mean = cp.mean(targets_cp, axis=0)
        
        nrmse_per_param = cp.asnumpy(rmse / y_mean)
        total_nrmse = float(cp.mean(rmse / y_mean))
    else:
        mse = np.mean((preds - targets)**2, axis=0)
        rmse = np.sqrt(mse)
        y_mean = np.mean(targets, axis=0)
        
        nrmse_per_param = rmse / y_mean
        total_nrmse = np.mean(nrmse_per_param)
    
    print(f"\n總體 NRMSE: {total_nrmse:.4f}")
    print("(論文標準: < 0.2 為良好, < 0.05 為優秀)")
    print("-" * 30)
    params = ['a', 'b', 'c', 'delta']
    for i, p in enumerate(params):
        print(f"參數 {p:5s} NRMSE: {nrmse_per_param[i]:.4f}")
        
    # 顯示前 3 筆樣本的對比
    print("\n[預測範例] (預測值 vs 真實值):")
    for i in range(3):
        print(f"樣本 {i+1}:")
        print(f"  Pred  : {preds[i]}")
        print(f"  Actual: {targets[i]}")

if __name__ == "__main__":
    if not os.path.exists(FEATURE_PATH):
        print(f"錯誤：找不到特徵檔 {FEATURE_PATH}")
        print("請先執行 extract_features_v2.py 生成特徵！")
    else:
        train_and_evaluate()