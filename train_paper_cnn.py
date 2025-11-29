import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import wandb

# Import modules
from model import config
from model.dataset import TuringDataset
from model.utils import calculate_multidim_nrmse, get_next_version

# ==========================================
# 1. 定義論文中的極簡 CNN 架構
# ==========================================
class PaperMinimalCNN(nn.Module):
    """
    實作論文 'Learning System Parameters from Turing Patterns' 中的極簡 CNN。
    架構：Conv2d -> ReLU -> Flatten -> Linear (Hidden) -> ReLU -> Linear (Output)
    論文標記法: (nk / np / nf)
    """
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN, self).__init__()
        
        # 1. Convolutional Layer
        # 論文: "convolving the inputs with learnable kernels" [cite: 1814]
        # 輸入 Channel 固定為 1 (只看 u) [cite: 1936]
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        
        # 計算 Flatten 後的特徵數量
        # 無 Padding 下，輸出寬度 = Input - Kernel + 1
        out_dim = input_size - np_size + 1
        self.flat_features = nk * out_dim * out_dim
        
        # 2. Fully Connected Layer (Hidden)
        # 論文: "one fully-connected layer" [cite: 1820]
        self.fc1 = nn.Linear(self.flat_features, nf)
        
        # 3. Output Layer (4 parameters)
        self.fc_out = nn.Linear(nf, 4) 

    def forward(self, x):
        # x shape: (Batch, 1, 128, 128)
        
        # Layer 1: Conv + ReLU [cite: 1821]
        x = self.conv1(x)
        x = torch.relu(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Layer 2: FC + ReLU [cite: 1821]
        x = self.fc1(x)
        x = torch.relu(x)
        
        # Output Layer
        # 雖然論文沒特別提最後一層激活，但因為我們參數有 normalize 到 [0,1]，加 Sigmoid 比較合理
        x = self.fc_out(x)
        return torch.sigmoid(x)

# ==========================================
# 2. 訓練流程
# ==========================================
def train(args):
    # 設定 WandB
    wandb.init(project=config.WANDB_PROJECT, name=f"Paper-CNN-({args.nk}_{args.np}_{args.nf})", reinit=True)
    
    print(f"🚀 Training Paper-Style CNN: nk={args.nk}, np={args.np}, nf={args.nf}")
    
    # 載入數據
    print("📂 Loading datasets...")
    train_set = TuringDataset(config.TRAIN_PATH)
    val_set = TuringDataset(config.VAL_PATH)
    
    # 設定 Scaler (反正規化用)
    dataset = train_set
    dataset.scaler.to_device(config.DEVICE)
    
    train_loader = DataLoader(train_set, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 初始化模型
    model = PaperMinimalCNN(nk=args.nk, np_size=args.np, nf=args.nf, input_size=128).to(config.DEVICE)
    
    # 論文設定: Adam optimizer [cite: 1932]
    # 論文沒寫 LR，但通常極簡架構可以用較大 LR，這裡先用 1e-3
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    # 論文使用 MSE Loss [cite: 1932]
    criterion = nn.MSELoss()
    
    # 儲存設定
    ckpt_dir = "checkpoint"
    if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
    current_num = get_next_version(ckpt_dir, r"PaperCNN_.*_(?P<num>\d+)_.*\.pth")
    base_filename = f"PaperCNN_nk{args.nk}_np{args.np}_nf{args.nf}_{current_num}"
    best_model_path = os.path.join(ckpt_dir, f"{base_filename}_best.pth")
    
    best_nrmse = float('inf')
    
    # 訓練迴圈
    for epoch in range(config.EPOCHS):
        model.train()
        total_loss = 0
        
        for u_batch, _, params_target in tqdm(train_loader, desc=f"Ep {epoch+1}"):
            u_batch = u_batch.to(config.DEVICE)
            params_target = params_target.to(config.DEVICE)
            
            # 論文只使用 "raw simulation data of the first species" [cite: 1936]
            # 所以我們只取 Channel 0 (u)
            # TuringDataset 回傳的是 (Batch, 1, 128, 128)，剛好符合
            if u_batch.shape[1] > 1:
                u_input = u_batch[:, 0:1, :, :]
            else:
                u_input = u_batch
            
            optimizer.zero_grad()
            preds = model(u_input)
            
            loss = criterion(preds, params_target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        # Validation
        model.eval()
        all_targets = []
        all_preds = []
        with torch.no_grad():
            for u_batch, _, p_batch in val_loader:
                u_batch = u_batch.to(config.DEVICE)
                if u_batch.shape[1] > 1:
                    u_input = u_batch[:, 0:1, :, :]
                else:
                    u_input = u_batch
                    
                p_pred = model(u_input)
                all_targets.append(p_batch.cpu())
                all_preds.append(p_pred.cpu())
        
        all_targets = torch.cat(all_targets, dim=0).numpy()
        all_preds = torch.cat(all_preds, dim=0).numpy()
        nrmse = calculate_multidim_nrmse(all_targets, all_preds)
        
        avg_loss = total_loss / len(train_loader)
        print(f"Ep {epoch+1} | Train MSE: {avg_loss:.5f} | Val NRMSE: {nrmse:.2%}")
        
        wandb.log({"train/mse": avg_loss, "val/nrmse": nrmse})
        
        if nrmse < best_nrmse:
            best_nrmse = nrmse
            torch.save(model.state_dict(), best_model_path)
            print(f"🏆 Saved Best: {nrmse:.2%}")

    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 論文中提到的架構參數 [cite: 2093]
    parser.add_argument('--nk', type=int, default=5, help='Number of kernels (filters)')
    parser.add_argument('--np', type=int, default=5, help='Kernel size (np x np)')
    parser.add_argument('--nf', type=int, default=5, help='Neurons in Fully Connected layer')
    
    args = parser.parse_args()
    train(args)