import torch
import torch.nn as nn

# --- 原本的 CNN (保持不變) ---
class ParameterNet(nn.Module):
    def __init__(self):
        super(ParameterNet, self).__init__()
        # ... (這裡是你原本的 CNN 程式碼) ...
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 4), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.regressor(x)
        return x

# --- 新增: MLP (Regression Baseline) ---
class MLPNet(nn.Module):
    def __init__(self, input_size=128*128):
        super(MLPNet, self).__init__()
        
        self.net = nn.Sequential(
            nn.Flatten(), # 把 (1, 128, 128) 壓扁成 (16384)
            
            # 第一層：輸入層 -> 隱藏層
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2), # 防止過擬合
            
            # 第二層
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # 第三層
            nn.Linear(512, 128),
            nn.ReLU(),
            
            # 輸出層 (4個參數)
            nn.Linear(128, 4),
            nn.Sigmoid() # 限制在 0~1 (配合 Scaler)
        )

    def forward(self, x):
        return self.net(x)

# --- Forward Surrogate (參數 -> 圖片) ---
class ForwardSurrogate(nn.Module):
    def __init__(self):
        super(ForwardSurrogate, self).__init__()
        # Input: 4 params -> Output: 128x128 image
        self.fc = nn.Sequential(
            nn.Linear(4, 256),
            nn.ReLU(),
            nn.Linear(256, 256 * 8 * 8),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1), # 8 -> 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 16 -> 32
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),   # 32 -> 64
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),    # 64 -> 128
            nn.Sigmoid() 
        )

    def forward(self, p):
        x = self.fc(p).view(-1, 256, 8, 8)
        return self.decoder(x)