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

# --- Paper Surrogate (Inverse of Paper CNN) ---
class PaperSurrogate(nn.Module):
    """
    PaperCNN 的逆架構：參數 -> 圖像
    架構：Linear (Input) -> ReLU -> Linear (Hidden) -> ReLU -> Unflatten -> ConvTranspose2d -> Sigmoid
    """
    def __init__(self, nk=5, np_size=5, nf=5, output_size=128):
        super(PaperSurrogate, self).__init__()
        
        self.nk = nk
        self.np_size = np_size
        
        # 計算 Unflatten 的特徵圖大小
        # ConvTranspose2d (stride=1, padding=0): H_out = H_in + kernel_size - 1
        # 我們希望 H_out = 128 => H_in = 128 - np_size + 1
        self.feature_map_size = output_size - np_size + 1
        self.flat_features = nk * self.feature_map_size * self.feature_map_size
        
        # 1. Input Layer (4 parameters)
        self.fc_in = nn.Linear(4, nf)
        
        # 2. Hidden Layer
        self.fc_hidden = nn.Linear(nf, self.flat_features)
        
        # 3. Deconvolutional Layer
        self.deconv1 = nn.ConvTranspose2d(in_channels=nk, out_channels=1, kernel_size=np_size, stride=1, padding=0)

    def forward(self, p):
        # p shape: (Batch, 4)
        
        # Layer 1: FC + ReLU
        x = self.fc_in(p)
        x = torch.relu(x)
        
        # Layer 2: FC + ReLU
        x = self.fc_hidden(x)
        x = torch.relu(x)
        
        # Unflatten
        x = x.view(x.size(0), self.nk, self.feature_map_size, self.feature_map_size)
        
        # Layer 3: Deconv + Sigmoid
        x = self.deconv1(x)
        return torch.sigmoid(x)

# --- Paper CNN (Minimal Architecture) ---
class PaperCNN(nn.Module):
    """
    實作論文 'Learning System Parameters from Turing Patterns' 中的極簡 CNN。
    架構：Conv2d -> ReLU -> Flatten -> Linear (Hidden) -> ReLU -> Linear (Output)
    論文標記法: (nk / np / nf)
    """
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperCNN, self).__init__()
        
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