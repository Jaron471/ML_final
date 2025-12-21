import torch
import torch.nn as nn

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