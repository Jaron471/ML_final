import torch
import torch.nn as nn
import torch.nn.functional as F

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


class PaperMinimalCNN(nn.Module):
    """
    使用 circular padding 保留週期性邊界條件，不損失空間信息
    """
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN, self).__init__()
        
        self.np_size = np_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        
        # 使用 circular padding 後，輸出尺寸與輸入相同
        out_dim = input_size
        self.flat_features = nk * out_dim * out_dim
        
        self.fc1 = nn.Linear(self.flat_features, nf)
        self.fc_out = nn.Linear(nf, 4) 

    def forward(self, x):
        # Circular padding: (left, right, top, bottom)
        pad = self.np_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode='circular')
    def forward(self, x):
        x = torch.relu(self.conv1(x))

class PaperMinimalCNN2(nn.Module):
    """
    2層 CNN，使用 circular padding 保留週期性邊界條件
    """
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN2, self).__init__()
        
        self.np_size = np_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=nk, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        
        # 使用 circular padding 後，輸出尺寸與輸入相同
        out_dim = input_size
        self.flat_features = nk * out_dim * out_dim
        
        self.fc1 = nn.Linear(self.flat_features, nf)
        self.fc_out = nn.Linear(nf, 4) 

    def forward(self, x):
        # Layer 1 with circular padding
        pad = self.np_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode='circular')
        x = torch.relu(self.conv1(x))

class PaperMinimalCNN2MaxPool(nn.Module):
    """
    2層 CNN + MaxPool，使用 circular padding 保留週期性邊界條件
    """
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN2MaxPool, self).__init__()
        
        self.np_size = np_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=nk, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 使用 circular padding 保持尺寸，然後 pooling 減半
        d1 = input_size // 2  # After pool1
        d2 = d1 // 2          # After pool2
        
        self.flat_features = nk * d2 * d2
        
        self.fc1 = nn.Linear(self.flat_features, nf)
        self.fc_out = nn.Linear(nf, 4) 

    def forward(self, x):
        # Layer 1 with circular padding
        pad = self.np_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode='circular')
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        
        # Layer 2 with circular padding
        x = F.pad(x, (pad, pad, pad, pad), mode='circular')
        x = torch.relu(self.conv2(x))

class PaperMinimalCNN2Stride(nn.Module):
    """
    2層 CNN + Stride，使用 circular padding 保留週期性邊界條件
    """
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN2Stride, self).__init__()
        
        self.np_size = np_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=np_size, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=nk, out_channels=nk, kernel_size=np_size, stride=2, padding=0)
        
        # 使用 circular padding 保持尺寸，stride 減半兩次
        d1 = input_size // 2  # After conv1 (stride 2)
        d2 = d1 // 2          # After conv2 (stride 2)
        
        self.flat_features = nk * d2 * d2
        
        self.fc1 = nn.Linear(self.flat_features, nf)
        self.fc_out = nn.Linear(nf, 4) 

    def forward(self, x):
        # Layer 1 with circular padding and stride
        pad = self.np_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode='circular')
        x = torch.relu(self.conv1(x))
        
        # Layer 2 with circular padding and stride
        x = F.pad(x, (pad, pad, pad, pad), mode='circular')
        x = torch.relu(self.conv2(x))
        
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return torch.sigmoid(x)