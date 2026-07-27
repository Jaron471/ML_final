import torch
import torch.nn as nn


class MLPNet(nn.Module):
    def __init__(self, input_size=128*128, output_dim=4):
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
            
            # 輸出層（GM 為 4 個參數；三參數 PDE 可傳 output_dim=3）
            nn.Linear(128, output_dim),
            nn.Sigmoid() # 限制在 0~1 (配合 Scaler)
        )

    def forward(self, x):
        return self.net(x)


class PaperMinimalCNN(nn.Module):
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128, output_dim=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, nk, kernel_size=np_size)
        out_dim = input_size - np_size + 1
        self.fc1 = nn.Linear(nk * out_dim * out_dim, nf)
        self.fc_out = nn.Linear(nf, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc_out(x))


class PaperMinimalCNN2(nn.Module):
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128, output_dim=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, nk, kernel_size=np_size)
        self.conv2 = nn.Conv2d(nk, nk, kernel_size=np_size)
        out_dim = input_size - 2 * np_size + 2
        self.fc1 = nn.Linear(nk * out_dim * out_dim, nf)
        self.fc_out = nn.Linear(nf, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc_out(x))


class PaperMinimalCNN2MaxPool(nn.Module):
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128, output_dim=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, nk, kernel_size=np_size)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(nk, nk, kernel_size=np_size)
        self.pool2 = nn.MaxPool2d(2, 2)
        d1 = input_size - np_size + 1
        d2 = d1 // 2
        d3 = d2 - np_size + 1
        d4 = d3 // 2
        self.fc1 = nn.Linear(nk * d4 * d4, nf)
        self.fc_out = nn.Linear(nf, output_dim)

    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc_out(x))


class PaperMinimalCNN2Stride(nn.Module):
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128, output_dim=4):
        super().__init__()
        self.conv1 = nn.Conv2d(1, nk, kernel_size=np_size, stride=2)
        self.conv2 = nn.Conv2d(nk, nk, kernel_size=np_size, stride=2)
        d1 = (input_size - np_size) // 2 + 1
        d2 = (d1 - np_size) // 2 + 1
        self.fc1 = nn.Linear(nk * d2 * d2, nf)
        self.fc_out = nn.Linear(nf, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc_out(x))


MODEL_ARCHES = ("mlp", "cnn1", "cnn2", "cnn2pool", "cnn2stride")


def build_model(
    model_arch,
    nk=5,
    np_size=5,
    nf=5,
    input_size=128,
    output_dim=4,
):
    """Create a model from the architecture metadata stored in a checkpoint."""
    builders = {
        "mlp": lambda: MLPNet(
            input_size=input_size * input_size,
            output_dim=output_dim,
        ),
        "cnn1": lambda: PaperMinimalCNN(
            nk, np_size, nf, input_size, output_dim
        ),
        "cnn2": lambda: PaperMinimalCNN2(
            nk, np_size, nf, input_size, output_dim
        ),
        "cnn2pool": lambda: PaperMinimalCNN2MaxPool(
            nk, np_size, nf, input_size, output_dim
        ),
        "cnn2stride": lambda: PaperMinimalCNN2Stride(
            nk, np_size, nf, input_size, output_dim
        ),
    }
    try:
        return builders[model_arch.lower()]()
    except KeyError as exc:
        raise ValueError(
            f"Unknown model architecture {model_arch!r}; choose from {MODEL_ARCHES}"
        ) from exc
