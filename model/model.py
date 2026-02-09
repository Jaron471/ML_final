import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ==========================================
# Original Models (Keep for compatibility)
# ==========================================

class MLPNet(nn.Module):
    def __init__(self, input_size=128*128):
        super(MLPNet, self).__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)

class PaperMinimalCNN(nn.Module):
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN, self).__init__()
        self.np_size = np_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        out_dim = input_size
        self.flat_features = nk * out_dim * out_dim
        self.fc1 = nn.Linear(self.flat_features, nf)
        self.fc_out = nn.Linear(nf, 4) 

    def forward(self, x):
        pad = self.np_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode='circular')
        x = torch.relu(self.conv1(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return torch.sigmoid(x)

class PaperMinimalCNN2(nn.Module):
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN2, self).__init__()
        self.np_size = np_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels=nk, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        out_dim = input_size
        self.flat_features = nk * out_dim * out_dim
        self.fc1 = nn.Linear(self.flat_features, nf)
        self.fc_out = nn.Linear(nf, 4) 

    def forward(self, x):
        pad = self.np_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode='circular')
        x = torch.relu(self.conv1(x))
        x = F.pad(x, (pad, pad, pad, pad), mode='circular')
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return torch.sigmoid(x)

class PaperMinimalCNN2MaxPool(nn.Module):
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN2MaxPool, self).__init__()
        self.np_size = np_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=nk, out_channels=nk, kernel_size=np_size, stride=1, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        d2 = (input_size // 2) // 2
        self.flat_features = nk * d2 * d2
        self.fc1 = nn.Linear(self.flat_features, nf)
        self.fc_out = nn.Linear(nf, 4) 

    def forward(self, x):
        pad = self.np_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode='circular')
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.pad(x, (pad, pad, pad, pad), mode='circular')
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return torch.sigmoid(x)

class PaperMinimalCNN2Stride(nn.Module):
    def __init__(self, nk=5, np_size=5, nf=5, input_size=128):
        super(PaperMinimalCNN2Stride, self).__init__()
        self.np_size = np_size
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=nk, kernel_size=np_size, stride=2, padding=0)
        self.conv2 = nn.Conv2d(in_channels=nk, out_channels=nk, kernel_size=np_size, stride=2, padding=0)
        d2 = (input_size // 2) // 2
        self.flat_features = nk * d2 * d2
        self.fc1 = nn.Linear(self.flat_features, nf)
        self.fc_out = nn.Linear(nf, 4) 

    def forward(self, x):
        pad = self.np_size // 2
        x = F.pad(x, (pad, pad, pad, pad), mode='circular')
        x = torch.relu(self.conv1(x))
        x = F.pad(x, (pad, pad, pad, pad), mode='circular')
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return torch.sigmoid(x)

class PaperFlexibleCNN(nn.Module):
    def __init__(self, nk=5, np_size=5, nf=5, layers=2, sampling='none', input_size=128):
        super(PaperFlexibleCNN, self).__init__()
        self.np_size = np_size
        self.layers = layers
        self.sampling = sampling
        
        self.conv_layers = nn.ModuleList()
        self.pools = nn.ModuleList() # Only used if pooling
        
        current_dim = input_size
        in_c = 1
        
        for i in range(layers):
            stride = 2 if sampling == 'stride' else 1
            dilation = 2**i if sampling == 'dilated' else 1
            
            conv = nn.Conv2d(in_channels=in_c, out_channels=nk, kernel_size=np_size, stride=stride, padding=0, dilation=dilation)
            self.conv_layers.append(conv)
            
            if sampling == 'stride':
                current_dim = current_dim // 2
            elif sampling == 'maxpool':
                self.pools.append(nn.MaxPool2d(kernel_size=2, stride=2))
                current_dim = current_dim // 2
            elif sampling == 'avgpool':
                self.pools.append(nn.AvgPool2d(kernel_size=2, stride=2))
                current_dim = current_dim // 2
            # dilated: current_dim stays same
            
            in_c = nk # Next layer input channels = nk
            
        self.flat_features = nk * current_dim * current_dim
        self.fc1 = nn.Linear(self.flat_features, nf)
        self.fc_out = nn.Linear(nf, 4)

    def forward(self, x):
        pad_base = self.np_size // 2
        
        for i in range(self.layers):
            # Circular padding before conv
            # Adjust padding for dilation
            dilation = self.conv_layers[i].dilation[0]
            pad = pad_base * dilation
            
            x = F.pad(x, (pad, pad, pad, pad), mode='circular')
            x = torch.relu(self.conv_layers[i](x))
            
            if self.sampling in ['maxpool', 'avgpool']:
                x = self.pools[i](x)
                
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc_out(x)
        return torch.sigmoid(x)

# ==========================================
