import torch
from torch.utils.data import Dataset
import numpy as np

from .utils import MinMaxScaler

class TuringDataset(Dataset):
    def __init__(self, npz_path):
        print(f"Loading dataset from {npz_path}...")
        data = np.load(npz_path)
        
        # 讀取 U 和 V
        # 形狀調整為 (N, Channel=1, H, W)
        self.u = data['u'].astype(np.float32)[:, np.newaxis, :, :]
        self.v = data['v'].astype(np.float32)[:, np.newaxis, :, :]
        
        # 讀取參數並堆疊
        params = np.stack([data['a'], data['b'], data['c'], data['delta']], axis=1).astype(np.float32)
        
        # 初始化並擬合 Scaler
        self.scaler = MinMaxScaler(
            data_min=np.min(params, axis=0), 
            data_max=np.max(params, axis=0)
        )
        
        # 預先做正規化
        self.params_norm = self.scaler.transform(params)
        
        print(f"Data loaded. Shape: {self.u.shape}")

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        # 回傳: U(圖像), V(圖像, 用於Loss), Params(正規化後的 Label)
        return self.u[idx], self.v[idx], self.params_norm[idx]