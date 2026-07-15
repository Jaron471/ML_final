import torch
from torch.utils.data import Dataset, Subset
import numpy as np

from .utils import MinMaxScaler

class TuringDataset(Dataset):
    def __init__(self, npz_path, num_samples=None, val_split=0.25, seed=42):
        """
        Args:
            npz_path: 資料檔路徑
            num_samples: 使用的樣本總數 (train+val)。None 表示使用全部
            val_split: 驗證集比例 (預設 0.25 即 3:1)
            seed: 隨機種子，確保取樣可重現
        """
        print(f"Loading dataset from {npz_path}...")
        data = np.load(npz_path)
        
        # 讀取 U 和 V
        # 形狀調整為 (N, Channel=1, H, W)
        u_full = data['u'].astype(np.float32)[:, np.newaxis, :, :]
        v_full = data['v'].astype(np.float32)[:, np.newaxis, :, :]
        
        # 讀取參數並堆疊
        params_full = np.stack([data['a'], data['b'], data['c'], data['delta']], axis=1).astype(np.float32)
        
        total_available = len(u_full)
        
        # 資料取樣（如果指定 num_samples）
        if num_samples is not None and num_samples < total_available:
            print(f"Sampling {num_samples} from {total_available} samples (seed={seed})...")
            rng = np.random.default_rng(seed)
            sample_indices = rng.choice(total_available, size=num_samples, replace=False)
            self.u = u_full[sample_indices]
            self.v = v_full[sample_indices]
            params = params_full[sample_indices]
        else:
            self.u = u_full
            self.v = v_full
            params = params_full
            num_samples = total_available
        
        # 初始化並擬合 Scaler
        self.scaler = MinMaxScaler(
            data_min=np.min(params, axis=0), 
            data_max=np.max(params, axis=0)
        )
        
        # 預先做正規化
        self.params_norm = self.scaler.transform(params)
        
        # 計算 train/val 分割
        n_val = int(num_samples * val_split)
        n_train = num_samples - n_val
        
        # 固定 seed 分割 train/val indices
        rng_split = np.random.default_rng(seed)
        indices = rng_split.permutation(num_samples)
        self.train_indices = indices[:n_train]
        self.val_indices = indices[n_train:]
        
        print(f"Data loaded: Total={num_samples}, Train={n_train}, Val={n_val}")

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        # 回傳: U(圖像), V(圖像, 用於Loss), Params(正規化後的 Label)
        return self.u[idx], self.v[idx], self.params_norm[idx]
    
    def get_train_val_datasets(self):
        """返回 train 和 val 的 Subset"""
        train_dataset = Subset(self, self.train_indices)
        val_dataset = Subset(self, self.val_indices)
        return train_dataset, val_dataset