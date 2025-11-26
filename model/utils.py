import numpy as np
import torch

class MinMaxScaler:
    """
    將參數縮放到 [0, 1] 以利神經網絡訓練，
    同時支援 Tensor 操作以便在 Loss 計算時反推回真實物理數值。
    """
    def __init__(self, data_min, data_max):
        self.min_val = np.array(data_min, dtype=np.float32)
        self.max_val = np.array(data_max, dtype=np.float32)
        self.range_val = self.max_val - self.min_val + 1e-8
        
        # 預先轉為 Tensor 方便 GPU 計算
        self.min_t = None
        self.range_t = None

    def transform(self, data):
        """Numpy 轉換 (用於 Dataset)"""
        return (data - self.min_val) / self.range_val

    def inverse_transform_numpy(self, data):
        """Numpy 反轉換 (用於視覺化)"""
        return data * self.range_val + self.min_val

    def to_device(self, device):
        """將參數搬移到 GPU (用於 Loss 計算)"""
        self.min_t = torch.tensor(self.min_val, dtype=torch.float32, device=device)
        self.range_t = torch.tensor(self.range_val, dtype=torch.float32, device=device)

    def inverse_transform_tensor(self, data_tensor):
        """Tensor 反轉換 (用於 Physics Loss)"""
        if self.min_t is None:
            raise ValueError("Please call to_device() first.")
        return data_tensor * self.range_t + self.min_t