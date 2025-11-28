import numpy as np
import torch
import os
import re
import glob

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

def calculate_multidim_nrmse(targets, preds):
    """
    計算多維 NRMSE (Normalized Root Mean Square Error)
    遵循論文定義：
    1. 計算所有樣本與所有參數的誤差平方和。
    2. 除以總預測點數 (m * d) 後開根號，得到 RMSE。
    3. 除以真實參數平均向量的歐幾里得範數 (||y_mean||) 進行標準化。
    
    Args:
        targets (np.ndarray): 真實值 (m, d)
        preds (np.ndarray): 預測值 (m, d)
        
    Returns:
        float: NRMSE
    """
    num_samples = targets.shape[0]        # m (樣本數量)
    num_parameters = targets.shape[1]     # d (參數數量)
    num_items = num_samples * num_parameters  # m * d
    
    total_squared_error = np.sum((targets - preds)**2)
    rmse_multi_dim = np.sqrt(total_squared_error / num_items)
    
    # 歸一化因子：真實參數平均向量的歐幾里得範數
    y_mean_vector = np.mean(targets, axis=0)
    norm_y_mean = np.linalg.norm(y_mean_vector)
    
    nrmse = rmse_multi_dim / norm_y_mean if norm_y_mean != 0 else 0.0
    return nrmse

def get_next_version(directory, pattern_regex):
    """
    掃描目錄中符合 regex 的檔案，找出最大的版本號 (num) 並回傳 num + 1。
    Regex 必須包含一個名為 'num' 的 group，例如: r"Surrogate_.*_(?P<num>\d+)_.*\.pth"
    """
    if not os.path.exists(directory):
        return 1
        
    max_num = 0
    regex = re.compile(pattern_regex)
    
    for filename in os.listdir(directory):
        match = regex.match(filename)
        if match:
            try:
                num = int(match.group('num'))
                if num > max_num:
                    max_num = num
            except ValueError:
                continue
                
    return max_num + 1

def find_model_path(directory, pattern_regex, version=None):
    """
    根據 regex 尋找模型檔案。
    如果 version (num) 為 None，則回傳版本號最大的檔案。
    如果 version 指定，則回傳該版本的檔案。
    """
    if not os.path.exists(directory):
        return None
        
    regex = re.compile(pattern_regex)
    candidates = []
    
    for filename in os.listdir(directory):
        match = regex.match(filename)
        if match:
            try:
                num = int(match.group('num'))
                candidates.append((num, os.path.join(directory, filename)))
            except ValueError:
                continue
    
    if not candidates:
        return None
        
    # Sort by version number
    candidates.sort(key=lambda x: x[0])
    
    if version is None:
        # Return the latest (largest num)
        return candidates[-1][1]
    else:
        # Find specific version
        for num, path in candidates:
            if num == version:
                return path
        return None