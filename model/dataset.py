import torch
from torch.utils.data import Dataset
import numpy as np

from .utils import MinMaxScaler

class TuringDataset(Dataset):
    def __init__(
        self,
        npz_path,
        scaler=None,
        parameter_names=None,
        indices=None,
    ):
        print(f"Loading dataset from {npz_path}...")
        with np.load(npz_path) as data:
            # 讀取 U 和 V，形狀調整為 (N, Channel=1, H, W)。
            self.u = data["u"].astype(np.float32)[:, np.newaxis, :, :]
            self.v = data["v"].astype(np.float32)[:, np.newaxis, :, :]

            if parameter_names is None:
                if "parameter_names" in data.files:
                    parameter_names = tuple(
                        str(name) for name in data["parameter_names"].tolist()
                    )
                elif all(name in data.files for name in ("a", "b", "c", "delta")):
                    parameter_names = ("a", "b", "c", "delta")
                elif all(name in data.files for name in ("A", "B", "d")):
                    parameter_names = ("A", "B", "d")
                elif all(name in data.files for name in ("a", "b", "d")):
                    parameter_names = ("a", "b", "d")
                else:
                    raise ValueError(
                        "Cannot infer parameter names from the NPZ keys; "
                        "pass parameter_names explicitly."
                    )
            self.parameter_names = tuple(parameter_names)
            missing = [
                name for name in self.parameter_names if name not in data.files
            ]
            if missing:
                raise KeyError(f"Missing parameter arrays in NPZ: {missing}")
            params = np.stack(
                [data[name] for name in self.parameter_names], axis=1
            ).astype(np.float32)

        if indices is not None:
            indices = np.asarray(indices, dtype=np.int64)
            self.u = self.u[indices]
            self.v = self.v[indices]
            params = params[indices]

        self.num_parameters = len(self.parameter_names)
        
        # Scaler 只能由 training set fit。Validation/test 必須傳入同一個
        # scaler，否則模型輸出與標籤會落在不同的 normalized coordinate。
        self.scaler = scaler if scaler is not None else MinMaxScaler(
            data_min=np.min(params, axis=0),
            data_max=np.max(params, axis=0),
        )
        if len(self.scaler.min_val) != self.num_parameters:
            raise ValueError(
                "Scaler dimension does not match dataset parameters: "
                f"{len(self.scaler.min_val)} != {self.num_parameters}"
            )
        
        # 預先做正規化
        self.params_norm = self.scaler.transform(params)
        
        print(
            f"Data loaded. Shape: {self.u.shape}; "
            f"parameters={self.parameter_names}"
        )

    def __len__(self):
        return len(self.u)

    def __getitem__(self, idx):
        # 回傳: U(圖像), V(圖像, 用於Loss), Params(正規化後的 Label)
        return self.u[idx], self.v[idx], self.params_norm[idx]
