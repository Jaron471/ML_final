import torch
import torch.nn as nn
import torch.nn.functional as F
from . import config_new

class PhysicsLoss(nn.Module):
    def __init__(self):
        super(PhysicsLoss, self).__init__()
        self.dx = config_new.DX
        self.s = config_new.S_DIFFUSION
        
        # 定義拉普拉斯算子卷積核 (Finite Difference Laplacian)
        # [[0, 1, 0], [1, -4, 1], [0, 1, 0]]
        laplace_k = torch.tensor([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        # 使用 register_buffer 確保這個 tensor 會跟隨模型移動到 GPU，但不會被視為可訓練參數
        self.register_buffer('laplace_kernel', laplace_k)

    def laplacian(self, tensor):
        """
        計算 2D Laplacian。
        關鍵：使用 circular padding 模擬週期性邊界 (Periodic Boundary Condition)，
        這與 FFT 生成資料的邏輯一致。
        """
        # Padding: (left, right, top, bottom)
        padded = F.pad(tensor, (1, 1, 1, 1), mode='circular')
        return F.conv2d(padded, self.laplace_kernel) / (self.dx**2)

    def forward(self, u, v, params_real):
        """
        Args:
            u, v: 真實的物理場圖像 (Batch, 1, N, N)
            params_real: 已經反正規化回真實物理數值的參數 (Batch, 4) -> [a, b, c, delta]
        """
        # 拆解參數
        a = params_real[:, 0].view(-1, 1, 1, 1)
        b = params_real[:, 1].view(-1, 1, 1, 1)
        c = params_real[:, 2].view(-1, 1, 1, 1)
        delta = params_real[:, 3].view(-1, 1, 1, 1)

        # 1. 計算擴散項 (Diffusion: D * Laplacian)
        lap_u = self.laplacian(u)
        lap_v = self.laplacian(v)

        # 2. 計算反應項 (Reaction)
        # Gierer-Meinhardt 方程
        # 注意：加上 epsilon 避免除以 0
        v_safe = torch.clamp(v, min=1e-6)
        denom = v_safe * (1 + c * u**2)
        
        # Ru = a - b*u + u^2 / (v * (1 + c*u^2))
        Ru = a - (b * u) + (u**2 / denom)
        
        # Rv = u^2 - v
        Rv = (u**2) - v

        # 3. 計算時間導數 (Residual)
        # du/dt = s * lap_u + Ru
        du_dt = self.s * lap_u + Ru
        
        # dv/dt = s * delta * lap_v + Rv
        dv_dt = self.s * delta * lap_v + Rv

        # 4. 回傳 Masked Residual Loss (只在斑紋區域計算)
        # --- 策略二：Masked Physics Loss ---
        # 建立遮罩：只關注 u 大於該張圖平均值的地方 (即斑點/條紋本體)
        # 背景區域通常充滿數值噪聲，會干擾梯度學習
        u_mean = u.mean(dim=(2, 3), keepdim=True)
        mask = (u > u_mean).float()  # 轉為 0.0 或 1.0 的浮點數遮罩

        # 套用遮罩到 Residual 上
        masked_du = du_dt * mask
        masked_dv = dv_dt * mask

        # 計算 Loss (只除以有效像素數量，避免被大量背景稀釋)
        # 加上 1e-8 避免除以 0
        effective_pixels = torch.sum(mask) + 1e-8
        
        loss_u = torch.sum(masked_du**2) / effective_pixels
        loss_v = torch.sum(masked_dv**2) / effective_pixels

        return loss_u + loss_v