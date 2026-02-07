import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class GradNorm:
    """
    GradNorm: Gradient Normalization for Multi-Task Learning
    動態調整多任務loss的權重，確保各個任務的梯度範數保持平衡
    """
    def __init__(self, num_tasks, alpha=1.5, device='cpu'):
        self.num_tasks = num_tasks
        self.alpha = alpha
        self.device = device
        
        # 初始化權重 (第一個任務權重固定為1，其餘為可學習參數)
        self.weights = nn.Parameter(torch.ones(num_tasks - 1, device=device))
        
    def get_weights(self):
        """獲取當前權重 (第一個任務權重為1)"""
        return torch.cat([torch.ones(1, device=self.device), self.weights])
    
    def compute_grad_norm(self, model, loss):
        """計算特定loss下的梯度範數"""
        # 清除之前的梯度
        model.zero_grad()
        
        # 反向傳播
        loss.backward(retain_graph=True)
        
        # 計算梯度範數
        total_norm = 0
        param_count = 0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        grad_norm = total_norm ** (1. / 2) if param_count > 0 else 0.0
        return grad_norm
    
    def adjust_weights(self, model, losses, initial_weights):
        """
        使用GradNorm調整權重 - 簡化版本
        
        Args:
            model: 神經網路模型
            losses: 各個任務的loss (list)
            initial_weights: 初始權重
            
        Returns:
            adjusted_weights: 調整後的權重
        """
        if len(losses) != self.num_tasks:
            raise ValueError(f"Expected {self.num_tasks} losses, got {len(losses)}")
        
        # 計算各任務的梯度範數
        grad_norms = []
        for loss in losses:
            grad_norm = self.compute_grad_norm(model, loss)
            grad_norms.append(grad_norm)
        
        grad_norms = torch.tensor(grad_norms, device=self.device)
        
        # GradNorm算法 - 簡化版本
        # 根據梯度範數的比例調整權重
        if grad_norms[0] > 0:
            # 計算權重調整因子，使所有任務的加權梯度範數相似
            weight_factors = grad_norms / grad_norms[0]
            adjusted_weights = initial_weights / weight_factors
            
            # 限制權重範圍
            adjusted_weights = torch.clamp(adjusted_weights, min=0.1, max=5.0)
            
            # 更新可學習權重 (簡單的指數移動平均)
            with torch.no_grad():
                target_weights = adjusted_weights[1:]  # 除了第一個任務
                if hasattr(self, 'ema_weights'):
                    self.ema_weights = 0.9 * self.ema_weights + 0.1 * target_weights
                else:
                    self.ema_weights = target_weights
                
                self.weights.data = self.ema_weights
            
        return self.get_weights()

class PhysicsLoss(nn.Module):
    def __init__(self, dx=1.0, s_diffusion=0.4):
        super(PhysicsLoss, self).__init__()
        self.dx = dx
        self.s = s_diffusion
        
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
        #mask = (u > u_mean).float()  # 轉為 0.0 或 1.0 的浮點數遮罩
        mask = (u == u).float()

        # 套用遮罩到 Residual 上
        #masked_du = du_dt * mask
        #masked_dv = dv_dt * mask
        masked_du = du_dt
        masked_dv = dv_dt

        # 計算 Loss (只除以有效像素數量，避免被大量背景稀釋)
        # 加上 1e-8 避免除以 0
        effective_pixels = torch.sum(mask) + 1e-8
        
        loss_u = torch.sum(masked_du**2) / effective_pixels
        loss_v = torch.sum(masked_dv**2) / effective_pixels

        return loss_u + loss_v
