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
        mask = (u > u_mean).float()  # 轉為 0.0 或 1.0 的浮點數遮罩

        # 套用遮罩到 Residual 上
        masked_du = du_dt 
        masked_dv = dv_dt 

        # 計算 Loss (只除以有效像素數量，避免被大量背景稀釋)
        # 加上 1e-8 避免除以 0
        effective_pixels = torch.sum(mask) + 1e-8
        
        loss_u = torch.sum(masked_du**2) / effective_pixels
        loss_v = torch.sum(masked_dv**2) / effective_pixels

        return loss_u + loss_v
class FourierLoss(nn.Module):
    """比較頻譜幅度 (忽略相位/位置差異，專注於波長與方向)"""
    def __init__(self, radius=0.7):
        super(FourierLoss, self).__init__()
        self.radius = radius

    def forward(self, recon_img, true_img):
        B, C, H, W = recon_img.shape
        # 2D FFT
        fft_recon = torch.fft.fft2(recon_img)
        fft_true = torch.fft.fft2(true_img)
        
        # 取幅度譜 (Magnitude)
        mag_recon = torch.abs(fft_recon)
        mag_true = torch.abs(fft_true)
        
        # 建立低頻遮罩 (Mask)
        yy, xx = torch.meshgrid(
            torch.linspace(-1, 1, H, device=recon_img.device),
            torch.linspace(-1, 1, W, device=recon_img.device),
            indexing="ij"
        )
        rr = torch.sqrt(xx**2 + yy**2)
        mask = (rr < self.radius).float()
        
        # 計算遮罩後的 MSE
        diff = (mag_recon - mag_true) ** 2
        return torch.mean(diff * mask)

class HistogramLoss(nn.Module):
    """比較像素值分佈 (忽略位置，專注於黑白比例與對比度)"""
    def __init__(self):
        super(HistogramLoss, self).__init__()
        
    def forward(self, recon_img, true_img):
        # Flatten: (B, C, H, W) -> (B, H*W)
        b = recon_img.size(0)
        recon_flat = recon_img.view(b, -1)
        true_flat = true_img.view(b, -1)
        
        # Sorting (排序後比較 = 比較分佈)
        recon_sorted, _ = torch.sort(recon_flat, dim=1)
        true_sorted, _ = torch.sort(true_flat, dim=1)
        
        # L1 Loss on sorted pixels
        return torch.mean(torch.abs(recon_sorted - true_sorted))
    

import torch
import torch.nn as nn
import torch.fft

class SpectralPhysicsLoss(nn.Module):
    def __init__(self, dx=1.0, N=128, s_diffusion=0.4, device='cpu'):
        super(SpectralPhysicsLoss, self).__init__()
        self.dx = dx
        self.s = s_diffusion
        self.N = N
        self.device = device
        
        # --- 預計算頻率項 (與生成代碼完全一致) ---
        # kx = 2 * pi * fftfreq(N, d=dx)
        # 這裡用 PyTorch 實作相同邏輯
        # PyTorch 的 fftfreq 回傳值範圍是 [0, 0.5, -0.5, ...]，需要乘上 2*pi/dx ? 
        # 讓我們對齊 cupy 的邏輯: kx = 2 * cp.pi * cp.fft.fftfreq(N, d=dx)
        
        # torch.fft.fftfreq(n, d) returns f. The angular freq is 2*pi*f
        freq = torch.fft.fftfreq(N, d=dx, device=device)
        k = 2 * torch.pi * freq
        
        kx = k.view(1, 1, N, 1) # (Batch, Channel, H, W) 廣播準備
        ky = k.view(1, 1, 1, N)
        
        # Laplacian Eigenvalues: -(kx^2 + ky^2)
        # 注意: 這裡計算完後要存成 buffer，避免反向傳播更新它
        self.register_buffer('laplacian_operator', -(kx**2 + ky**2))

    def laplacian_spectral(self, tensor):
        """
        使用 FFT 計算 Laplacian
        Tensor shape: (B, 1, H, W)
        """
        # 1. 轉到頻域
        fft_coeff = torch.fft.fft2(tensor)
        
        # 2. 乘上 Laplacian 算子 (element-wise)
        # 這裡對應 d^2/dx^2 + d^2/dy^2
        fft_lap = fft_coeff * self.laplacian_operator
        
        # 3. 轉回時域 (取實部)
        return torch.fft.ifft2(fft_lap).real

    def forward(self, u, v, params_real):
        """
        與原本的 PhysicsLoss 介面一致，但內部微分改用 FFT
        """
        # 拆解參數
        a = params_real[:, 0].view(-1, 1, 1, 1)
        b = params_real[:, 1].view(-1, 1, 1, 1)
        c = params_real[:, 2].view(-1, 1, 1, 1)
        delta = params_real[:, 3].view(-1, 1, 1, 1)

        # 1. 計算擴散項 (使用 Spectral Laplacian)
        lap_u = self.laplacian_spectral(u)
        lap_v = self.laplacian_spectral(v)

        # 2. 計算反應項 (與原本相同)
        v_safe = torch.clamp(v, min=1e-6)
        denom = v_safe * (1 + c * u**2)
        
        Ru = a - (b * u) + (u**2 / denom)
        Rv = (u**2) - v

        # 3. 計算時間導數 (Residual)
        # du/dt = s * lap_u + Ru
        du_dt = self.s * lap_u + Ru
        
        # dv/dt = s * delta * lap_v + Rv
        dv_dt = self.s * delta * lap_v + Rv

        # 4. Masked Loss (保持原本的 Mask 策略)
        u_mean = u.mean(dim=(2, 3), keepdim=True)
        mask = (u > u_mean).float()

        masked_du = du_dt * mask
        masked_dv = dv_dt * mask

        effective_pixels = torch.sum(mask) + 1e-8
        
        loss_u = torch.sum(masked_du**2) / effective_pixels
        loss_v = torch.sum(masked_dv**2) / effective_pixels

        return loss_u + loss_v


class HighOrderPhysicsLoss(nn.Module):
    def __init__(self, dx=1.0, s_diffusion=0.4):
        super(HighOrderPhysicsLoss, self).__init__()
        self.dx = dx
        self.s = s_diffusion
        
        # --- 5點差分係數 (4th Order Accuracy) ---
        # 針對二階導數 f''(x)，係數為: [-1/12, 4/3, -5/2, 4/3, -1/12]
        weights_1d = torch.tensor([-1/12, 4/3, -5/2, 4/3, -1/12], dtype=torch.float32)
        
        # 🔥 修改點：建立 5x5 的卷積核 (中間填係數，其他填 0)
        # 這樣才能正確消耗掉 2D Padding，讓輸出維度回到 128x128
        
        # Kernel X: 計算水平微分
        # 在 5x5 矩陣中，只有中間那一行 (row 2) 有值
        k_x_2d = torch.zeros(1, 1, 5, 5)
        k_x_2d[0, 0, 2, :] = weights_1d
        self.k_x = k_x_2d / (self.dx**2)
        
        # Kernel Y: 計算垂直微分
        # 在 5x5 矩陣中，只有中間那一道 (col 2) 有值
        k_y_2d = torch.zeros(1, 1, 5, 5)
        k_y_2d[0, 0, :, 2] = weights_1d
        self.k_y = k_y_2d / (self.dx**2)
        
        # 使用 register_buffer
        self.register_buffer('kernel_x', self.k_x)
        self.register_buffer('kernel_y', self.k_y)

    def laplacian(self, tensor):
        """
        計算 5x5 Laplacian
        """
        # Padding: (Left, Right, Top, Bottom)
        # 輸入 128x128 -> Padding 後變成 132x132
        padded = F.pad(tensor, (2, 2, 2, 2), mode='circular')
        
        # Conv2d: 輸入 132x132，核 5x5
        # 輸出大小 = 132 - 5 + 1 = 128 (完美回到原尺寸)
        d2x = F.conv2d(padded, self.kernel_x)
        d2y = F.conv2d(padded, self.kernel_y)
        
        return d2x + d2y

    def forward(self, u, v, params_real):
        # 拆解參數
        a = params_real[:, 0].view(-1, 1, 1, 1)
        b = params_real[:, 1].view(-1, 1, 1, 1)
        c = params_real[:, 2].view(-1, 1, 1, 1)
        delta = params_real[:, 3].view(-1, 1, 1, 1)

        # 1. 計算擴散項 (使用修正後的 5x5 高階差分)
        lap_u = self.laplacian(u)
        lap_v = self.laplacian(v)

        # 2. 計算反應項 (保持不變)
        v_safe = torch.clamp(v, min=1e-6)
        denom = v_safe * (1 + c * u**2)
        
        Ru = a - (b * u) + (u**2 / denom)
        Rv = (u**2) - v

        # 3. 計算殘差
        du_dt = self.s * lap_u + Ru
        dv_dt = self.s * delta * lap_v + Rv

        # 4. Masked Loss
        u_mean = u.mean(dim=(2, 3), keepdim=True)
        mask = (u > u_mean).float()

        masked_du = du_dt * mask
        masked_dv = dv_dt * mask

        effective_pixels = torch.sum(mask) + 1e-8
        
        loss_u = torch.sum(masked_du**2) / effective_pixels
        loss_v = torch.sum(masked_dv**2) / effective_pixels

        return loss_u + loss_v
    

class TopKPhysicsLoss(nn.Module):
    def __init__(self, dx=1.0, s_diffusion=0.4, top_k_ratio=0.05):
        """
        Args:
            top_k_ratio (float): 只取前 K% 誤差最大的像素來算 Loss (預設 0.05 = 5%)
        """
        super(TopKPhysicsLoss, self).__init__()
        self.dx = dx
        self.s = s_diffusion
        self.top_k_ratio = top_k_ratio
        
        # --- 保持原本的 3x3 卷積核 (Finite Difference Laplacian) ---
        laplace_k = torch.tensor([[0, 1, 0],
                                  [1, -4, 1],
                                  [0, 1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        
        self.register_buffer('laplace_kernel', laplace_k)

    def laplacian(self, tensor):
        # 保持原本的 Padding 邏輯
        padded = F.pad(tensor, (1, 1, 1, 1), mode='circular')
        return F.conv2d(padded, self.laplace_kernel) / (self.dx**2)

    def forward(self, u, v, params_real):
        # 1. 拆解參數 (保持不變)
        a = params_real[:, 0].view(-1, 1, 1, 1)
        b = params_real[:, 1].view(-1, 1, 1, 1)
        c = params_real[:, 2].view(-1, 1, 1, 1)
        delta = params_real[:, 3].view(-1, 1, 1, 1)

        # 2. 計算物理項 (保持不變)
        lap_u = self.laplacian(u)
        lap_v = self.laplacian(v)

        v_safe = torch.clamp(v, min=1e-6)
        denom = v_safe * (1 + c * u**2)
        
        Ru = a - (b * u) + (u**2 / denom)
        Rv = (u**2) - v

        du_dt = self.s * lap_u + Ru
        dv_dt = self.s * delta * lap_v + Rv

        # ==========================================
        # 🔥 修改重點：Top-K Hard Pixel Mining
        # ==========================================
        
        # 1. 計算每個像素的物理誤差平方 (Squared Residual)
        # shape: (Batch, 1, H, W)
        res_u_sq = du_dt**2
        res_v_sq = dv_dt**2
        
        # 2. 依然使用 Mask 過濾掉純背景 (因為背景的誤差通常是數值噪聲)
        u_mean = u.mean(dim=(2, 3), keepdim=True)
        mask = (u > u_mean).float()
        
        # 將 Mask 外的誤差歸零
        res_u_sq = res_u_sq * mask
        res_v_sq = res_v_sq * mask
        
        # 3. 展平 (Batch, H*W) 以便排序
        batch_size = u.size(0)
        flat_u = res_u_sq.view(batch_size, -1)
        flat_v = res_v_sq.view(batch_size, -1)
        
        # 4. 計算 K 的大小 (例如總像素的 5%)
        # 假設 128x128 = 16384 pixels, 5% ≈ 819 個點
        num_pixels = flat_u.size(1)
        k = int(num_pixels * self.top_k_ratio)
        # 確保 k 至少為 1
        k = max(1, k)
        
        # 5. 取出前 K 大的誤差 (Top-K)
        # values: (Batch, K)
        topk_u, _ = torch.topk(flat_u, k, dim=1)
        topk_v, _ = torch.topk(flat_v, k, dim=1)
        
        # 6. 計算平均 Loss (只針對這 K 個最難搞的像素)
        loss_u = torch.mean(topk_u)
        loss_v = torch.mean(topk_v)

        return loss_u + loss_v

class HighOrderTopKPhysicsLoss(nn.Module):
    def __init__(self, dx=1.0, s_diffusion=0.4, top_k_ratio=0.05):
        super(HighOrderTopKPhysicsLoss, self).__init__()
        self.dx = dx
        self.s = s_diffusion
        self.top_k_ratio = top_k_ratio
        
        # --- 1. 使用 5x5 高階差分係數 (4th Order Accuracy) ---
        # 這是為了 "精準"
        weights_1d = torch.tensor([-1/12, 4/3, -5/2, 4/3, -1/12], dtype=torch.float32)
        
        # Kernel X: 填入 5x5 矩陣的中間行
        k_x_2d = torch.zeros(1, 1, 5, 5)
        k_x_2d[0, 0, 2, :] = weights_1d
        self.k_x = k_x_2d / (self.dx**2)
        
        # Kernel Y: 填入 5x5 矩陣的中間列
        k_y_2d = torch.zeros(1, 1, 5, 5)
        k_y_2d[0, 0, :, 2] = weights_1d
        self.k_y = k_y_2d / (self.dx**2)
        
        self.register_buffer('kernel_x', self.k_x)
        self.register_buffer('kernel_y', self.k_y)

    def laplacian(self, tensor):
        # 使用 Padding=2 (因為核大小是 5)
        padded = F.pad(tensor, (2, 2, 2, 2), mode='circular')
        d2x = F.conv2d(padded, self.kernel_x)
        d2y = F.conv2d(padded, self.kernel_y)
        return d2x + d2y

    def forward(self, u, v, params_real):
        # 1. 拆解參數
        a = params_real[:, 0].view(-1, 1, 1, 1)
        b = params_real[:, 1].view(-1, 1, 1, 1)
        c = params_real[:, 2].view(-1, 1, 1, 1)
        delta = params_real[:, 3].view(-1, 1, 1, 1)

        # 2. 計算物理項 (使用高階 5x5)
        lap_u = self.laplacian(u)
        lap_v = self.laplacian(v)

        v_safe = torch.clamp(v, min=1e-6)
        denom = v_safe * (1 + c * u**2)
        Ru = a - (b * u) + (u**2 / denom)
        Rv = (u**2) - v

        du_dt = self.s * lap_u + Ru
        dv_dt = self.s * delta * lap_v + Rv

        # ==========================================
        # 3. Top-K Hard Pixel Mining (這是為了 "專注")
        # ==========================================
        
        res_u_sq = du_dt**2
        res_v_sq = dv_dt**2
        
        # Mask 濾掉背景
        u_mean = u.mean(dim=(2, 3), keepdim=True)
        mask = (u > u_mean).float()
        
        flat_u = (res_u_sq * mask).view(u.size(0), -1)
        flat_v = (res_v_sq * mask).view(v.size(0), -1)
        
        # 只取前 5% 的誤差
        num_pixels = flat_u.size(1)
        k = int(num_pixels * self.top_k_ratio)
        k = max(1, k)
        
        topk_u, _ = torch.topk(flat_u, k, dim=1)
        topk_v, _ = torch.topk(flat_v, k, dim=1)
        
        loss_u = torch.mean(topk_u)
        loss_v = torch.mean(topk_v)

        return loss_u + loss_v