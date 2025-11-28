#!/usr/bin/env python3
"""
GradNorm 測試腳本
展示如何在PINN訓練中使用GradNorm動態調整loss權重
"""

import torch
import torch.nn as nn
import torch.optim as optim
from model.loss import GradNorm

def test_gradnorm():
    """簡單的GradNorm測試"""

    # 創建一個簡單的模型
    model = nn.Linear(10, 1)

    # 創建兩個任務的loss
    criterion1 = nn.MSELoss()
    criterion2 = nn.L1Loss()

    # 初始化GradNorm
    gradnorm = GradNorm(num_tasks=2, alpha=1.5, device='cpu')
    initial_weights = torch.tensor([1.0, 0.1])  # 初始權重

    # 模擬訓練數據
    x = torch.randn(32, 10)
    y1 = torch.randn(32, 1)  # 任務1目標
    y2 = torch.randn(32, 1)  # 任務2目標

    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    print("🔧 GradNorm 測試開始")
    print(f"初始權重: {initial_weights}")

    for step in range(5):
        optimizer.zero_grad()

        # 前向傳播
        pred = model(x)

        # 計算兩個任務的loss
        loss1 = criterion1(pred, y1)
        loss2 = criterion2(pred, y2)

        losses = [loss1, loss2]

        # 使用GradNorm調整權重
        current_weights = gradnorm.adjust_weights(model, losses, initial_weights)

        # 計算加權總loss
        total_loss = current_weights[0] * loss1 + current_weights[1] * loss2

        print(".4f"
              ".4f")

        # 反向傳播和優化
        total_loss.backward()
        optimizer.step()

    print("✅ GradNorm 測試完成")

if __name__ == "__main__":
    test_gradnorm()