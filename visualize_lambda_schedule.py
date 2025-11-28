#!/usr/bin/env python3
"""
可視化 surrogate loss 的漸進引入過程
"""

import torch
import matplotlib.pyplot as plt

def compute_lambda_phy(epoch, max_lambda=0.01, start_epoch=20, end_epoch=60):
    """計算指定epoch的lambda_phy值"""
    if epoch < start_epoch:
        return 0.0
    elif epoch >= end_epoch:
        return max_lambda
    else:
        # Sigmoid-based gradual increase
        progress = (epoch - start_epoch) / (end_epoch - start_epoch)
        sigmoid_value = 1 / (1 + torch.exp(torch.tensor(-10 * (progress - 0.5))))  # Sigmoid with steepness 10
        return max_lambda * sigmoid_value.item()

def plot_lambda_schedule():
    """繪製lambda_phy隨epoch變化的曲線"""
    epochs = list(range(100))
    lambda_values = [compute_lambda_phy(epoch) for epoch in epochs]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, lambda_values, 'b-', linewidth=2, label='Gradual Surrogate Loss')

    # 添加關鍵點標記
    plt.axvline(x=20, color='r', linestyle='--', alpha=0.7, label='Start Introduction (Ep 20)')
    plt.axvline(x=60, color='g', linestyle='--', alpha=0.7, label='Full Strength (Ep 60)')
    plt.axhline(y=0.01, color='orange', linestyle='--', alpha=0.7, label='Max Lambda (0.01)')

    # 舊的方法對比
    old_lambda = [0.01 if epoch >= 30 else 0.0 for epoch in epochs]
    plt.plot(epochs, old_lambda, 'r--', linewidth=1, alpha=0.7, label='Old Step Function')

    plt.xlabel('Epoch')
    plt.ylabel('Lambda Physics (λ)')
    plt.title('Gradual Introduction of Surrogate Loss in Phase 2&3')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 保存圖片
    plt.savefig('surrogate_loss_schedule.png', dpi=150, bbox_inches='tight')
    plt.show()

def print_schedule_table():
    """打印關鍵epoch的lambda值"""
    print("Surrogate Loss Gradual Introduction Schedule:")
    print("=" * 50)
    print("Epoch  | Lambda | Description")
    print("-" * 35)

    key_epochs = [0, 19, 20, 30, 40, 50, 60, 70, 99]
    for epoch in key_epochs:
        lambda_val = compute_lambda_phy(epoch)
        if epoch == 0:
            desc = "Pure supervised learning"
        elif epoch == 19:
            desc = "Last epoch of pure learning"
        elif epoch == 20:
            desc = "Start gradual introduction"
        elif epoch == 60:
            desc = "Reach maximum lambda"
        elif epoch == 99:
            desc = "Training end"
        else:
            progress = (epoch - 20) / (60 - 20) * 100
            desc = f"~{progress:.0f}% of max lambda"

        print("5d")

if __name__ == "__main__":
    print_schedule_table()
    print("\nGenerating visualization...")
    try:
        plot_lambda_schedule()
        print("✅ Visualization saved as 'surrogate_loss_schedule.png'")
    except ImportError:
        print("⚠️  matplotlib not available, skipping visualization")