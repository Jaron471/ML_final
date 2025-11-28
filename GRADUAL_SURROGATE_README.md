# Surrogate Loss 漸進引入改進

## 概述

原本在 Phase 2&3 中，surrogate loss (物理循環一致性損失) 是通過一個簡單的 step function 引入的：

```python
lambda_phy = 0.0 if epoch < 30 else 0.01  # 前30個epoch為0，之後突然跳到0.01
```

這種突然的轉變可能會造成訓練不穩定。現在我們改為使用 sigmoid 函數實現平滑的漸進引入。

## 新實現

### 漸進調度參數
```python
max_lambda = 0.01      # 最大lambda值
start_epoch = 20       # 開始引入的epoch
end_epoch = 60         # 達到最大值的epoch
```

### Sigmoid 漸進函數
```python
if epoch < start_epoch:
    lambda_phy = 0.0
elif epoch >= end_epoch:
    lambda_phy = max_lambda
else:
    progress = (epoch - start_epoch) / (end_epoch - start_epoch)
    sigmoid_value = 1 / (1 + torch.exp(-10 * (progress - 0.5)))
    lambda_phy = max_lambda * sigmoid_value.item()
```

## 訓練階段

| 階段 | Epoch範圍 | Lambda值 | 描述 |
|------|-----------|----------|------|
| **純監督學習** | 1-19 | 0.0 | 專注學習基本映射 |
| **漸進引入** | 20-59 | 0.0 → 0.01 | Sigmoid平滑過渡 |
| **滿載運轉** | 60-100 | 0.01 | 完整物理約束 |

## 優勢

### 1. **更平滑的訓練動態**
- 避免突然引入損失造成的梯度衝擊
- 模型有時間適應物理約束

### 2. **更好的收斂特性**
- 減少訓練不穩定性
- 更有可能找到更好的局部最優

### 3. **可調節的過渡速度**
- 通過調整sigmoid陡度參數控制過渡速度
- 可以根據具體任務調整start_epoch和end_epoch

## 可視化

運行以下命令查看lambda調度曲線：
```bash
python visualize_lambda_schedule.py
```

這會生成 `surrogate_loss_schedule.png` 圖片，顯示：
- 新的sigmoid漸進曲線 (藍色實線)
- 舊的step function對比 (紅色虛線)
- 關鍵時間點標記

## 技術細節

### Sigmoid函數參數
- **陡度**: 10 (控制過渡的陡峭程度)
- **中心點**: 0.5 (sigmoid函數的中心位置)
- **範圍**: 自動映射到[start_epoch, end_epoch]區間

### 數學公式
```
progress = (epoch - start_epoch) / (end_epoch - start_epoch)
sigmoid_value = 1 / (1 + exp(-10 × (progress - 0.5)))
lambda_phy = max_lambda × sigmoid_value
```

## 使用方式

這個改進會自動應用到所有hybrid訓練中，不需要額外配置。訓練輸出會顯示當前epoch的lambda值：

```
[Ep 25] Sup=0.0456 | Val NRMSE [δ=12.34% | Multi=8.90%] (λ=0.0023)
[Ep 40] Sup=0.0345 | Val NRMSE [δ=9.87% | Multi=6.54%] (λ=0.0078)
[Ep 60] Sup=0.0234 | Val NRMSE [δ=7.12% | Multi=4.56%] (λ=0.0100)
```

## 兼容性

- **向後兼容**: 不影響現有配置
- **自動應用**: 所有hybrid訓練都會使用新的調度
- **無性能損失**: 計算開銷可以忽略不計

這個改進讓surrogate loss的引入更加平滑自然，有望提升模型的最終性能和訓練穩定性。