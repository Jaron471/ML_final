# Physical CNN：Reaction–Diffusion PDE 反問題

本專案使用 CNN 從穩態 Turing pattern 預測 reaction–diffusion PDE
參數，比較三種方法：

1. **Pure CNN**：只有參數 supervised loss。
2. **Physical CNN**：supervised loss 加上原始 PDE residual。
3. **Soft-Constrained Physical CNN**：保留 Physical CNN，訓練時再加入
   PDE 專屬的 integrated／elimination soft constraints。

所有正式模型在 inference 都只執行：

```text
u pattern → CNN → predicted parameters
```

沒有解析 inverse solver、least-squares projection、參數覆寫或
inference-time postprocessing。

## 支援的 PDE

| PDE | 預測參數 | 正式模型 |
| --- | --- | --- |
| Gierer–Meinhardt | `a, b, c, delta` | `cnn1` |
| Schnakenberg | `a, b, d` | `cnn2stride` |
| Brusselator | `A, B, d` | `cnn1` |
| Lengyel–Epstein | `a, phi, r` | `cnn2stride` |

Soft constraints 的共用建構原則是：

1. 使用穩態條件令時間微分為零。
2. 利用週期邊界下空間平均 Laplacian 為零。
3. 對 PDE 積分、消去未觀測場或建立 moment equation。
4. 將推導出的關係當作 differentiable penalty，而不是直接解參數。

具體公式依 PDE 而不同，實作位於
[`model/soft_constraints.py`](model/soft_constraints.py)。

## 資料 protocol

所有正式結果統一使用：

```text
12,000 training / 4,000 validation / 4,000 final evaluation
model seed = 42
split seed = 42
checkpoint = lowest validation normalized RMSE
```

### Gierer–Meinhardt

```text
gm_data/gm_train_data.npz       12,000
gm_data/gm_validation_data.npz   4,000
gm_data/gm_eval_data.npz         4,000
```

可重新下載：

```bash
python download_dataset.py
```

### 其他三個 PDE

每個 `*_train_data.npz` 有 16,000 筆。Trainer 使用固定 permutation
切成 12,000 training 與 4,000 validation；獨立的
`*_eval_data.npz` 保留 4,000 筆作 final evaluation。

```text
schnakenberg_data/
brusselator_data/
lengyel_epstein_data/
```

每個資料目錄只保留：

- `*_train_data.npz`
- `*_eval_data.npz`
- 參數 CSV
- convergence report
- generation summary

完整 20,000 筆合併檔與 GPU shards 是上述 train/eval 的重複資料，已
移除；需要時可用 `data_generate/` 內的固定 seed generator 重建。

## 環境

專案不保存 `.venv`。重新建立：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

資料生成另需與 CUDA 相容的 CuPy wheel。

## 訓練

### Gierer–Meinhardt

```bash
python train_gm.py --loss-type pure
python train_gm.py --loss-type physical
python train_gm.py --loss-type soft_physical
```

輸出至 `standardized_split_experiments/gm/`。

### Schnakenberg

```bash
python train_schnakenberg.py \
  --loss-type pure --model-arch cnn2stride \
  --validation-count 4000 --validation-seed 42

python train_schnakenberg.py \
  --loss-type physical --model-arch cnn2stride \
  --validation-count 4000 --validation-seed 42 \
  --lambda-physics 0.01

python train_schnakenberg.py \
  --loss-type soft_physical --model-arch cnn2stride \
  --validation-count 4000 --validation-seed 42 \
  --lambda-physics 0.01 \
  --lambda-identity 0.1 --lambda-elimination 0 \
  --soft-head-only
```

Brusselator 使用 `train_brusselator.py`；Lengyel–Epstein 使用
`train_lengyel_epstein.py`。正式 Soft 超參數與完整指令記錄在各
checkpoint 的 `training_config` 和
[`standardized_split_experiments/REPORT.md`](standardized_split_experiments/REPORT.md)。

## 評估

三個三參數 PDE：

```bash
python compare_standardized_pde_methods.py \
  --pde schnakenberg \
  --experiment-dir standardized_split_experiments/schnakenberg
```

將 `--pde` 與目錄替換成 `brusselator` 或 `lengyel_epstein` 即可。

GM：

```bash
python compare_gm_methods.py
```

Evaluator 會輸出：

- normalized／real parameter metrics
- per-sample PDE residual
- steady identity error
- Pure、Physical、Soft 的 pairwise 5,000 次 bootstrap
- JSON、CSV 與逐樣本 predictions

## 最新結果

### 四個 PDE

| PDE | Pure | Physical | Soft-Physical | 最佳 overall |
| --- | ---: | ---: | ---: | --- |
| Gierer–Meinhardt | 0.159949 | 0.155344 | **0.152762** | Soft-Physical |
| Schnakenberg | **0.046468** | 0.046995 | 0.046955 | Pure |
| Brusselator | 0.133998 | 0.137034 | **0.133553** | Soft-Physical |
| Lengyel–Epstein | 0.070772 | **0.070030** | 0.070698 | Physical |

Soft-Physical 相對 Pure：

- 四個 PDE 的 PDE residual 都下降。
- GM、Brusselator 與 Lengyel–Epstein 的 overall 顯著改善。
- Schnakenberg 改善 `d` 與物理一致性，但 overall 低於 Pure。

完整報告：

- [`standardized_split_experiments/REPORT.md`](standardized_split_experiments/REPORT.md)

GM 專屬詳細報告：

- [`standardized_split_experiments/gm/REPORT.md`](standardized_split_experiments/gm/REPORT.md)

## 精簡後的專案結構

```text
ML_final/
├── model/                         # CNN、dataset、physical/soft losses
├── data_generate/                 # 四個 PDE 的資料生成程式
├── gm_data/                       # GM train/validation/evaluation
├── *_data/                        # 其他 PDE 的 train/eval 與生成報告
├── train_gm.py
├── train_brusselator.py           # 三參數 PDE 共用 trainer
├── train_schnakenberg.py
├── train_lengyel_epstein.py
├── compare_gm_methods.py
├── compare_standardized_pde_methods.py
└── standardized_split_experiments/
    ├── gm/
    ├── schnakenberg/
    ├── brusselator/
    └── lengyel_epstein/
```

舊的 development trials、hard-projection 實驗、16k 無 validation
checkpoints、重複資料與暫存檔均不屬於目前的 Physical CNN 主線。

## Evaluation 限制

三個 PDE 的 4,000 筆 evaluation 曾在早期實驗中被查看。最新
checkpoints 僅使用新切出的 validation 選模，沒有使用 evaluation
挑選 epoch；若後續再依現有 evaluation 調整超參數，正式論文應重新生成
一批完全未見的 final test。
