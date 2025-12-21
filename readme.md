# ML_final — Turing Pattern 參數反推（Unified Training Pipeline）

這個專案的目標是從 **Turing pattern**（反應擴散系統產生的斑紋）影像中，反推出對應的物理參數 **[a, b, c, delta]**。

核心入口為 [train_unified.py](train_unified.py)，該腳本統一支援：

- **MLP**（全連接網路）
- **論文式極簡 CNN**（1-layer / 2-layer / 2-layer+MaxPool / 2-layer+Stride）
- 兩種訓練策略：
	- **Pure（純監督）**：只用 supervised loss
	- **Physical（PINN-like）**：supervised loss + 物理殘差（Physics Loss）

同時提供 [verify_comparison.py](verify_comparison.py) 針對不同 data fraction / loss type / model arch 的評估報表。

---

## 1. 專案

### 訓練流程統整

[train_unified.py](train_unified.py) 將「資料載入、模型建立、optimizer/scheduler、loss 設計、存檔與 WandB 記錄」整合為一致的訓練流程 (pipeline)：

1. **資料載入**
	 - 使用 [model/dataset.py](model/dataset.py) 的 `TuringDataset` 讀取 `train_data.npz` / `val_data.npz`。
	 - 影像：`u`、`v` 皆會變成 `(N, 1, 128, 128)`。
	 - 標籤：`a,b,c,delta` 會堆成 `(N, 4)`，再透過 MinMaxScaler 正規化到 `[0,1]`。
	 - 可用 `--data-fraction` 做 **data ablation**（固定 seed=42 的 random_split）。

2. **模型（Model Architecture）**
	 - `mlp`：使用 [model/model.py](model/model.py) 的 `MLPNet`（Flatten → 1024 → 512 → 128 → 4，最後 sigmoid）。
	 - `cnn1/cnn2/cnn2pool/cnn2stride`：在 [train_unified.py](train_unified.py) 內定義的 PaperMinimalCNN 系列。
		 - 以 `nk`（channels）、`np`（kernel size）、`nf`（hidden neurons）控制容量。

3. **Optimizer / Scheduler**
	 - MLP：AdamW（預設 `lr=1e-4`，weight_decay=0.01）+ gradient clipping (`max_norm=1.0`)。
	 - CNN：Adam（預設 `lr=1e-3`）。
	 - Scheduler：**Warmup(10% epochs) + CosineAnnealing**（SequentialLR）。

4. **Loss 設計（最重要）**
	 - Supervised Loss：
		 - MLP：**weighted L1**（權重在 [model/config.py](model/config.py) 的 `LOSS_WEIGHTS`）
		 - CNN：MSE
	 - Physics Loss（physical / PINN-like）：
		 - 由 [model/loss.py](model/loss.py) 的 `PhysicsLoss` 計算 Gierer–Meinhardt 反應擴散方程殘差
		 - Laplacian 使用 circular padding 以符合週期邊界
		 - **Masked Physics Loss**：只在 `u > mean(u)` 的斑紋區域計算殘差，避免背景噪聲稀釋梯度
	 - Physics loss 權重：`loss = loss_sup + λ * loss_phy`
		 - `λ` 由 `--lambda-phy` 指定，且可用 `--phys-gradual` 做 gradual / warmup（CNN: epoch 20 step；MLP: epoch 20→60 sigmoid warmup）。

5. **Checkpoint 存檔**
	 - 預設輸出到資料夾 `paper_checkpoints/`
	 - 每次訓練都會存：
		 - `*_last.pth`（每個 epoch 覆蓋一次）
		 - `*_best.pth`（validation NRMSE 最佳才更新）
	 - 命名格式（由程式自動遞增版本號）：
		 - `PaperPINN_{ARCH}_{LOSS}_nk{nk}_frac{fraction}_{version}_{best|last}.pth`（CNN 類）
		 - `PaperPINN_{ARCH}_{LOSS}_frac{fraction}_{version}_{best|last}.pth`（MLP）

6. **評估指標**
	 - training / validation 主要看 `NRMSE`（normalized space, multi-dim joint）
	 - [verify_comparison.py](verify_comparison.py) 會輸出更多表格：NRMSE / R²

---

## 2. 專案結構

常用檔案：

- [train_unified.py](train_unified.py)：主要訓練入口（MLP/CNN + pure/physical）
- [verify_comparison.py](verify_comparison.py)：批次載入 checkpoints，輸出比較報表
- [download_dataset.py](download_dataset.py)：從 Google Drive 下載資料集到專案根目錄
- [download_checkpoints.py](download_checkpoints.py)：下載預訓練 checkpoints 到 `paper_checkpoints/`
- [model/config.py](model/config.py)：路徑、超參數、loss weights、WandB project
- [model/dataset.py](model/dataset.py)：NPZ → Dataset（含 MinMax 正規化）
- [model/loss.py](model/loss.py)：PhysicsLoss（含 Mask）+ GradNorm
- [model/utils.py](model/utils.py)：MinMaxScaler、NRMSE、checkpoint 版本號工具

資料檔（預期存在於根目錄）：

- `train_data.npz` / `val_data.npz` / `test_data.npz`
- `turing_patterns_dataset_merged.npz`（合併大資料集；主要供檢查或重新分割資料使用）

---

## 3. Dataset 格式（NPZ）

`TuringDataset` 預期輸入的 `.npz` 檔案至少包含以下 keys：

- `u`: shape `(N, 128, 128)`
- `v`: shape `(N, 128, 128)`
- `a`, `b`, `c`, `delta`: shape `(N,)`

訓練過程中，`u`、`v` 將被轉換為 `(N, 1, 128, 128)`，且 `[a,b,c,delta]` 會被正規化至 `[0,1]` 區間。

---

## 4. 環境安裝（Windows / Linux / macOS）

建議 Python 3.10+。

### 4.1 建立虛擬環境（Windows PowerShell）

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 4.2 安裝依賴

本專案至少需要：`torch, numpy, tqdm, wandb, gdown, pandas`。

若使用 NVIDIA GPU，建議依照 PyTorch 官網指示安裝對應 CUDA 版本的 torch。

安裝範例（CPU 版，簡單可跑）：

```bash
pip install -r requirements.txt
```

---

## 5. 資料準備

### 5.1 下載資料集（會把檔案放到根目錄）

```bash
python download_dataset.py
```

注意：此腳本會將 Google Drive 資料夾下載至暫存區後，再 **搬移至專案根目錄**。若根目錄已存在同名檔案，將會被覆蓋。

### 5.2（選配）下載 paper checkpoints

```bash
python download_checkpoints.py
```

---

## 6. 訓練：train_unified.py

### 6.1 最常用指令（快速開始）

1) CNN1 + Pure（25% data）

```bash
python train_unified.py --model-arch cnn1 --use-loss pure --data-fraction 0.25 --nk 5 --np 5 --nf 5
```

2) CNN1 + Physical（25% data）

```bash
python train_unified.py --model-arch cnn1 --use-loss physical --data-fraction 0.25 --nk 5 --np 5 --nf 5 --phys-gradual
```

3) MLP + Pure（10% data）

```bash
python train_unified.py --model-arch mlp --use-loss pure --data-fraction 0.1
```

4) MLP + Physical（10% data）

```bash
python train_unified.py --model-arch mlp --use-loss physical --data-fraction 0.1 --phys-gradual
```

### 6.2 參數說明

- `--model-arch`：`mlp | cnn1 | cnn2 | cnn2pool | cnn2stride`
- `--use-loss`：
	- `pure`：只跑 supervised
	- `physical`：supervised + physics residual
- `--data-fraction`：資料抽樣比例（例如 0.1 = 10%）
- `--nk --np --nf`：CNN 的容量超參數
- `--lambda-phy`：physics loss 權重（預設 0.01）
- `--phys-gradual`：逐步引入 physics loss
- `--lr`：指定學習率（不填則：CNN=1e-3，MLP=1e-4）

### 6.3 輸出在哪裡

訓練完成後會在 `paper_checkpoints/` 看到：

- `..._best.pth`：validation NRMSE 最佳
- `..._last.pth`：最後一次 epoch

---

## 7. WandB（實驗紀錄）

訓練預設會呼叫 `wandb.init(...)`。

使用者可選擇：

1) 登入使用

```bash
wandb login
```

2) 離線模式（不使用 WandB 上傳）

Windows PowerShell：

```bash
$env:WANDB_MODE="disabled"
python train_unified.py --model-arch cnn2 --use-loss pure --data-fraction 0.25 --nk 5 --np 5 --nf 5
```

（或設定環境變數 `WANDB_API_KEY` 讓它自動登入。）

---

## 8. 評估：verify_comparison.py

這支腳本會：

- 讀 `test_data.npz`
- 載入 `paper_checkpoints/` 內指定的模型檔
- 產出不同 data fraction 的 Pure vs Physical 對照表（NRMSE / R² / ...）

執行：

```bash
python verify_comparison.py
```

備註：目前 `verify_comparison.py` 內的 `model_files` 需手動指定檔名；若訓練了新模型，需將其檔名加入列表方可進行評估。

---

## 9. 常見問題（Troubleshooting）

### 9.1 找不到資料檔

確認根目錄存在：`train_data.npz` / `val_data.npz` / `test_data.npz`。
如果檔名不同，請改 [model/config.py](model/config.py) 的 `TRAIN_PATH / VAL_PATH / TEST_PATH`。

### 9.2 ImportError: 找不到 'model'

請確認是在「專案根目錄」下執行：

```bash
python train_unified.py ...
```

### 9.3 沒 GPU 可以跑嗎？

可以，程式會自動選擇 `cpu`。但訓練速度會明顯變慢。

### 9.4 `--phys-gradual` 看起來無法關掉？

目前 `train_unified.py` 裡 `--phys-gradual` 的 argparse 設定是 `action='store_true'` 且 `default=True`，因此預設為 True。
若需要「完全固定 λ」的版本，可將 argparse 的 default 修改為 False（或改為 `--no-phys-gradual` 形式）。

---

## 10. 快速檢查 NPZ（可選）

可使用 [inspect_turing_npz.py](inspect_turing_npz.py) 快速檢視 `.npz` 的 keys 與 shape。
注意：該檔案內的 `NPZ_PATH` 可能需自行修改為當前存在的 npz 檔名。

---

## 11. 資料生成 (Data Generation)

若需自行重新生成資料集，請參考 `data_generate/` 資料夾下的腳本。

### 11.1 生成流程

1. **參數生成 (`param-gen.py`)**
   - 根據論文範圍隨機採樣 `a, b, c, delta`。
   - 使用 `fsolve` 計算穩態，並檢查 Turing Instability 條件（Jacobian Trace/Det）。
   - 篩選出符合條件的參數組合。

2. **模擬與生成 (`gm_cupy.py`)**
   - 使用 **CuPy** 進行 GPU 加速求解 Gierer-Meinhardt 方程。
   - 採用 FFT 計算 Laplacian，大幅提升速度。
   - 輸出 `u` 與 `v` 的最終穩態斑紋。

3. **資料合併 (`merge.py`)** (可選)
   - 若分批生成多個 `.npz` 檔，可使用此腳本將其合併為一個大檔案。

4. **資料分割 (`split.py`)**
   - 將生成的 `.npz` 分割為 `train`, `val`, `test`。
   - 預設比例可於腳本內調整（目前設定為固定數量）。

### 11.2 執行方式

```bash
# 1. 生成參數 (範例)
python data_generate/param-gen.py

# 2. 執行模擬 (需安裝 cupy)
python data_generate/gm_cupy.py

# 3. 分割資料
python data_generate/split.py
```
