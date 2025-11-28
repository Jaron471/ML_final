# Turing Pattern Inversion Network (TPIN)

這是一個利用深度學習 (CNN/MLP) 與物理約束神經網絡 (PINN) 來逆向推導 Gierer-Meinhardt (GM) 反應擴散方程參數的專案。

-----

### 1. 數據生成流程 (Data Generation Pipeline)

這部分的程式碼負責生成符合 Turing Instability 條件的參數，並利用數值模擬產生對應的斑紋圖像。

*   **`param-gen.py` (參數篩選)**
    *   **功能**：基於線性穩定性分析 (Linear Stability Analysis)，在廣大的參數空間中隨機採樣，篩選出能產生圖靈斑紋的參數組合 $(a, b, c, \delta)$。
    *   **產出**：`qualified_turing_params_20000.csv` (包含合格參數的列表)。

*   **`gm_cupy.py` / `gm_mps.py` (模擬引擎)**
    *   **功能**：讀取 CSV 中的參數，利用數值方法 (Finite Difference + FFT) 解 GM 方程。
    *   **版本**：
        *   `gm_cupy.py`: 基於 **CuPy** (NVIDIA GPU) 的高效能版本。
        *   `gm_mps.py`: 針對 **Mac GPU (MPS)** 加速的版本。
    *   **產出**：`.npz` 檔案 (包含圖像 $u, v$ 和對應參數)。

*   **`merge.py` (數據整理)**
    *   **功能**：將分批生成的 `.npz` 檔案合併成一個完整的數據集，並處理 ID 排序。
    *   **產出**：`turing_patterns_dataset_merged.npz`。

-----

### 2. 模型核心庫 (`model/` 資料夾)

*   **`model/config.py` (統一配置)**
    *   **功能**：統一管理所有訓練與模型的超參數。
    *   **硬體設定**：自動偵測 CUDA/MPS/CPU。
    *   **物理參數**：`N_GRID=128`, `DX=1.0`, `S_DIFFUSION=0.4`。
    *   **訓練參數**：`BATCH_SIZE=32`, `EPOCHS=100`, `LEARNING_RATE=1e-4`。
    *   **數據分割**：
        *   `TRAIN_RATIO = 0.6` (60% 訓練)
        *   `VAL_RATIO = 0.2` (20% 驗證)
        *   `TEST_RATIO = 0.2` (20% 測試)
*   **`model/model.py` (大腦結構)**
    *   **功能**：定義神經網絡架構。
        *   `ParameterNet`: CNN 架構，提取圖像空間特徵。
        *   `MLPNet`: 全連接層架構 (Baseline)。
        *   `ForwardSurrogate`: 參數→圖像生成器 (混合方法使用)。
*   **`model/loss.py` (物理導師)**
    *   **功能**：定義各種損失函數。
        *   `PhysicsLoss`: 計算 PDE 殘差。
        *   `FourierLoss`: 頻域特徵損失。
        *   `HistogramLoss`: 統計分佈損失。
        *   `GradNorm`: 動態權重調整算法。
*   **`model/dataset.py` (資料搬運工)**
    *   **功能**：負責讀取 `.npz`，並進行數據正規化 (Normalization)。
*   **`model/utils.py` (工具箱)**
    *   **功能**：包含 `MinMaxScaler`、`calculate_multidim_nrmse` (多維 NRMSE 計算)、以及模型版本管理工具 (`get_next_version`, `find_model_path`)。

-----

### 3. 執行與驗證 (Execution & Verification)

*   **`train.py` (統一訓練腳本)**
    *   **功能**：整合了所有訓練模式 (Pure, PINN, Hybrid) 的單一入口。
    *   **主要參數**：
        *   `--pretrain`: 執行 Phase 1 (Surrogate) 訓練。
        *   `--train-model`: 選擇模型架構 (`CNN` / `MLP`)。
        *   `--use-loss`: 選擇 Loss 類型 (`pure`, `physical`, `surrogate`)。
        *   `--phys-gradual`: 啟用漸進式 Loss 引入。
        *   `--data-fraction`: 數據消融測試 (0.0 ~ 1.0)。
        *   `--surrogate-num`: 指定使用的 Surrogate 版本。
    *   **模型管理**：
        *   Phase 1 模型存於 `checkpoint/surrogate/`，命名格式 `Surrogate_{type}_{num}_{wandb}.pth`。
        *   Phase 2 模型存於 `checkpoint/`，命名格式 `{model}_{loss}_{num}_{wandb}_{best/last}.pth`。
    *   **數據分割**：
        *   Step 1: 80% (Train+Val) vs 20% (Test)。
        *   Step 2: 從 80% 中再分 3:1 為 Train 與 Val。
        *   Data Ablation 僅影響 Train+Val 的總量，Test Set 保持固定。

*   **`verify.py` (統一驗證腳本)**
    *   **功能**：在獨立的 **Test Set** (20%) 上評估模型效能。
    *   **參數**：
        *   `--model-type`: 模型架構 (預設 CNN)。
        *   `--loss-type`: Loss 類型 (預設 pure)。
        *   `--num`: 訓練編號 (預設最新)。
        *   `--suffix`: 模型後綴 (預設 best)。
        *   `--model-path`: 直接指定路徑 (最高優先級)。
    *   **指標**：計算 $R^2$, MAE, RMSE, 以及多維 NRMSE。

*   **`visual_check.py` (視覺/閉環驗證)**
    *   **功能**：將模型預測的參數帶回 GM 方程重新模擬，比較「原圖」與「重建圖」的相似度。

-----

### 4. 舊版腳本 (Deprecated)

以下腳本已被 `train.py` 取代，保留僅供參考：
*   `train_pinn.py`
*   `train_hybrid.py`
*   `model/config_pinn.py`
*   `model/config_hybrid.py`

-----

### 5. 新興方法 (Emerging Approaches)

#### 4.1 圖論電阻距離直方圖特徵 (RDH Features)

*   **`FE.py` (圖論特徵提取)**
    *   **功能**：將圖靈斑紋圖像轉換為圖論拓撲特徵。
        *   **加權圖構建**：根據像素值高低建立鄰接矩陣，高值區域間權重為1.0，跨越邊界權重為ε。
        *   **電阻距離計算**：基於電學類比計算節點間電阻距離。
        *   **兩階段處理**：先掃描數據集確定全局R_max，再生成12-bin直方圖特徵。
    *   **產出**：前12維是RDH特徵向量，用來捕捉圖像的拓撲結構特徵，再加上第1維增強特徵：最大濃度，用來捕捉圖案的絕對濃度資訊。
    *   **應用**：可用於特徵工程、圖像分類或作為傳統機器學習的輸入。

#### 4.2 混合循環訓練架構 (Hybrid Surrogate + Inverse)

*   **`train_hybrid.py` (雙向循環訓練)**
    *   **功能**：通過Forward Surrogate與Inverse CNN的循環一致性訓練，強化物理約束。
        *   **Phase 1**：訓練Forward Surrogate (參數→圖像)，使用FourierLoss + HistogramLoss，可選物理損失。
        *   **Phase 2&3**：訓練Inverse CNN (圖像→參數)，結合監督學習與循環物理約束。
        *   **課程學習**：使用sigmoid函數實現surrogate loss的平滑漸進引入 (epoch 20-60)，避免訓練不穩定。
        *   **優化器**：AdamW (weight_decay=0.01)，提供更好的泛化性能。
        *   **模型保存**：自動保存最佳驗證損失模型和最終模型。
    *   **產出**：`checkpoint/forward_surrogate.pth` (正向模擬器) + `checkpoint/inverse_cnn_hybrid.pth` (逆向預測器) + `checkpoint/CNN_PINN_best.pth` (最佳模型)。
    *   **優勢**：通過雙向映射驗證，獲得更強的物理一致性；平滑的損失引入提升訓練穩定性。

*   **`visualize_lambda_schedule.py` (損失調度可視化)**
    *   **功能**：可視化surrogate loss的漸進引入調度曲線。
    *   **產出**：`surrogate_loss_schedule.png` (調度曲線圖) + 控制台表格輸出。
    *   **使用**：`python visualize_lambda_schedule.py`

*   **`GRADUAL_SURROGATE_README.md` (詳細技術文檔)**
    *   **功能**：詳細說明surrogate loss漸進引入的技術實現和優勢。
    *   **內容**：sigmoid調度參數、訓練階段說明、可視化指南、兼容性說明。

-----

### 🚀 如何使用 (完整 Pipeline)

#### Step 1: 生成參數與數據

1.  生成合格參數表：
    ```bash
    python param-gen.py
    ```
2.  執行模擬 (選擇適合你硬體的腳本)：
    ```bash
    # NVIDIA GPU
    python gm_cupy.py
    # Mac Silicon
    python gm_mps.py
    ```
3.  合併數據 (如果分批生成)：
    ```bash
    python merge.py
    ```

#### Step 2: 設定實驗

打開 `model/config.py`，確認：
1.  `NPZ_PATH`: 指向合併後的 `.npz` 檔。
2.  `MODEL_TYPE`: 選擇 `"CNN"` 或 `"MLP"`。
3.  `USE_PHYSICS`: 選擇 `True` (PINN) 或 `False` (Pure Data)。

#### Step 3: 開始訓練

**推薦使用統一運行腳本** (無需手動切換config)：

```bash
# 🚀 快速運行 (使用預設配置: hybrid + train)
python3 run_experiment.py

# 自定義配置
python3 run_experiment.py --method pinn --action train
python3 run_experiment.py --method hybrid --action verify

# 混合循環方法 (Hybrid) 進階控制
# --phase 參數僅適用於 hybrid + train
python3 run_experiment.py --method hybrid --action train --phase both           # 訓練 Phase 1 + Phase 2&3 (預設)
python3 run_experiment.py --method hybrid --action train --phase surrogate_only # 只訓練 Phase 1 (Surrogate)
python3 run_experiment.py --method hybrid --action train --phase inverse_only   # 只訓練 Phase 2&3 (Inverse CNN)
```

**VS Code 一鍵執行**：直接在編輯器中運行 `run_experiment.py`，會使用最上方的默認配置！

**或者直接運行各別腳本** (需手動設定config)：
```bash
# 原始PINN方法
python3 train_pinn.py

# 混合循環方法 (預設執行 both phases)
python3 train_hybrid.py
```

#### Step 4: 驗證成效

**使用統一運行腳本**：

```bash
# 快速驗證 (使用預設配置)
python3 run_experiment.py --action verify

# 自定義驗證
python3 run_experiment.py --method pinn --action verify
python3 run_experiment.py --method hybrid --action visual_check
```

**或者直接運行驗證腳本**：
```bash
# 數值指標 (R², MAE, NRMSE)
python3 verify.py --method pinn    # 或 --method hybrid
python3 visual_check.py --method pinn  # 或 --method hybrid

# 可視化損失調度 (僅適用於hybrid方法)
python3 visualize_lambda_schedule.py
```

> **關於 NRMSE 指標的說明**：
> 本專案採用的 NRMSE 計算方式遵循論文中的「完全平均」(Fully Averaged) 定義：
> 1. 計算所有樣本與所有參數的誤差平方和。
> 2. 除以總預測點數 ($m \times d$) 後開根號，得到 RMSE。
> 3. 除以真實參數平均向量的歐幾里得範數 ($||y_{mean}||$) 進行標準化。
> 
> 公式：$NRMSE = \frac{\sqrt{\frac{1}{m \cdot d} \sum_{i=1}^{m} \sum_{j=1}^{d} (y_{i,j} - \hat{y}_{i,j})^2}}{||\bar{y}||_2}$

#### 進階方法使用指南

##### 方法A: 圖論特徵提取 (RDH Features)
```bash
# 使用統一腳本
python run_experiment.py --method fe --action extract

# 或直接運行
python FE.py
```
**注意**：需要安裝 CuPy，適合 NVIDIA GPU 環境。

##### 方法B: 混合循環訓練 (Hybrid Training)
```bash
# Phase 1: 訓練 Forward Surrogate
python run_experiment.py --method hybrid --action train --phase surrogate_only

# Phase 2&3: 訓練 Inverse CNN (需先有 Phase 1 模型)
python run_experiment.py --method hybrid --action train --phase inverse_only

# 一次跑完所有階段 (預設)
python run_experiment.py --method hybrid --action train --phase both

# 可視化損失調度曲線
python visualize_lambda_schedule.py
```
**產出**：`checkpoint/forward_surrogate.pth` + `checkpoint/inverse_cnn_hybrid.pth` + `checkpoint/CNN_PINN_best.pth`

-----

### 📊 方法比較總表

| 方法 | 核心技術 | 優勢 | 適用場景 | 計算需求 | LR Scheduler | 優化器 |
|------|----------|------|----------|----------|-------------|--------|
| **原始PINN** (`train_pinn.py`) | CNN/MLP + PDE殘差 | 直接物理約束，理論嚴謹 | 標準逆問題 | 中等 | Warm-up + Cosine | AdamW |
| **RDH特徵** (`FE.py`) | 圖論電阻距離 | 拓撲結構捕捉，解釋性強 | 特徵分析，可視化 | 高 (GPU) | Reduce LR on Plateau | Adam |
| **混合循環** (`train_hybrid.py`) | 雙向循環一致性 + 漸進損失 | 強物理一致性，平滑訓練，魯棒性高 | 高精度應用 | 高 | Warm-up + Cosine | AdamW |

**建議使用順序**：從原始PINN開始 → 嘗試混合循環 → 視需要提取RDH特徵進行分析。

-----

### 🔄 最新改進 (Latest Improvements)

#### v2.1 - 訓練穩定性與性能優化
*   **優化器升級**：全方法統一使用AdamW (weight_decay=0.01)，提升泛化性能
*   **學習率調度改進**：添加Warm-up階段 (前10% epochs)，避免訓練初期不穩定
*   **模型保存增強**：自動保存最佳驗證損失模型和最終訓練模型
*   **Surrogate Loss漸進引入**：使用sigmoid函數實現平滑過渡，避免突然損失跳躍
*   **Phase 1物理損失選項**：可選擇是否在正向代理訓練中添加物理約束
*   **可視化工具**：新增`visualize_lambda_schedule.py`用於損失調度分析
*   **詳細文檔**：新增`GRADUAL_SURROGATE_README.md`技術說明文檔

#### 訓練穩定性提升
- **之前**：Surrogate loss在epoch 30突然從0跳到0.01，造成梯度衝擊
- **現在**：使用sigmoid函數在epoch 20-60間平滑過渡，提升收斂穩定性

#### 性能指標改善
- 更平滑的訓練曲線
- 更好的最終收斂效果
- 增強的模型魯棒性

**查看詳細改進**：參考 `GRADUAL_SURROGATE_README.md` 獲取完整技術說明。
