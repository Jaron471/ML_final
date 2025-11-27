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

*   **`model/config.py` (控制中心)**
    *   **功能**：存放所有的超參數 (Hyperparameters) 和路徑設定。
    *   **重點**：修改 `MODEL_TYPE` ("CNN" 或 "MLP") 和 `USE_PHYSICS` (True/False) 來切換實驗設定。
*   **`model/model.py` (大腦結構)**
    *   **功能**：定義神經網絡架構。
        *   `ParameterNet`: CNN 架構，提取圖像空間特徵。
        *   `MLPNet`: 全連接層架構 (Baseline)。
*   **`model/loss.py` (物理導師)**
    *   **功能**：定義 **Physics Loss**。計算 PDE 殘差，確保預測參數符合物理定律。
*   **`model/dataset.py` (資料搬運工)**
    *   **功能**：負責讀取 `.npz`，並進行數據正規化 (Normalization)。
*   **`model/utils.py` (工具箱)**
    *   **功能**：包含 `MinMaxScaler` 等輔助工具。

-----

### 3. 執行與驗證 (Execution & Verification)

*   **`train.py` (訓練腳本)**
    *   **功能**：讀取數據並訓練模型。支援 WandB 監控。
    *   **產出**：訓練好的模型權重 (例如 `CNN_PINN.pth`)。
*   **`verify.py` (數值驗證)**
    *   **功能**：計算測試集的 $R^2$, MAE, NRMSE，評估模型準確度。
*   **`visual_check.py` (視覺/閉環驗證)**
    *   **功能**：將模型預測的參數帶回 GM 方程重新模擬，比較「原圖」與「重建圖」的相似度。

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

```bash
python train.py
```

#### Step 4: 驗證成效

1.  數值指標：
    ```bash
    python verify.py
    ```
2.  視覺驗證：
    ```bash
    python visual_check.py
    ```
