
-----

### 1\. 數據 (Data Generation)(這部分不重要)

這部分的程式碼負責扮演「上帝」，利用物理方程創造數據。

  * **`GM_new.py` (核心引擎)**
      * **功能**：定義了 Gierer-Meinhardt (GM) 反應擴散方程的數值解法。
      * **特點**：使用 **CuPy** (GPU 版的 NumPy) 搭配 FFT (快速傅立葉變換) 進行運算，這是加速模擬的關鍵。它包含計算穩態 (`steady_state`) 和時間步進 (`simulate_gm_cupy`) 的函式。
  * **`GPU_generate.py` (生產線)**
      * **功能**：這是用來大量生產數據的腳本。它會讀取 CSV 裡的參數範圍，利用多進程 (Multiprocessing) 呼叫 `GM_new.py`，生成幾千張圖靈斑紋圖像。
      * **產出**：儲存成 `.npz` 檔（包含圖像 $u, v$ 和對應的參數 $a, b, c, \delta$）。
  * **`merge_dataset.py` (數據整理)**
      * **功能**：如果你分批生成了好幾個 `.npz` 檔案，這個腳本可以把它們合併成一個大的，並且自動檢查、刪除重複的 ID。

-----

### 2\. 模型核心庫 (`model/` 資料夾)


  * **`model/config.py` (控制中心)**
      * **功能**：存放所有的超參數 (Hyperparameters) 和路徑設定。
      * **重點**：你要切換實驗（例如 CNN 改 MLP，或是關掉 Physics Loss），只需要改這裡的 `MODEL_TYPE` 或 `USE_PHYSICS`，不需要動其他程式碼。
  * **`model/model.py` (大腦結構)**
      * **功能**：定義神經網絡架構。
          * `ParameterNet`: CNN 架構，擅長抓取圖像的空間特徵（斑點大小、密度）。
          * `MLPNet`: 全連接層架構，用於作為 Baseline 對照組。
  * **`model/loss.py` (物理導師)**
      * **功能**：定義 **Physics Loss (PINN)**。它將模型預測出的參數帶回物理方程，計算 PDE 的殘差 (Residual)。如果預測的參數不符合物理定律，這裡會產生很大的 Loss。
  * **`model/dataset.py` (資料搬運工)**
      * **功能**：繼承 PyTorch 的 Dataset。負責讀取 `.npz`，並將參數進行 **正規化 (Normalization)** 縮放到 $[0, 1]$ 區間，讓模型好訓練。
  * **`model/utils.py` (工具箱)**
      * **功能**：定義了 `MinMaxScaler`，負責處理數值的正規化與反正規化。

-----

### 3\. 執行與驗證 (Execution & Verification)

這部分是你用來跟模型互動的介面。

  * **`train.py` (訓練腳本)**
      * **功能**：整合所有模組開始訓練。
      * **特點**：
          * 連接 **WandB** 進行雲端監控。
          * 使用 **Weighted Loss** (針對 $\delta$ 加權)。
          * 每個 Epoch 結束會計算驗證集的 NRMSE 並上傳圖表。
  * **`verify.py` (數值驗證)**
      * **功能**：訓練完後，用來「考試」的。
      * **產出**：計算 $R^2$ Score, MAE, NRMSE，並畫出「真實 vs 預測」的散點圖 (Scatter Plot)。這是量化模型準度最直接的證據。
  * **`visual_check.py` (視覺/閉環驗證)**
      * **功能**：最高級的驗證（Re-simulation）。
      * **流程**：`原圖` $\rightarrow$ `模型預測參數` $\rightarrow$ `丟回 GPU_generate 模擬` $\rightarrow$ `生成新圖`。
      * **目的**：直接比較原圖和新圖長得像不像。如果像，證明模型真正學會了物理機制。

-----

### 🚀 如何使用 (完整 Pipeline)

#### Step 1: 準備數據

如果你還沒有 `.npz` 數據，或者想生成新的：

1.  確保有 GPU 環境。
2.  執行生成：
    ```bash
    python GPU_generate.py
    ```
3.  (選用) 如果有多個檔案，執行合併：
    ```bash
    python merge_dataset.py
    ```

#### Step 2: 設定參數

打開 `model/config.py`，確認以下幾點：

1.  `NPZ_PATH`: 指向你剛生成的檔案。
2.  `LOSS_WEIGHTS`: 建議設為 `[1.0, 5.0, 0.0, 5.0]` (忽略 $c$，加重 $\delta$)。
3.  `MODEL_TYPE`: 設為 `"CNN"`。
4.  `USE_PHYSICS`: 設為 `True`。

#### Step 3: 開始訓練

執行訓練腳本，並觀察 WandB 上的曲線：

```bash
python train.py
```

  * 模型會儲存為 `turing_cnn_pinn.pth` (或是根據 config 動態命名的檔案)。

#### Step 4: 驗證成效

訓練結束後，執行數值驗證看分數：

```bash
python verify.py
```

*(查看 NRMSE 是否小於 10%，特別是 delta)*

#### Step 5: 終極檢查

執行視覺驗證，看看模型是否學會了「重現」圖靈斑紋：

```bash
python visual_check.py
```

*(打開生成的 `visual_validation_resimulation.png`，比較左邊和中間的圖)*