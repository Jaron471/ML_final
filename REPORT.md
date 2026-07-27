# 統一 12k／4k／4k PDE 訓練報告

## 實驗目的

將 Gierer–Meinhardt、Schnakenberg、Brusselator、Lengyel–Epstein
的資料與訓練流程統一為：

- training：12,000 筆
- validation：4,000 筆
- final evaluation：4,000 筆
- model seed：42
- generated PDE train/validation split seed：42；GM 使用固定預先切分
- epochs：100
- checkpoint selection：最低 validation normalized RMSE
- final evaluation：載入 validation 最佳 checkpoint 後才執行

GM 原始資料本來就是獨立的 12,000／4,000／4,000。其他三個 PDE 的
`*_train_data.npz` 各有 16,000 筆，使用
`numpy.random.default_rng(42).permutation(16000)` 固定取 4,000 筆作為
validation，其餘 12,000 筆作為 training。原本獨立的 4,000 筆
`*_eval_data.npz` 保持不變。

三個 PDE 寫出的 split 檔 SHA-256 均為：

```text
66340e6eda6115c98d70e281d98b2173528170f627b2c0a139b2c4129d7c5a9a
```

代表它們使用完全相同的位置索引切分。Pure、Physical、Soft-Physical
在同一 PDE 中也共用完全相同的 scaler、模型 seed 與資料順序。

## 保留的模型與 loss 設定

這次沒有針對新的 validation 重新調整 soft loss 超參數，而是保留先前
已使用的架構與方法設定，只改變資料 protocol 與 checkpoint selection。

| PDE | Architecture | Physical 設定 | Soft-Physical 額外設定 |
| --- | --- | --- | --- |
| Gierer–Meinhardt | `cnn1` | `lambda_physics=0.01` | `integrated=0.1`、`delta-normal=0.1`、feature gradient scale `0.5` |
| Schnakenberg | `cnn2stride` | `lambda_physics=0.01` | `identity=0.1`、只更新既有 parameter head |
| Brusselator | `cnn1` | baseline 使用 `lambda_physics=0.01` | `physics=0.0065`、`identity=0.15`、`elimination=0.05`、feature gradient scale `0.8`、supervised raw weights `0.75,1.3,0.9` |
| Lengyel–Epstein | `cnn2stride` | baseline 使用 `lambda_physics=0.01` | `physics=0.001`、`identity=0.5`、`elimination=1.0`、完整 CNN gradient |

所有方法在 inference 都只執行原始 `model(u)`：

- 無 hard projection
- 無 least-squares parameter solve
- 無解析參數覆寫
- 無 inference-time postprocessing

## Final evaluation 結果

### Overall normalized RMSE

| PDE | Pure | Physical | Soft-Physical | 最佳方法 |
| --- | ---: | ---: | ---: | --- |
| Gierer–Meinhardt | 0.159949 | 0.155344 | **0.152762** | Soft-Physical |
| Schnakenberg | **0.046468** | 0.046995 | 0.046955 | Pure |
| Brusselator | 0.133998 | 0.137034 | **0.133553** | Soft-Physical |
| Lengyel–Epstein | 0.070772 | **0.070030** | 0.070698 | Physical |

Validation 選出的最佳 epochs：

| PDE | Pure | Physical | Soft-Physical |
| --- | ---: | ---: | ---: |
| Gierer–Meinhardt | 71 | 96 | 94 |
| Schnakenberg | 100 | 100 | 98 |
| Brusselator | 99 | 97 | 87 |
| Lengyel–Epstein | 99 | 100 | 99 |

### Gierer–Meinhardt

| 指標 | Pure | Physical | Soft-Physical |
| --- | ---: | ---: | ---: |
| Overall normalized RMSE | 0.159949 | 0.155344 | **0.152762** |
| `a` RMSE | **0.100423** | 0.116403 | 0.116737 |
| `b` RMSE | **0.203363** | 0.219102 | 0.226760 |
| `c` RMSE | 0.583804 | 0.452916 | **0.409043** |
| `delta` RMSE | 42.992294 | 37.852795 | **36.168640** |
| Mean PDE residual | 0.970302 | 0.825136 | **0.735990** |

Soft-Physical 相對 Pure：

- overall：改善 4.493%
- `c`：改善 29.935%
- `delta`：改善 15.872%
- PDE residual：改善 24.148%
- `a`、`b` 分別惡化 16.245% 與 11.505%

Overall paired squared-error difference 的 5,000 次 bootstrap 95% CI 為
`[-2.699e-3, -1.821e-3]`，Soft-Physical overall 改善具有統計支持。
完整 GM 分析見 `gm/REPORT.md`。

### Schnakenberg

| 指標 | Pure | Physical | Soft-Physical |
| --- | ---: | ---: | ---: |
| Overall normalized RMSE | **0.046468** | 0.046995 | 0.046955 |
| `a` RMSE | **0.021431** | 0.022652 | 0.022671 |
| `b` RMSE | **0.031349** | 0.031496 | 0.031364 |
| `d` RMSE | 5.222952 | 5.118591 | **5.112582** |
| Mean PDE residual | 0.024965 | **0.022669** | 0.022724 |

Soft-Physical 相對 Pure：

- overall：惡化 1.047%
- `a`：惡化 5.785%
- `b`：惡化 0.047%
- `d`：改善 2.113%
- PDE residual：改善 8.978%

Overall paired squared-error difference 的 5,000 次 bootstrap 95% CI 為
`[2.923e-5, 6.196e-5]`，完整大於零，因此 Soft-Physical 的 overall
確實比 Pure 差；PDE residual difference CI 為
`[-2.632e-3, -1.837e-3]`，物理一致性則顯著改善。

### Brusselator

| 指標 | Pure | Physical | Soft-Physical |
| --- | ---: | ---: | ---: |
| Overall normalized RMSE | 0.133998 | 0.137034 | **0.133553** |
| `A` RMSE | **0.143544** | 0.187758 | 0.152124 |
| `B` RMSE | **0.091096** | 0.108050 | 0.091936 |
| `d` RMSE | 8.566790 | **5.729907** | 8.057796 |
| Mean PDE residual | 1.044832 | **0.645245** | 0.889291 |

Soft-Physical 相對 Pure：

- overall：改善 0.332%
- `A`：惡化 5.977%
- `B`：惡化 0.922%
- `d`：改善 5.941%
- PDE residual：改善 14.887%

Overall paired squared-error difference 的 95% CI 為
`[-2.010e-4, -3.519e-5]`，Soft-Physical overall 改善具有統計支持。
PDE residual difference CI 為 `[-0.164281, -0.146929]`，也顯著改善。

Physical baseline 的 `d` 與 PDE residual 改善幅度更大，但犧牲
`A`、`B`，使 overall 比 Pure 惡化 2.266%。

### Lengyel–Epstein

| 指標 | Pure | Physical | Soft-Physical |
| --- | ---: | ---: | ---: |
| Overall normalized RMSE | 0.070772 | **0.070030** | 0.070698 |
| `a` RMSE | 0.182062 | 0.182656 | **0.180946** |
| `phi` RMSE | 0.054102 | 0.054529 | **0.053985** |
| `r` RMSE | 16.783356 | **16.475641** | 16.786493 |
| Mean PDE residual | 0.192671 | **0.172035** | 0.191524 |

Soft-Physical 相對 Pure：

- overall：改善 0.105%
- `a`：改善 0.613%
- `phi`：改善 0.215%
- `r`：惡化 0.019%，可視為幾乎相同
- PDE residual：改善 0.595%

Overall paired squared-error difference 的 95% CI 為
`[-1.785e-5, -2.858e-6]`；PDE residual difference CI 為
`[-1.612e-3, -6.781e-4]`，兩者均支持 Soft-Physical 優於 Pure。

Physical baseline 是這個 PDE 的 overall 最佳方法：相對 Pure 改善
1.049%，主要來自 `r` 改善 1.833%，PDE residual 改善 10.710%。

## 結論

統一為 12k／4k／4k 並使用 validation 選模後：

1. 四個 Soft-Physical 都比 Pure 有較低的 PDE residual。
2. GM、Brusselator 與 Lengyel–Epstein 的 Soft-Physical overall
   顯著優於 Pure。
3. Schnakenberg 的 Soft-Physical 改善 `d` 與 PDE residual，但犧牲
   `a`，因此 overall 顯著低於 Pure。
4. Soft-Physical 並沒有在所有 PDE、所有參數上全面勝出。
5. 目前各 PDE 的最佳 overall 方法不同：GM 與 Brusselator 為
   Soft-Physical、Schnakenberg 為 Pure、Lengyel–Epstein 為 Physical。

這次的正式結論應描述為：

> Training-only soft constraints 在四個 PDE 上都能提升物理一致性，
> 並在 GM、Brusselator 與 Lengyel–Epstein 上顯著改善整體參數預測；
> Schnakenberg 則存在參數 accuracy 與 physical consistency 的
> trade-off。

## 產出位置

共用 trainer：

- `../train_brusselator.py`
- `../train_gm.py`（GM 四參數專屬）

三方法統一比較：

- `../compare_standardized_pde_methods.py`
- `../compare_gm_methods.py`

每個 PDE 目錄均包含：

- 3 個 validation-best checkpoints
- 3 個 metrics JSON
- 3 個完整 training histories
- three-method comparison JSON／CSV
- 4,000 筆逐樣本 predictions、PDE residual 與 identity error

三個三參數 PDE 另外保存固定 train/validation split indices；GM 使用
原本就已獨立的 train／validation／evaluation 檔案。

輸出目錄：

- `schnakenberg/`
- `brusselator/`
- `lengyel_epstein/`
- `gm/`

## 資料使用限制

三參數 PDE 的三組 4,000 筆 evaluation 在先前的 baseline 與 soft-constraint
實驗中已經被查看過，因此無法重新稱為完全未見的 test set。本次
checkpoint selection 僅使用新切出的 4,000 筆 validation，沒有用
evaluation 挑選 epoch；但若後續根據本報告反覆調整 soft
hyperparameters，正式論文應重新生成一組未見 evaluation，或事先鎖定
所有設定後再做一次新的 final test。

此外，GM evaluation 有 3,893/4,000 筆模擬到
`final_step=25000`，而不是依 tolerance 提前停止，因此不能全部視為
嚴格穩態；GM soft constraints 應解讀為近似 regularizer。
