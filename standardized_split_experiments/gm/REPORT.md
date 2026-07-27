# Gierer–Meinhardt：Pure、Physical 與 Soft Physical 比較

## 實驗設定

- 資料：12,000 training / 4,000 validation / 4,000 test
- 模型：原版 `cnn1`（`nk=5`、kernel 5、hidden 5）
- 輸入：`u`
- 輸出：`[a, b, c, delta]`
- Epochs：100；seed：42
- 三組使用相同初始化、batch 順序、optimizer 與 scheduler
- 以 validation normalized RMSE 選擇 best checkpoint
- 統計：5,000 次 paired bootstrap

新版仍直接使用 CNN 輸出，不含解析求解、least-squares、hard projection
或參數後處理。

## Final evaluation 結果

| 指標 | Pure | Physical | Soft Physical |
| --- | ---: | ---: | ---: |
| Overall normalized RMSE | 0.159949 | 0.155344 | **0.152762** |
| Paper joint NRMSE | 0.167151 | 0.162339 | **0.159641** |
| `a` RMSE | **0.100423** | 0.116403 | 0.116737 |
| `b` RMSE | **0.203363** | 0.219102 | 0.226760 |
| `c` RMSE | 0.583804 | 0.452916 | **0.409043** |
| `delta` RMSE | 42.992294 | 37.852795 | **36.168640** |
| Masked PDE residual | 0.970302 | 0.825136 | **0.735990** |
| Best epoch | 71 | 96 | 94 |

相對 Pure：

- Physical：overall 改善 **2.88%**、`c` 改善 **22.42%**、
  `delta` 改善 **11.95%**。
- Soft Physical：overall 改善 **4.49%**、`c` 改善 **29.93%**、
  `delta` 改善 **15.87%**。
- Soft Physical 的 PDE residual 相對 Pure 降低 **24.15%**，相對原
  Physical 再降低 **10.80%**。
- Pure 在 `a`、`b` 仍然最好；Soft Physical 分別較 Pure 差
  16.24% 與 11.51%。

## Bootstrap 結論

- Physical vs Pure overall difference 95% CI：
  `[-0.001814, -0.001096]`。
- Soft Physical vs Pure overall difference 95% CI：
  `[-0.002699, -0.001821]`。
- Soft Physical vs Physical overall difference 95% CI：
  `[-0.000940, -0.000653]`。

三個 CI 均完全小於 0，因此 overall 排名具有統計支持：

> **Soft Physical > Physical > Pure**

逐參數結果則是：

- `c`、`delta`：Soft Physical 顯著優於 Physical 與 Pure。
- `a`：Soft Physical 與 Physical 差異不顯著，但兩者都比 Pure 差。
- `b`：Pure 最好，Soft Physical 最差。

## 新版方法

原版 Physical：

\[
L=L_{\mathrm{sup}}+\lambda_{\mathrm{PDE}}L_{\mathrm{PDE}}
\]

Soft Physical 額外加入：

\[
L_{\mathrm{new}}
=L_{\mathrm{sup}}+\lambda_{\mathrm{PDE}}L_{\mathrm{PDE}}
+\lambda_{\mathrm{int}}L_{\mathrm{integrated}}
+\lambda_{\delta}L_{\delta\text{-normal}}
\]

- `L_integrated`：第一條穩態 PDE 在週期空間上的積分 identity。
- `L_delta-normal`：第二條 PDE 與 `Δv` 內積後得到的 scalar constraint。
- 兩者只在訓練時使用；推論仍是 `u → CNN → [a,b,c,delta]`。

最終權重：

- `lambda_physics=0.01`
- `lambda_integrated=0.1`
- `lambda_delta=0.1`
- soft constraint 對共享 CNN features 的 gradient scale：`0.5`

## 資料限制

evaluation set 有 3,893/4,000 筆的 `final_step=25000`，即模擬跑到設定的最大
步數，而不是依收斂條件提前停止。因此這批場不能全部視為嚴格穩態；
steady soft constraints 在此是近似 regularizer。若要做更強的物理解釋，
應重新產生通過明確 convergence tolerance 的資料再驗證。

## 輸出檔案

- Checkpoints、metrics、training histories、bootstrap、CSV 與逐筆
  predictions：`standardized_split_experiments/gm/`
- 訓練程式：`train_gm.py`
- 比較程式：`compare_gm_methods.py`
