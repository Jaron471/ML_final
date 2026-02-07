import numpy as np
import pandas as pd
from scipy.optimize import fsolve
import time

# --- 隨機種子設定 (確保可重現性) ---
SEED = 42
np.random.seed(SEED)

# --- 參數範圍 (基於論文 5.4 節) ---
PARAM_RANGES = {
    'a': [0.01, 0.7],
    'b': [0.4, 2.0],
    'c': [0.03, 7.0],
    'delta': [20.0, 200.0]
}

# --- 穩定性檢查函式 (從先前檔案複製，用於獨立驗證) ---

def calculate_gierer_meinhardt_steady_state(a, b, c):
    """計算 Gierer-Meinhardt 模型的均勻穩態 u* 和 v*。"""
    def steady_state_eq(u):
        if u <= 0: return 1e6
        return b * u - (a + 1.0 / (1.0 + c * u**2))

    u_initial_guess = (a + 1.0) / b
    try:
        # 使用 fsolve 尋找根
        u_star = fsolve(steady_state_eq, u_initial_guess, maxfev=200)[0]
    except:
        return None, None

    if u_star <= 0 or not np.isclose(steady_state_eq(u_star), 0, atol=1e-8):
        return None, None

    v_star = u_star**2
    return u_star, v_star

def check_gierer_meinhardt_turing_stability(a, b, c, delta):
    """
    檢查參數是否滿足局部穩定性和 DDI 條件。
    回傳值: (is_turing_stable, u_star, v_star)
    """
    u_star, v_star = calculate_gierer_meinhardt_steady_state(a, b, c)
    if u_star is None:
        return False, None, None

    # --- Jacobian 矩陣 J 項 ---
    f_u = -b + (2.0 * u_star) / (v_star * (1.0 + c * u_star**2)**2)
    f_v = -1.0 / (v_star * (1.0 + c * u_star**2))
    g_u = 2.0 * u_star
    g_v = -1.0

    # 1. 均勻穩態局部穩定性 (Trace < 0, Det > 0)
    Trace_J = f_u + g_v
    Det_J = f_u * g_v - f_v * g_u
    if not (Trace_J < 0 and Det_J > 0):
        return False, None, None

    # 2. 擴散驅動不穩定性 (DDI, 必須不穩定)
    ddi_trace_term = delta * f_u + g_v
    LHS = ddi_trace_term**2
    RHS = 4.0 * delta * Det_J

    if not (ddi_trace_term > 0 and LHS > RHS):
        return False, None, None

    # 參數合格
    return True, u_star, v_star

# --- 參數生成器函式 ---

def generate_qualified_turing_params(target_count=20000, output_filename='qualified_turing_params_final.csv'):
    """
    隨機生成參數，並使用 DDI 條件進行驗證，直到累積到目標數量。

    參數:
        target_count (int): 目標的合格參數數量。
        output_filename (str): 儲存合格參數的 CSV 檔案名稱。
    """

    start_time = time.time()
    qualified_params = []
    attempts = 0

    print(f"--- 啟動合格圖靈參數生成器 (目標數量: {target_count}) ---")
    print(f"參數範圍: {PARAM_RANGES}")

    while len(qualified_params) < target_count:
        attempts += 1

        # 1. 隨機生成參數 (在指定範圍內取均勻分佈亂數)
        a = np.random.uniform(*PARAM_RANGES['a'])
        b = np.random.uniform(*PARAM_RANGES['b'])
        c = np.random.uniform(*PARAM_RANGES['c'])
        delta = np.random.uniform(*PARAM_RANGES['delta'])

        # 2. 執行 DDI 檢查
        is_stable, u_star, v_star = check_gierer_meinhardt_turing_stability(a, b, c, delta)

        if is_stable:
            # 3. 合格則儲存
            current_id = len(qualified_params) + 1
            qualified_params.append({
                'id': current_id,
                'a': a,
                'b': b,
                'c': c,
                'delta': delta
            })

            # 顯示進度
            if current_id % 100 == 0:
                print(f"  [Progress] 累計合格: {current_id}/{target_count} | 總嘗試次數: {attempts} | 合格率: {current_id/attempts:.4f}")

    end_time = time.time()

    # 4. 儲存結果到 CSV 檔案
    qualified_df = pd.DataFrame(qualified_params)
    qualified_df.to_csv(output_filename, index=False)

    print("\n========================================================")
    print("✅ 參數生成完成!")
    print(f"總耗時: {end_time - start_time:.2f} 秒")
    print(f"最終合格參數數量: {len(qualified_df)}")
    print(f"總嘗試次數: {attempts}")
    print(f"最終合格率: {len(qualified_df) / attempts:.4f}")
    print(f"檔案已儲存至: {output_filename}")
    print("========================================================")

    return qualified_df

# --- 執行範例 ---
# 注意: 使用固定隨機種子 (SEED=42)，確保每次執行結果可重現。
# 運行以下函式將生成並儲存檔案。

# 執行生成 20,000 筆參數的任務
if __name__ == '__main__':
    # 為了演示，我們使用一個較小的數量 (例如 1000 筆)
    # 如果您想生成 20,000 筆，請將 target_count 設置為 20000
    target_count = 20500  # 修改此數值以改變目標數量
    output_filename = f'qualified_turing_params_{target_count}.csv'
    generated_df = generate_qualified_turing_params(
        target_count=target_count,
        output_filename=output_filename
    )
    # print(generated_df.head()) # 打印前幾行查看