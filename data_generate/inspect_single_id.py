# -*- coding: utf-8 -*-
import argparse
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# Attempt to import gm_mps from current directory
try:
    from gm_mps import simulate_gm_mps, calculate_gierer_meinhardt_steady_state, PARAM_CSV, SEED
except ImportError:
    # Handle case where script is run from root directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from gm_mps import simulate_gm_mps, calculate_gierer_meinhardt_steady_state, PARAM_CSV, SEED

def inspect_id(target_id):
    if not os.path.exists(PARAM_CSV):
        print(f"❌ 找不到參數檔: {PARAM_CSV}")
        return

    df = pd.read_csv(PARAM_CSV)
    
    # Try to find row by 'id' column
    if "id" in df.columns:
        target_row = df[df["id"] == target_id]
        if target_row.empty:
            print(f"⚠️ CSV 中找不到 id={target_id}。嘗試直接使用 index={target_id}...")
            if 0 <= target_id < len(df):
                target_row = df.iloc[[target_id]]
            else:
                print(f"❌ Index {target_id} 超出範圍 (0~{len(df)-1})")
                return
    else:
        # Fallback to index
        if 0 <= target_id < len(df):
            target_row = df.iloc[[target_id]]
            # If no 'id' column, we assume id is index
            target_row["id"] = target_id 
        else:
             print(f"❌ Index {target_id} 超出範圍 (0~{len(df)-1})")
             return
    
    row = target_row.iloc[0]
    
    # Robustly get ID
    pid = int(row.get("id", target_id))
    a = float(row["a"])
    b = float(row["b"])
    c = float(row["c"])
    delta = float(row["delta"])
    
    print(f"🔍 正在模擬 ID {pid} (a={a:.4f}, b={b:.4f}, c={c:.4f}, delta={delta:.4f})...")
    
    # Calculate steady state
    u_star, v_star = calculate_gierer_meinhardt_steady_state(a, b, c)
    if u_star is None:
        print("❌ 無法計算穩態 (Invalid Steady State)")
        return

    print(f"   穩態: u*={u_star:.4f}, v*={v_star:.4f}")
    
    # Run simulation
    start_time = time.time()
    # Note: simulate_gm_mps inside uses global constants (MAX_T_STEPS etc.) from gm_mps module
    # If we wanted to override them we'd need to modify the function signature or module vars.
    # For now we use the logic defined in gm_mps (including the dynamic extension).
    u, v, steps = simulate_gm_mps(pid, a, b, c, delta, u_star, v_star, seed=SEED+pid)
    elapsed = time.time() - start_time
    
    # Stats
    u_std = np.std(u)
    u_range = np.max(u) - np.min(u)
    print(f"✅ 模擬完成 ({elapsed:.2f}s). Steps: {steps}")
    print(f"   u_std: {u_std:.6f}")
    print(f"   u_range: {u_range:.6f}")
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    im1 = axes[0].imshow(u, cmap='magma', origin='lower')
    axes[0].set_title(f"Activator u (ID {pid})\nStd={u_std:.4f}")
    plt.colorbar(im1, ax=axes[0])
    
    im2 = axes[1].imshow(v, cmap='viridis', origin='lower')
    axes[1].set_title(f"Inhibitor v (ID {pid})")
    plt.colorbar(im2, ax=axes[1])
    
    output_png = f"generate_id_{pid}.png"
    plt.tight_layout()
    plt.savefig(output_png)
    print(f"📸 結果圖已儲存: {output_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and inspect Turing pattern for a specific ID")
    parser.add_argument("id", type=int, help="The parameter ID (0-19999)")
    args = parser.parse_args()
    
    inspect_id(args.id)
