#!/usr/bin/env python3
"""
統一實驗運行腳本 (Unified Experiment Runner)

這個腳本讓你可以通過簡單的參數來決定要訓練或驗證哪種方法，
無需手動切換不同的config文件和訓練腳本。

快速使用:
    python run_experiment.py                    # 使用預設配置直接運行
    python run_experiment.py --method pinn     # 切換方法，動作使用預設
    python run_experiment.py --action verify   # 切換動作，方法使用預設

參數說明:
    --method: 選擇方法 (預設: hybrid)
        - pinn: 原始PINN方法 (CNN/MLP + PDE殘差)
        - hybrid: 混合循環訓練 (Forward Surrogate + Inverse CNN)
        - fe: 圖論特徵提取 (RDH Features)

    --action: 選擇動作 (預設: train)
        - train: 訓練模型
        - verify: 數值驗證 (R², MAE, NRMSE)
        - visual_check: 視覺閉環驗證 (重新模擬比較)
        - extract: 提取特徵 (僅適用於 fe 方法)

    --phase: 訓練階段 (僅適用於 hybrid + train)
        - both: 訓練 Phase 1 (Surrogate) + Phase 2&3 (Inverse) [預設]
        - surrogate_only: 只訓練 Phase 1 (Surrogate)
        - inverse_only: 只訓練 Phase 2&3 (Inverse CNN)

範例:
    # 直接運行 (使用預設 hybrid + train)
    python run_experiment.py

    # 訓練原始PINN方法
    python run_experiment.py --method pinn --action train

    # 訓練Hybrid方法 (只訓練Surrogate)
    python run_experiment.py --method hybrid --action train --phase surrogate_only

    # 訓練Hybrid方法 (只訓練Inverse CNN，假設Surrogate已存在)
    python run_experiment.py --method hybrid --action train --phase inverse_only

    # 驗證混合循環方法
    python run_experiment.py --method hybrid --action verify

    # 提取圖論特徵
    python run_experiment.py --method fe --action extract
"""

# ==========================================
# 🚀 快速配置區塊 (修改這裡來設定默認行為)
# ==========================================
DEFAULT_METHOD = 'pinn'  # 預設方法: 'pinn', 'hybrid', 或 'fe'
DEFAULT_ACTION = 'train'   # 預設動作: 'train', 'verify', 'visual_check', 或 'extract'

# 💡 提示: 修改上面的變數來改變一鍵執行的預設行為！
#    例如: DEFAULT_METHOD = 'hybrid' 會讓直接運行時使用PINN方法

import argparse
import sys
import os
import subprocess

def main():
    parser = argparse.ArgumentParser(
        description="統一實驗運行腳本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        '--method',
        choices=['pinn', 'hybrid', 'fe'],
        default=DEFAULT_METHOD,
        help=f'選擇實驗方法 (預設: {DEFAULT_METHOD})'
    )

    parser.add_argument(
        '--action',
        choices=['train', 'verify', 'visual_check', 'extract'],
        default=DEFAULT_ACTION,
        help=f'選擇執行動作 (預設: {DEFAULT_ACTION})'
    )

    parser.add_argument(
        '--phase',
        choices=['both', 'surrogate_only', 'inverse_only'],
        default='both',
        help='[Hybrid only] 訓練階段: both=Phase1+Phase2&3, surrogate_only=只Phase1, inverse_only=只Phase2&3'
    )

    args = parser.parse_args()

    print("🚀 Turing Pattern Inversion Network - 統一運行腳本")
    print(f"📋 方法: {args.method.upper()}")
    print(f"🎯 動作: {args.action}")
    
    # 檢查是否使用默認配置
    using_defaults = (args.method == DEFAULT_METHOD and args.action == DEFAULT_ACTION)
    if using_defaults:
        print("⚡ 使用預設配置快速運行")
    else:
        print("🔧 使用自定義配置")
    print("-" * 50)

    # 根據method和action決定要運行的腳本
    if args.method == 'pinn':
        if args.action == 'train':
            print("▶️  開始訓練原始PINN方法...")
            run_script('train_pinn.py')

        elif args.action == 'verify':
            print("🔍 開始驗證原始PINN方法...")
            run_command(['python', 'verify.py', '--method', 'pinn'])

        elif args.action == 'visual_check':
            print("👁️  開始視覺驗證原始PINN方法...")
            run_command(['python', 'visual_check.py', '--method', 'pinn'])

    elif args.method == 'hybrid':
        if args.action == 'train':
            print("▶️  開始訓練混合循環方法...")
            print(f"📍 訓練階段: {args.phase}")
            run_script('train_hybrid.py', phase=args.phase)

        elif args.action == 'verify':
            print("🔍 開始驗證混合循環方法...")
            run_command(['python', 'verify.py', '--method', 'hybrid'])

        elif args.action == 'visual_check':
            print("👁️  開始視覺驗證混合循環方法...")
            run_command(['python', 'visual_check.py', '--method', 'hybrid'])

    elif args.method == 'fe':
        if args.action == 'extract':
            print("📊 開始提取圖論電阻距離特徵...")
            run_script('FE.py')

    print("\n✅ 任務完成！")

def modify_config_for_pinn():
    """臨時修改config_pinn.py以適應當前的工作流程"""
    # 這裡可以添加動態修改config的邏輯
    # 例如根據當前設置動態調整MODEL_SAVE_PATH
    pass

def run_script(script_name, phase='both'):
    """運行指定的Python腳本"""
    try:
        # 如果需要傳遞phase參數，通過環境變數傳遞
        env = os.environ.copy()
        if phase != 'both':
            env['HYBRID_TRAIN_PHASE'] = phase
        # 使用subprocess來運行腳本
        result = subprocess.run([sys.executable, script_name], env=env, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 運行腳本時發生錯誤: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ 找不到腳本文件: {script_name}")
        sys.exit(1)

def run_command(command_list):
    """運行命令列表"""
    try:
        result = subprocess.run(command_list, check=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"❌ 命令執行失敗: {' '.join(command_list)}")
        print(f"錯誤信息: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print(f"❌ 找不到命令: {command_list[0]}")
        sys.exit(1)

if __name__ == "__main__":
    main()