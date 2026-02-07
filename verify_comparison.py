import os
import sys
import re
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from model import config
from model.dataset import TuringDataset
from model.model import (
    MLPNet, PaperMinimalCNN, PaperMinimalCNN2, 
    PaperMinimalCNN2MaxPool, PaperMinimalCNN2Stride,
    PaperFlexibleCNN
)

def main():
    # Update this path if needed - using verified dataset without pattern-less samples
    TEST_PATH = "test_data_verified.npz"
    CKPT_DIR = "paper_checkpoints"
    
    if not os.path.exists(CKPT_DIR):
        print("Checkpoint directory not found.")
        return

    # Auto-scan all files in checkpoint dir
    model_files = [f for f in os.listdir(CKPT_DIR) if f.endswith('.pth') and 'best' in f]
    
    dataset = TuringDataset(TEST_PATH)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    results = []

    print(f"🚀 Found {len(model_files)} models. Starting Evaluation...")
    
    for filename in sorted(model_files):
        filepath = os.path.join(CKPT_DIR, filename)
        
        # Parse Filename
        # 1. MLP: PaperPINN_MLP_{loss}_n{num}...
        match_mlp = re.search(r'PaperPINN_MLP_(?P<loss>[a-z]+)(_lam(?P<lam>[\d\.]+))?_n(?P<num>\d+)', filename)
        
        # 2. Old CNN: PaperPINN_CNN{ver}_{loss}_nk{nk}_n{num}...
        match_cnn = re.search(r'PaperPINN_CNN(?P<ver>\d*[a-z]*)_(?P<loss>[a-z]+)(_lam(?P<lam>[\d\.]+))?_nk(?P<nk>\d+)_n(?P<num>\d+)', filename)

        # 4. New Flexible CNN: PaperPINN_CNN_Flex_L{L}_{samp}_{loss}_n{num}...
        match_flex = re.search(r'PaperPINN_CNN_Flex_L(?P<L>\d+)_(?P<samp>\w+)_(?P<loss>[a-z]+)(_lam(?P<lam>[\d\.]+))?_n(?P<num>\d+)', filename)

        lambda_val = "N/A"

        if match_flex:
            arch = 'flexible_cnn'
            layers = int(match_flex.group('L'))
            sampling = match_flex.group('samp')
            loss = match_flex.group('loss')
            nsamp = int(match_flex.group('num'))
            if match_flex.group('lam'): lambda_val = match_flex.group('lam')
            
            model = PaperFlexibleCNN(nk=5, np_size=5, nf=5, layers=layers, sampling=sampling).to(config.DEVICE)
            desc = f"FlexCNN (L={layers}, {sampling})"
            
        elif match_mlp:
            arch = 'mlp'
            loss = match_mlp.group('loss')
            nsamp = int(match_mlp.group('num'))
            if match_mlp.group('lam'): lambda_val = match_mlp.group('lam')
            model = MLPNet().to(config.DEVICE)
            desc = "MLP"
            
        elif match_cnn:
            ver = match_cnn.group('ver') # '1', '2', '2pool'
            loss = match_cnn.group('loss')
            nk = int(match_cnn.group('nk'))
            nsamp = int(match_cnn.group('num'))
            if match_cnn.group('lam'): lambda_val = match_cnn.group('lam')
            
            # Instantiate correct old CNN
            if ver == '1' or ver == '': cls = PaperMinimalCNN
            elif ver == '2': cls = PaperMinimalCNN2
            elif 'pool' in ver: cls = PaperMinimalCNN2MaxPool
            else: cls = PaperMinimalCNN2Stride
            
            model = cls(nk=nk, np_size=5, nf=5).to(config.DEVICE)
            desc = f"CNN{ver} (nk={nk})"
        else:
            continue

        # Load Weights
        try:
            model.load_state_dict(torch.load(filepath, map_location=config.DEVICE))
            model.eval()
        except Exception as e:
            print(f"Error loading {filename}: {e}")
            continue
            
        # Infer
        all_preds, all_targets = [], []
        with torch.no_grad():
            for u, _, p in loader:
                u = u.to(config.DEVICE)
                u_in = u[:, 0:1] if arch != 'mlp' else u
                preds = model(u_in)
                all_preds.append(preds.cpu().numpy())
                all_targets.append(p.numpy())
                
        # Metric
        preds = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)
        
        # Joint NRMSE
        diff = preds - targets
        rmse = np.sqrt(np.mean(diff**2))
        norm = np.linalg.norm(np.mean(targets, axis=0))
        nrmse = rmse / norm if norm > 0 else 0
        
        if lambda_val != "N/A":
             desc += f" (λ={lambda_val})"
        
        results.append({
            "Architecture": desc,
            "Loss": loss,
            "Lambda": lambda_val,
            "Samples": nsamp,
            "NRMSE": nrmse,
            "File": filename
        })
        print(f"  -> {desc} | {loss} (λ={lambda_val}) | n={nsamp} | NRMSE={nrmse:.2%}")

    # Print Table
    df = pd.DataFrame(results)
    if not df.empty:
        df = df.sort_values(['Samples', 'Architecture', 'Loss'])
        print("\n" + "="*80)
        print("SUMMARY RESULTS")
        print("="*80)
        print(df.to_string(index=False))
        
        # Save CSV
        df.to_csv("comparison_summary.csv", index=False)
        print("\nSaved to comparison_summary.csv")

if __name__ == "__main__":
    main()