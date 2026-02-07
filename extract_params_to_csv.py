import numpy as np
import pandas as pd
import os

INPUT_NPZ = "turing_patterns_dataset_merged.npz"
OUTPUT_CSV = "qualified_turing_params_20000_new.csv"

def extract_params():
    if not os.path.exists(INPUT_NPZ):
        print(f"Error: {INPUT_NPZ} not found.")
        return

    print(f"Loading {INPUT_NPZ}...")
    data = np.load(INPUT_NPZ)
    
    # Check keys
    required_keys = ['ids', 'a', 'b', 'c', 'delta']
    for k in required_keys:
        if k not in data:
            print(f"Error: Missing key '{k}' in dataset.")
            return

    # Extract
    ids = data['ids']
    a = data['a']
    b = data['b']
    c = data['c']
    delta = data['delta']

    print(f"Extracting {len(ids)} samples...")

    # Create DataFrame
    df = pd.DataFrame({
        'id': ids,
        'a': a,
        'b': b,
        'c': c,
        'delta': delta
    })

    # Ensure id is int (if not already)
    df['id'] = df['id'].astype(int)

    # Save to CSV using high precision formatting to capture full float32 value
    print(f"Saving to {OUTPUT_CSV} with high precision...")
    df.to_csv(OUTPUT_CSV, index=False, float_format='%.17g')
    print("Done.")

if __name__ == "__main__":
    extract_params()
