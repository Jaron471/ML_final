import os
import pandas as pd
import re

def rename_files():
    checkpoint_dir = 'paper_checkpoints'
    csv_file = 'flexible_experiments_results.csv'
    
    if not os.path.exists(checkpoint_dir):
        print(f"Directory {checkpoint_dir} does not exist.")
        return

    # Load CSV to update it later
    try:
        df = pd.read_csv(csv_file)
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return

    files = os.listdir(checkpoint_dir)
    pattern = re.compile(r"^(.*_n\d+)_(\d+)(_best\.pth)$")
    
    renames = {} # old_name -> new_name
    conflicts = []
    
    # Analyze renames
    processed_configs = set()
    
    for filename in files:
        match = pattern.match(filename)
        if match:
            prefix = match.group(1)
            # current_id = match.group(2) # unused
            suffix = match.group(3)
            
            new_name = f"{prefix}_1{suffix}"
            
            if filename == new_name:
                continue # Already correct
                
            if new_name in renames.values():
                conflicts.append(new_name)
            
            # Check if target file already exists on disk (and isn't the source)
            if os.path.exists(os.path.join(checkpoint_dir, new_name)) and new_name != filename:
                 # Be careful: if A_1 exists and we want to rename A_2 to A_1, we have a conflict/overwrite issue.
                 # BUT, we assumed we deleted all unlisted files.
                 # If A_1 exists, it matches the CSV?
                 # If A_2 exists, it matches the CSV?
                 # If both exist, then both are in CSV (implied valid).
                 # If both are in CSV, we have two entries for same config.
                 conflicts.append(new_name)

            renames[filename] = new_name
    
    if conflicts:
        print("ERROR: Renaming would cause conflicts/overwrites. The following target filenames would start colliding:")
        for c in set(conflicts):
            print(c)
        return

    print(f"Proposed renames: {len(renames)}")
    
    # Perform renames on disk
    count = 0
    file_map = {} # old -> new (for CSV update)
    
    for old_name, new_name in renames.items():
        old_path = os.path.join(checkpoint_dir, old_name)
        new_path = os.path.join(checkpoint_dir, new_name)
        
        try:
            os.rename(old_path, new_path)
            file_map[old_name] = new_name
            count += 1
            # print(f"Renamed: {old_name} -> {new_name}")
        except Exception as e:
            print(f"Failed to rename {old_name}: {e}")

    print(f"Successfully renamed {count} files.")
    
    # Update CSV
    if count > 0:
        print("Updating CSV file...")
        updated_rows = 0
        for i, row in df.iterrows():
            current_file = row['ModelFile']
            if current_file in file_map:
                df.at[i, 'ModelFile'] = file_map[current_file]
                updated_rows += 1
        
        df.to_csv(csv_file, index=False)
        print(f"CSV updated. Modified {updated_rows} rows.")

if __name__ == "__main__":
    rename_files()
