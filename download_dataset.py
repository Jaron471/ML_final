import gdown
import os
import shutil

def download_dataset():
    url = "https://drive.google.com/drive/folders/18Vc_uFunWvOcYppVVmc9QZz21nX6Zop3?usp=sharing"
    # Use a temporary folder to download the content first
    temp_output = "temp_dataset_download"

    if not os.path.exists(temp_output):
        os.makedirs(temp_output)
        print(f"Created temporary directory: {temp_output}")
    
    print(f"Downloading dataset folder from {url} to {temp_output}...")
    # download_folder downloads the contents of the remote folder into the output directory
    gdown.download_folder(url, output=temp_output, quiet=False, use_cookies=False)
    print("Download complete.")
    
    print("Moving files to root directory...")
    files_moved = 0
    for item in os.listdir(temp_output):
        source = os.path.join(temp_output, item)
        destination = os.path.join(".", item)
        
        # Skip hidden files like .DS_Store if they exist
        if item.startswith('.'):
            continue
            
        print(f"Moving {item} to ./")
        if os.path.exists(destination):
            if os.path.isdir(destination):
                shutil.rmtree(destination)
            else:
                os.remove(destination)
        
        shutil.move(source, destination)
        files_moved += 1

    print(f"Moved {files_moved} items to root directory.")

    # Clean up
    try:
        os.rmdir(temp_output)
        print(f"Removed temporary directory: {temp_output}")
    except OSError:
        print(f"Note: Temporary directory {temp_output} could not be removed (might not be empty).")

if __name__ == "__main__":
    download_dataset()
