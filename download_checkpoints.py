import gdown
import os

def download_paper_checkpoints():
    url = "https://drive.google.com/drive/folders/1gkxT9i4RxdbJRtpnhmpY8-Dpk0BjBcM5?usp=sharing"
    output = "paper_checkpoints"

    if not os.path.exists(output):
        os.makedirs(output)
        print(f"Created directory: {output}")
    
    print(f"Downloading folder from {url} to {output}...")
    gdown.download_folder(url, output=output, quiet=False, use_cookies=False)
    print("Download complete.")

if __name__ == "__main__":
    download_paper_checkpoints()
