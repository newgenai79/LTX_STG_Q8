import os
import requests
from tqdm import tqdm

# Repository details
repo_url = "https://huggingface.co/konakona/ltxvideo_q8/resolve/main/"
files_to_download = [
    "config.json",
    "diffusion_pytorch_model.safetensors"
]

# Local directory to save files
local_dir = "./konakona/ltxvideo_q8"

# Function to download files with progress bar
def download_files(repo_url, files, local_dir):
    for file_path in files:
        try:
            # Construct the full URL for the file
            file_url = repo_url + file_path
            local_file_path = os.path.join(local_dir, file_path)
            
            # Create local directory structure
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Start downloading the file
            print(f"Downloading {file_path}...")
            response = requests.get(file_url, stream=True)
            response.raise_for_status()
            
            # Get the total file size from headers
            total_size = int(response.headers.get('content-length', 0))
            
            # Download with progress bar
            with open(local_file_path, "wb") as file, tqdm(
                desc=file_path,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
                    bar.update(len(chunk))
            print(f"Downloaded: {file_path}")
        except Exception as e:
            print(f"Failed to download {file_path}: {e}")

# Run the download
download_files(repo_url, files_to_download, local_dir)
