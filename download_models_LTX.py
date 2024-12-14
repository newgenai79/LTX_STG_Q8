import os
import requests
from tqdm import tqdm

# Repository details
repo_url = "https://huggingface.co/Lightricks/LTX-Video/resolve/main/"
files_to_download = [
    "model_index.json",
    "scheduler/noise_scheduler_config.json",
    "scheduler/scheduler_config.json",
    "tokenizer/added_tokens.json",
    "tokenizer/special_tokens_map.json",
    "tokenizer/spiece.model",
    "tokenizer/tokenizer_config.json",
    "transformer/config.json",
    "transformer/diffusion_pytorch_model-00001-of-00002.safetensors",
    "transformer/diffusion_pytorch_model-00002-of-00002.safetensors",
    "transformer/diffusion_pytorch_model.safetensors.index.json",
    "transformer/transformer_config.json",
    "vae/config.json",
    "vae/diffusion_pytorch_model.safetensors",
    "vae/vae_config.json"
]

# Local directory to save files
local_dir = "./Lightricks/LTX-Video"

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
