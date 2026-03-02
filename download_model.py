# download_model.py
import argparse
import os
import shutil
from huggingface_hub import snapshot_download

def download_model(model_name, save_path, ignore_patterns=None):
    """
    Downloads a model from Hugging Face using snapshot_download and saves it to a specified local path.
    """
    if os.path.exists(save_path):
        print(f"Directory '{save_path}' already exists. Removing it to ensure a fresh download.")
        shutil.rmtree(save_path)

    print(f"Downloading model '{model_name}' to '{save_path}'...")
    try:
        # Use a temporary directory for the download
        tmp_dir = save_path + "_tmp"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)

        snapshot_download(
            repo_id=model_name,
            local_dir=tmp_dir,
            local_dir_use_symlinks=False, # Important for Docker volumes
            ignore_patterns=ignore_patterns,
        )

        # Move from temp to final destination
        os.rename(tmp_dir, save_path)
        print("Model downloaded and saved successfully.")
    except Exception as e:
        print(f"An error occurred while downloading the model: {e}")
        # Clean up the temp directory if the download failed
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        # Also clean up the target directory if it was partially created
        if os.path.exists(save_path):
            shutil.rmtree(save_path)
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a Hugging Face model for local use.")
    parser.add_argument("--model_name", type=str, required=True, help="The Hugging Face model identifier (e.g., 'unsloth/mistral-7b-v0.1').")
    parser.add_argument("--save_path", type=str, required=True, help="The local directory path to save the model.")
    parser.add_argument("--ignore_patterns", type=str, nargs='*', help="Glob patterns of files to ignore during download (e.g., '*.safetensors', '*.gguf').")

    args = parser.parse_args()
    download_model(args.model_name, args.save_path, ignore_patterns=args.ignore_patterns)

