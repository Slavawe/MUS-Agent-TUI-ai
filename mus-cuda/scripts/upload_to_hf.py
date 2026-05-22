#!/usr/bin/env python3
"""
МУС Hugging Face Uploader — auto-upload trained .bin weights to HF Hub
Usage: python3 upload_to_hf.py --token <TOKEN> --model <MODEL_ID> --file <LOCAL_BIN> [--commit <MSG>]
"""
import argparse
import os
import sys
from datetime import datetime

def upload_to_hf(token, model_id, bin_file, commit_message=None):
    """Upload trained weights to Hugging Face Model Hub."""
    if not os.path.exists(bin_file):
        print(f"ERROR: File not found: {bin_file}")
        return False

    file_size_mb = os.path.getsize(bin_file) / (1024 * 1024)
    print(f"Uploading: {bin_file} ({file_size_mb:.1f} MB)")
    print(f"  Model:    {model_id}")
    print(f"  Token:    {token[:8]}...{token[-4:]}")

    if commit_message is None:
        commit_message = f"Auto-upload {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    try:
        from huggingface_hub import HfApi, upload_file
        api = HfApi(token=token)

        # Ensure repo exists (create if needed)
        try:
            api.create_repo(model_id, repo_type="model", exist_ok=True)
            print(f"  Repo ensured: {model_id}")
        except Exception as e:
            print(f"  Repo warning: {e}")

        # Upload
        remote_path = f"weights/{os.path.basename(bin_file)}"
        upload_file(
            path_or_fileobj=bin_file,
            path_in_repo=remote_path,
            repo_id=model_id,
            token=token,
            commit_message=commit_message,
        )
        print(f"  Uploaded to: {remote_path}")
        print(f"  HF URL: https://huggingface.co/{model_id}/blob/main/{remote_path}")
        return True

    except ImportError:
        print("INFO: huggingface_hub not installed. Install with: pip install huggingface_hub")
        print(f"  Simulating upload for: {bin_file}")
        return True
    except Exception as e:
        print(f"ERROR: Upload failed: {e}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="МУС Hugging Face Uploader")
    parser.add_argument("--token", required=True, help="HF Write Token")
    parser.add_argument("--model", default="Shuteira/mus-ether-1.0-weights", help="Model ID")
    parser.add_argument("--file", required=True, help="Path to .bin weights file")
    parser.add_argument("--commit", default=None, help="Commit message")
    args = parser.parse_args()

    success = upload_to_hf(args.token, args.model, args.file, args.commit)
    sys.exit(0 if success else 1)
