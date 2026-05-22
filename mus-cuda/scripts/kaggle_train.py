#!/usr/bin/env python3
"""
МУС Cloud Pipeline — Kaggle Training Runner
Автономный конвейер: компиляция → тренировка → выгрузка на Hugging Face
"""
import os, sys, subprocess, json, time, shutil
from pathlib import Path

CONFIG = {
    "hf_token": os.environ.get("HF_TOKEN", ""),
    "hf_repo": "Shuteira/mus-ether-1.0-weights",
    "model_size": os.environ.get("MUS_MODEL", "700m"),  # 400m | 700m | 1b
    "epochs": int(os.environ.get("MUS_EPOCHS", "10")),
}

def log(msg):
    print(f"[Uragan] {msg}", flush=True)

def step(name):
    log(f"═══ {name} ═══")

def check_cuda():
    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    log(f"GPU:\n{result.stdout[:500]}")
    return "Tesla" in result.stdout or "T4" in result.stdout or "P100" in result.stdout

def build():
    step("Compiling MUS for cloud...")
    build_dir = Path("/kaggle/working/build")
    build_dir.mkdir(exist_ok=True)
    
    result = subprocess.run([
        "cmake", "/kaggle/input/mus-source",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DMUS_BUILD_CLOUD=ON",
        "-DMUS_BUILD_MINIMAL=ON",
    ], cwd=build_dir, capture_output=True, text=True)
    log(result.stdout[-500:])
    
    result = subprocess.run(["make", "-j$(nproc)", "mus_train_kaggle"],
                          cwd=build_dir, capture_output=True, text=True)
    log(result.stdout[-500:])
    
    binary = build_dir / "mus_train_kaggle"
    if not binary.exists():
        log("BUILD FAILED!")
        return None
    log(f"Built: {binary}")
    return binary

def train(binary):
    step(f"Training {CONFIG['model_size']} for {CONFIG['epochs']} epochs...")
    
    result = subprocess.run([
        str(binary), CONFIG["model_size"],
        "--epochs", str(CONFIG["epochs"]),
    ], capture_output=True, text=True, timeout=72000)
    
    log(f"Training output:\n{result.stdout[-2000:]}")
    
    # Find the .bin weights
    weights_dir = Path("/kaggle/working/weights")
    bins = list(weights_dir.glob("*.bin"))
    if bins:
        bins.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return bins[0]
    return None

def upload_to_hf(weights_path):
    step("Uploading weights to Hugging Face...")
    
    if not CONFIG["hf_token"]:
        log("No HF_TOKEN set, skipping upload")
        return False
    
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=CONFIG["hf_token"])
        
        # Upload
        model_name = f"mus-{CONFIG['model_size']}-{int(time.time())}.bin"
        api.upload_file(
            path_or_fileobj=str(weights_path),
            path_in_repo=f"weights/{model_name}",
            repo_id=CONFIG["hf_repo"],
            repo_type="model",
        )
        log(f"Uploaded: {model_name}")
        
        # Update metadata
        api.upload_file(
            path_or_fileobj=json.dumps({
                "model": CONFIG["model_size"],
                "epochs": CONFIG["epochs"],
                "timestamp": time.time(),
                "framework": "C++17/CUDA",
            }, indent=2).encode(),
            path_in_repo="metadata/latest.json",
            repo_id=CONFIG["hf_repo"],
            repo_type="model",
        )
        return True
    except Exception as e:
        log(f"Upload failed: {e}")
        return False

def main():
    log("МУС Cloud Pipeline started")
    
    if not check_cuda():
        log("No CUDA GPU found!")
        return 1
    
    binary = build()
    if not binary:
        return 1
    
    weights = train(binary)
    if not weights:
        log("No weights produced!")
        return 1
    
    upload_to_hf(weights)
    log("МУС Pipeline completed successfully!")
    return 0

if __name__ == "__main__":
    sys.exit(main())