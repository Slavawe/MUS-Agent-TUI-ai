# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Uragan 1.0 — C++ CUDA Training
# ## Kaggle Auto-Trainer
# Run this notebook to train the Uragan model on Kaggle's Tesla T4/P100 GPUs
# and auto-upload weights to Hugging Face.

# %% [markdown]
# ### Setup

# %%
import os
import subprocess
import sys

HF_TOKEN = "your_hf_write_token_here"  # ← SET YOUR TOKEN
CONFIG = "auto"  # "auto", "400m", "700m"
print(f"HF_TOKEN: {HF_TOKEN[:8]}..." if HF_TOKEN != "your_hf_write_token_here" else "WARNING: Set HF_TOKEN!")

# %% [markdown]
# ### Run Pipeline

# %%
cmd = f"bash /kaggle/working/mus-cuda/scripts/kaggle_pipeline.sh {HF_TOKEN}"
if CONFIG != "auto":
    cmd += f" {CONFIG}"

print(f"Running: {cmd}")
result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
print(f"Exit code: {result.returncode}")

# %% [markdown]
# ### Verify Upload

# %%
if HF_TOKEN != "your_hf_write_token_here":
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    files = api.list_repo_files("Shuteira/uragan-1.0-weights", repo_type="model")
    print("Files on HF:")
    for f in files:
        print(f"  {f}")
