# ---
# jupyter:
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Uragan 1.0 — C++ CUDA Training
# ## Google Colab Auto-Trainer (Backup)
# Run this notebook to train Uragan on Colab GPUs (T4, V100, L4)
# with Google Drive caching.

# %% [markdown]
# ### 1. Mount Google Drive

# %%
from google.colab import drive
drive.mount('/content/drive')
print("Google Drive mounted")

# %% [markdown]
# ### 2. Set Token and Config

# %%
import os
HF_TOKEN = "your_hf_write_token_here"  # ← SET YOUR TOKEN
CONFIG = "auto"  # "auto", "400m", "700m"
print(f"HF_TOKEN set: {HF_TOKEN[:8]}..." if HF_TOKEN != "your_hf_write_token_here" else "SET YOUR TOKEN!")

# %% [markdown]
# ### 3. Run Pipeline

# %%
import subprocess
cmd = f"bash /content/mus-cuda/scripts/colab_pipeline.sh {HF_TOKEN}"
print(f"Running: {cmd}")
result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
print(f"Exit code: {result.returncode}")

# %% [markdown]
# ### 4. Verify Upload

# %%
if HF_TOKEN != "your_hf_write_token_here":
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    files = api.list_repo_files("Shuteira/mus-ether-1.0-weights", repo_type="model")
    print("Files on Hugging Face:")
    for f in files:
        print(f"  https://huggingface.co/Shuteira/uragan-1.0-weights/blob/main/{f}")
