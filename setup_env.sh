#!/usr/bin/env bash
# One-time environment setup for local fine-tuning of Llama-3.2-3B-Instruct
# Run this in a fresh terminal (PowerShell/WSL/Linux/macOS with an NVIDIA GPU + recent driver).
set -e

echo "== 1. Create a virtual environment =="
python3 -m venv unsloth-env
source unsloth-env/bin/activate   # on Windows (WSL not used): unsloth-env\Scripts\activate

echo "== 2. Upgrade pip =="
pip install --upgrade pip

echo "== 3. Install PyTorch with CUDA support =="
# Check your CUDA version first with: nvidia-smi
# This installs the CUDA 12.1 build; adjust the index-url if your driver needs a different version
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo "== 4. Install Unsloth and dependencies =="
pip install "unsloth[cu121-torch240] @ git+https://github.com/unslothai/unsloth.git"
pip install --no-deps trl peft accelerate bitsandbytes

echo "== 5. Install llama.cpp conversion tools (for GGUF export) =="
pip install gguf sentencepiece protobuf

echo "Done. Activate the environment in future sessions with:"
echo "  source unsloth-env/bin/activate"
