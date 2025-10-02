#!/bin/bash

# Script to download required HuggingFace models ahead of time

set -e

MODEL_NAME=${1:-"meta-llama/Llama-2-7b-hf"}
MODEL_DIR=${2:-"./models"}

mkdir -p ${MODEL_DIR}

echo "Downloading model: $MODEL_NAME to $MODEL_DIR"

python - << 'PY'
import sys, os
from huggingface_hub import snapshot_download

model_name = sys.argv[1]
model_dir = sys.argv[2]

snapshot_download(repo_id=model_name, local_dir=model_dir, local_dir_use_symlinks=False)
PY