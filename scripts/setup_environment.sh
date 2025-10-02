#!/bin/bash

# Setup script for Real-time Prefill KV Cache Compression

set -e

echo "Setting up Real-time Prefill KV Cache Compression environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Create necessary directories
echo "Creating directories..."
mkdir -p data/longbench
mkdir -p experiments/results
mkdir -p logs

# Download LongBench dataset (optional)
echo "LongBench dataset will be downloaded automatically during evaluation."
echo "If you want to pre-download it, run:"
echo "  python -c "from datasets import load_dataset; load_dataset('THUDM/LongBench', cache_dir='./data')""

echo ""
echo "Setup completed successfully!"
echo ""
echo "To run experiments:"
echo "  source venv/bin/activate"
echo "  ./scripts/run_longbench.sh"
echo ""