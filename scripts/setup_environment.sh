#!/bin/bash

# Setup script for Real-time Prefill KV Cache Compression

set -e

echo "Setting up Real-time Prefill KV Cache Compression environment..."

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3.9 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
echo "Installing requirements..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt


# Create necessary directories
echo "Creating directories..."
mkdir -p data/longbench
mkdir -p experiments/results
mkdir -p logs

echo "To run experiments:"
echo "  source venv/bin/activate"
echo "  ./scripts/run_longbench.sh"
echo ""