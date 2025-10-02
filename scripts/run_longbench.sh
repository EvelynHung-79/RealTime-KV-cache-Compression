#!/bin/bash

# Real-time Prefill KV Cache Compression Experiment Runner
# This script sets up the environment and runs the compression experiment

set -e  # Exit on any error

echo "=========================================="
echo "Real-time Prefill KV Cache Compression"
echo "=========================================="

# Configuration
MODEL_NAME="meta-llama/Llama-2-7b-hf"
DEVICE="cuda"
MAX_LENGTH=4096
OUTPUT_DIR="./experiments/results"

# Default hyperparameters (can be overridden)
ALPHA=${ALPHA:-0.4}
BETA=${BETA:-0.3}
GAMMA=${GAMMA:-0.3}
THETA_H=${THETA_H:-0.7}
THETA_M=${THETA_M:-0.3}

# Evaluation settings
MAX_SAMPLES=${MAX_SAMPLES:-50}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-100}

# Check if GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits
    echo ""
else
    echo "Warning: nvidia-smi not found. Running on CPU may be very slow."
    DEVICE="cpu"
fi

# Check if required directories exist
mkdir -p $OUTPUT_DIR
mkdir -p ./data/longbench

# Function to run experiment with specific configuration
run_experiment() {
    local exp_name=$1
    local additional_args=$2

    echo "Running experiment: $exp_name"
    echo "Additional arguments: $additional_args"

    python experiments/run_compression_experiment.py \
        --model_name $MODEL_NAME \
        --device $DEVICE \
        --max_length $MAX_LENGTH \
        --alpha $ALPHA \
        --beta $BETA \
        --gamma $GAMMA \
        --theta_h $THETA_H \
        --theta_m $THETA_M \
        --max_samples $MAX_SAMPLES \
        --max_new_tokens $MAX_NEW_TOKENS \
        --output_dir $OUTPUT_DIR \
        --experiment_name $exp_name \
        $additional_args

    echo "Experiment $exp_name completed!"
    echo ""
}

# Parse command line arguments
BASELINE=false
ABLATION=false
TASKS=""
EXPERIMENT_NAME=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --baseline)
            BASELINE=true
            shift
            ;;
        --ablation)
            ABLATION=true
            shift
            ;;
        --tasks)
            TASKS="--tasks $2"
            shift 2
            ;;
        --exp_name)
            EXPERIMENT_NAME="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --baseline     Run baseline comparison"
            echo "  --ablation     Run ablation study"
            echo "  --tasks TASKS  Specify tasks (e.g., 'narrativeqa qasper')"
            echo "  --exp_name NAME Experiment name"
            echo "  --help         Show this help"
            echo ""
            echo "Environment variables:"
            echo "  ALPHA, BETA, GAMMA: Importance scoring weights"
            echo "  THETA_H, THETA_M: Precision thresholds"
            echo "  MAX_SAMPLES: Max samples per task"
            echo "  MAX_NEW_TOKENS: Max tokens to generate"
            echo ""
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Set default experiment name if not provided
if [[ -z "$EXPERIMENT_NAME" ]]; then
    EXPERIMENT_NAME="compression_$(date +%Y%m%d_%H%M%S)"
fi

echo "Configuration:"
echo "  Model: $MODEL_NAME"
echo "  Device: $DEVICE"
echo "  Max Length: $MAX_LENGTH"
echo "  Alpha/Beta/Gamma: $ALPHA/$BETA/$GAMMA"
echo "  Theta H/M: $THETA_H/$THETA_M"
echo "  Max Samples: $MAX_SAMPLES"
echo "  Experiment: $EXPERIMENT_NAME"
echo ""

# Prepare additional arguments
ADDITIONAL_ARGS="$TASKS"

if [[ "$BASELINE" == true ]]; then
    ADDITIONAL_ARGS="$ADDITIONAL_ARGS --baseline"
fi

if [[ "$ABLATION" == true ]]; then
    ADDITIONAL_ARGS="$ADDITIONAL_ARGS --ablation"
fi

# Run the main experiment
run_experiment $EXPERIMENT_NAME "$ADDITIONAL_ARGS"

echo "=========================================="
echo "All experiments completed!"
echo "Results saved to: $OUTPUT_DIR/$EXPERIMENT_NAME"
echo "=========================================="

# Optional: Generate summary report
if command -v python &> /dev/null; then
    echo "Generating summary report..."
    python -c "
import json
import os
import sys

results_dir = '$OUTPUT_DIR/$EXPERIMENT_NAME'
summary_file = os.path.join(results_dir, 'experiment_summary.json')

if os.path.exists(summary_file):
    with open(summary_file, 'r') as f:
        data = json.load(f)

    print('\n=== EXPERIMENT SUMMARY ===')

    if 'compressed' in data:
        comp = data['compressed']
        print(f'Quality Score: {comp.get("overall_quality_score", "N/A"):.4f}')

        if 'compression_performance' in comp:
            perf = comp['compression_performance']
            print(f'Memory Savings: {perf.get("overall_avg_memory_savings", 0)*100:.1f}%')
            print(f'Compression Ratio: {perf.get("overall_avg_compression_ratio", 1.0):.3f}')

    if 'baseline' in data:
        baseline = data['baseline']
        print(f'Baseline Score: {baseline.get("overall_quality_score", "N/A"):.4f}')

    print('\nDetailed results available in:', results_dir)
else:
    print('Summary file not found. Check experiment logs for details.')
"
fi