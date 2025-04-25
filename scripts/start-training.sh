#!/bin/bash

# Setup script for energy-efficient distributed training

set -e  # Exit immediately if a command exits with a non-zero status

# Initialize environment
echo "Initializing training environment..."

# Check NVIDIA devices and MPS status
nvidia-smi || echo "Warning: nvidia-smi not available"

# Set CUDA_VISIBLE_DEVICES if empty
if [[ -z "$CUDA_VISIBLE_DEVICES" ]]; then
    export CUDA_VISIBLE_DEVICES="0"
    echo "CUDA_VISIBLE_DEVICES was empty, setting to: $CUDA_VISIBLE_DEVICES"
else
    echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
fi

echo "MPS Thread Percentage: $CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"

# Set up distributed training
echo "Setting up distributed training environment..."
echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "WORLD_SIZE: $WORLD_SIZE"
echo "Raw RANK value: $RANK"

# Extract rank from pod name (llm-energy-efficient-training-X)
if [[ -z "$RANK" ]]; then
    echo "RANK is empty, defaulting to 0"
    RANK_INT=0
elif [[ "$RANK" == "llm-energy-efficient-training-0" ]]; then
    echo "Master node detected"
    RANK_INT=0
elif [[ "$RANK" == *"-"* ]]; then
    # Extract the last digit from the pod name
    RANK_INT=${RANK##*-}
    echo "Extracted RANK $RANK_INT from pod name suffix"
else
    echo "Could not parse RANK from $RANK, defaulting to 0"
    RANK_INT=0
fi

# Export the numeric RANK for clarity
export RANK=$RANK_INT
echo "Using RANK: $RANK_INT"

# Check for required directories and files
mkdir -p /app/output /app/energy_stats 2>/dev/null
echo "Checking for model-output PVC..."
if [ ! -w "/app/output" ]; then
    echo "ERROR: Cannot write to /app/output directory. Check if PVC is mounted properly."
    # Instead of exiting, we'll keep the container running for debugging
    sleep 3600
fi

# Create the DeepSpeed config.json from training_config.yaml in /tmp (writable) on ALL nodes
echo "Creating DeepSpeed config JSON in /tmp (writable directory)..."
python -c "
import json
import yaml
import os

try:
    with open('/app/configs/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Make sure the adaptive_training section has all required fields
    if 'deepspeed' in config:
        if 'adaptive_training' not in config['deepspeed']:
            config['deepspeed']['adaptive_training'] = {}
        
        # Ensure all required fields exist with defaults
        adaptive_training = config['deepspeed']['adaptive_training']
        if 'min_thread_percentage' not in adaptive_training:
            adaptive_training['min_thread_percentage'] = 30
        if 'max_thread_percentage' not in adaptive_training:
            adaptive_training['max_thread_percentage'] = 80
    
    # Write the config file
    with open('/tmp/ds_config.json', 'w') as f:
        json.dump(config.get('deepspeed', {}), f, indent=2)
    
    print('Created DeepSpeed config file in /tmp/ds_config.json')
except Exception as e:
    print(f'Error creating DeepSpeed config: {e}')
    # Write a minimal valid JSON to prevent parsing errors
    with open('/tmp/ds_config.json', 'w') as f:
        json.dump({
            'adaptive_training': {
                'enabled': True,
                'energy_threshold_high': 250,
                'energy_threshold_low': 150,
                'min_thread_percentage': 30,
                'max_thread_percentage': 80
            }
        }, f)
    print('Created minimal fallback DeepSpeed config')
" || echo "Failed to create DeepSpeed config"

# Setup energy efficiency settings
if [ "$RANK_INT" -eq 0 ]; then
    # Master node handles additional monitoring
    echo "This is the master node (RANK 0). Starting energy monitoring..."
    # Set up MPS directories if needed
    mkdir -p /tmp/nvidia-mps /tmp/nvidia-log 2>/dev/null
    
    # Initialize MPS thread percentage file for each GPU
    nvidia-smi --query-gpu=index --format=csv,noheader | while read GPU_IDX; do
        echo "$CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" > "/tmp/nvidia-mps/gpu${GPU_IDX}_thread_percentage"
        chmod 666 "/tmp/nvidia-mps/gpu${GPU_IDX}_thread_percentage" 2>/dev/null
    done
    
    # Debug - list directories and files
    echo "Checking directories:"
    ls -la /app/configs/
    echo "Config file content:"
    cat /app/configs/training_config.yaml | head -n 20
    
    # Begin training with error handling
    echo "Starting training process with energy monitoring..."
    echo "Running: python /app/src/train.py --config /app/configs/training_config.yaml --deepspeed_config /tmp/ds_config.json"
    
    # Touch a file to indicate master is ready
    touch /tmp/master_ready
    chmod 666 /tmp/master_ready
    
    # Run in background with error handling
    (python /app/src/train.py --config /app/configs/training_config.yaml --deepspeed_config /tmp/ds_config.json 2>&1) &
    TRAIN_PID=$!
    
    # Monitor the training process
    echo "Training process started with PID: $TRAIN_PID"
    
    # Keep the container alive 
    wait $TRAIN_PID || {
        echo "Training process exited with error: $?"
        echo "Keeping container alive for debugging..."
        sleep 3600
    }
    
else
    # Worker nodes wait for master node to initialize
    echo "This is a worker node (RANK $RANK). Waiting for master node to initialize..."
    
    # Wait longer - up to 2 minutes for master to be ready
    MAX_WAIT=120
    WAIT_COUNT=0
    while [ ! -f /tmp/master_ready ] && [ $WAIT_COUNT -lt $MAX_WAIT ]; do
        echo "Waiting for master node to initialize... ($WAIT_COUNT/$MAX_WAIT seconds)"
        sleep 5
        WAIT_COUNT=$((WAIT_COUNT + 5))
    done
    
    if [ ! -f /tmp/master_ready ]; then
        echo "WARNING: Master node readiness signal not found after waiting $MAX_WAIT seconds."
        echo "Continuing anyway, but training may fail if master is not ready."
    else
        echo "Master node is ready, continuing with worker initialization"
    fi
    
    # Begin training with error handling
    echo "Starting training process..."
    
    # Run in background with error handling
    (python /app/src/train.py --config /app/configs/training_config.yaml --deepspeed_config /tmp/ds_config.json 2>&1) &
    TRAIN_PID=$!
    
    # Monitor the training process
    echo "Training process started with PID: $TRAIN_PID"
    
    # Keep the container alive
    wait $TRAIN_PID || {
        echo "Training process exited with error: $?"
        echo "Keeping container alive for debugging..."
        sleep 3600
    }
fi