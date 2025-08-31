#!/bin/bash

# ==============================================================================
# run_training_resilient.sh
#
# A robust "watchdog" script to run and automatically resume training.
# If the training script crashes, this harness will wait a few seconds and
# then re-launch it, automatically loading the latest best checkpoint.
#
# Usage:
#   ./run_training_resilient.sh
# ==============================================================================

# Keep track of the last run directory to ensure we always resume from the same run
RUN_DIR=""

while true; do
    CHECKPOINT_ARG=""
    
    # If we haven't identified the run directory yet, find the latest one.
    if [ -z "$RUN_DIR" ]; then
        LATEST_DIR=$(ls -td runs/DiffusionPolicy_Training/*/ | head -n 1)
        # Check if a directory was found
        if [ -n "$LATEST_DIR" ]; then
            RUN_DIR="$LATEST_DIR"
        fi
    fi
    
    # If a run directory has been established, look for a checkpoint inside it.
    if [ -n "$RUN_DIR" ]; then
        CHECKPOINT_PATH="${RUN_DIR}checkpoints/best_model.pth"
        if [ -f "$CHECKPOINT_PATH" ]; then
            echo "✅ Found checkpoint to resume from: $CHECKPOINT_PATH"
            CHECKPOINT_ARG="--resume_from_checkpoint $CHECKPOINT_PATH"
        else
            echo "🏁 Starting a new training run in: $RUN_DIR"
        fi
    else
        echo "🏁 Starting a brand new training run..."
    fi

    echo "🚀 Launching training..."
    
    # Launch the training script with the checkpoint argument (if it exists)
    # The `eval` command is used to correctly handle the case where CHECKPOINT_ARG is empty.
    eval python -m src.diffusion_policy.train $CHECKPOINT_ARG
    
    # Check the exit code of the python script
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Training completed successfully. Exiting harness."
        break # Exit the loop if training finished without errors
    else
        echo "❌ Training crashed with exit code $EXIT_CODE."
        echo "🔄 Relaunching from the last checkpoint in 10 seconds..."
        sleep 10
    fi
done