#!/bin/bash

# ==============================================================================
# run_evaluation_sweep.sh
#
# This script automates the process of evaluating a trained diffusion model
# checkpoint using the DDIM sampler with a varying number of inference steps.
#
# It finds the "best_model.pth" from the latest training run and runs the
# evaluation script for a predefined list of step counts.
#
# Usage:
#   ./run_evaluation_sweep.sh
# ==============================================================================

# --- Configuration ---

# This will automatically find the most recent training run directory.
# `ls -td` lists directories by modification time (newest first), and `head -n 1` gets the first one.
LATEST_RUN_DIR=$(ls -td runs/DiffusionPolicy_Training/*/ | head -n 1)

# Construct the full path to the best model checkpoint.
CHECKPOINT_PATH="${LATEST_RUN_DIR}checkpoints/best_model.pth"

# Define the DDIM step counts you want to evaluate.
# You can easily add or remove values from this list.
DDIM_STEPS=(10 20 30 50)

# --- Pre-flight Checks ---

# Check if the checkpoint file actually exists before starting.
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "❌ Error: Checkpoint file not found at '$CHECKPOINT_PATH'"
    echo "Please ensure that a training run has been completed successfully."
    exit 1
fi

echo "✅ Found latest checkpoint: $CHECKPOINT_PATH"
echo "🚀 Starting evaluation sweep for the following DDIM steps: ${DDIM_STEPS[*]}"
echo "=============================================================================="

# --- Main Evaluation Loop ---

# Loop through each value in the DDIM_STEPS array.
for steps in "${DDIM_STEPS[@]}"; do
    echo ""
    echo "--- Running evaluation for ${steps} DDIM steps ---"
    
    # Construct the command. The `\` at the end of lines is for readability.
    python -m src.evaluation.evaluate_prediction \
        --checkpoint "$CHECKPOINT_PATH" \
        --sampler ddim \
        --steps "$steps"
        
    # Check the exit code of the last command. If it's not 0, an error occurred.
    if [ $? -ne 0 ]; then
        echo "❌ Error: Evaluation failed for ${steps} steps. Aborting sweep."
        exit 1
    fi
    
    echo "--- Completed evaluation for ${steps} steps ---"
done

echo ""
echo "=============================================================================="
echo "✅ Evaluation sweep completed successfully!"
echo "All JSON result files have been saved to the checkpoint directory."
echo "You can now analyze the results in the '3_analyze_final_results.ipynb' notebook."