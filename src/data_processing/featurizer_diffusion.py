# src/data_processing/featurizer_diffusion.py

"""
Main featurizer script for the Diffusion Policy project.

This script orchestrates the conversion of raw, parsed .npz scenario files into
a structured, ML-ready dataset specifically for training a conditional trajectory
diffusion model.

The core logic is as follows:
1. It identifies all source .npz files from a specified directory.
2. For each scenario, it isolates a single, critical moment: the state at
   the 1-second mark (timestep_index=10).
3. It extracts a rich, structured 'state_dict' representing the world context
   at that moment using the FeatureExtractor.
4. It extracts the SDC's ground-truth trajectory for the subsequent 8 seconds
   (timesteps 11-90) and transforms it into the same ego-centric frame.
5. It saves each (Context, Target) pair as a single dictionary in a .pt file,
   creating a new, versioned dataset ready for training.

This process is parallelized across multiple CPU cores for efficiency.

To run:
conda activate virtuoso_env
python -m src.data_processing.featurizer_diffusion
"""

import os
import sys
import numpy as np
import torch
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import shutil
import traceback

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config
# Assuming FeatureExtractor and geometry are in the utils directory for this stage
from src.data_processing.feature_extractor import FeatureExtractor
from src.utils import geometry

# --- Global Vars for Worker Processes ---
CONFIG = None
FEATURE_EXTRACTOR = None

def init_worker(config_path: str):
    """Initializer for each worker process."""
    global CONFIG, FEATURE_EXTRACTOR
    print(f"Worker process {os.getpid()} initializing...")
    CONFIG = load_config(config_path)
    FEATURE_EXTRACTOR = FeatureExtractor(CONFIG)

def process_shard(npz_path: str) -> bool:
    """
    Processes one .npz file to generate a single (Context, Target) training sample.

    Args:
        npz_path: The full path to the source .npz scenario file.

    Returns:
        True if processing was successful, False otherwise.
    """
    try:
        data = np.load(npz_path, allow_pickle=True)
        scenario_id = data['scenario_id'].item()
        sdc_idx = data['sdc_track_index'].item()
        sdc_route = data['sdc_route']
        
        # --- NEW: Static Scenario Filter ---
        # We check the displacement over our full window of interest (t=1 to t=90)
        # to distinguish "waiting" from truly "parked" scenarios.
        start_timestep_filter = 1
        end_timestep_filter = 90
        
        # Ensure the scenario is long enough for this check
        if sdc_route.shape[0] > end_timestep_filter:
            start_pos = sdc_route[start_timestep_filter, :2] # xy position
            end_pos = sdc_route[end_timestep_filter, :2]
            total_displacement = np.linalg.norm(end_pos - start_pos)
            
            min_disp_thresh = CONFIG['features'].get('min_displacement_threshold', 1.0)
            if total_displacement < min_disp_thresh:
                # This scenario is uninformative (e.g., car is parked). Skip it.
                return False

        # --- NEW: SDC Data Quality Filter ---
        max_invalid_steps = CONFIG['features'].get('max_consecutive_invalid_steps', 4)
        sdc_valid_mask = data['valid_mask'][sdc_idx, :]

        longest_invalid_streak = 0
        current_invalid_streak = 0
        for valid in sdc_valid_mask[start_timestep_filter:end_timestep_filter+1]:
            if not valid:
                current_invalid_streak += 1
            else:
                longest_invalid_streak = max(longest_invalid_streak, current_invalid_streak)
                current_invalid_streak = 0
        longest_invalid_streak = max(longest_invalid_streak, current_invalid_streak) # Final check

        if longest_invalid_streak > max_invalid_steps:
            # The SDC track is too unreliable in this window. Skip it.
            return False

        # --- 1. The Anchor Point & Critical Filter ---
        # (The rest of the function continues from here as before)
        anchor_timestep_idx = 10
        
        # If the SDC is not valid at our anchor point, the scenario is useless for this task.
        if not data['valid_mask'][sdc_idx, anchor_timestep_idx]:
            # This is an expected condition, not an error.
            # print(f"Info: Skipping scenario {scenario_id} - SDC not valid at t=10.")
            return False

        # --- 2. Build the "Context": The state_dict at t=10 ---
        # This single call performs all complex feature extraction and transformations.
        state_dict = FEATURE_EXTRACTOR.extract_features(data, anchor_timestep_idx)

        # --- 3. Build the "Target": The Future Trajectory from t=11 to t=90 ---        
        # Check if the trajectory is long enough for an 8-second future.
        if sdc_route.shape[0] < anchor_timestep_idx + 81: # 10 (current) + 80 (future) + 1 (end)
             # print(f"Info: Skipping scenario {scenario_id} - trajectory too short.")
            return False

        # Get the global pose at the anchor point to define our reference frame.
        # Pose is [global_x, global_y, global_yaw]
        ego_pose_t10 = sdc_route[anchor_timestep_idx, [0, 1, 6]]

        # Slice the future 80 steps (8 seconds) in the global frame.
        future_global_trajectory_full = sdc_route[anchor_timestep_idx + 1 : anchor_timestep_idx + 81]
        
        # Transform future positions to the ego-centric frame of t=10.
        future_positions_ego = geometry.transform_points(
            future_global_trajectory_full[:, :2], ego_pose_t10
        )
        
        # Transform future headings to be relative to the heading at t=10.
        # We must wrap the result to the [-pi, pi] interval.
        future_headings_global = future_global_trajectory_full[:, 6]
        future_headings_relative = (future_headings_global - ego_pose_t10[2] + np.pi) % (2 * np.pi) - np.pi

        # Assemble the final target tensor: [x, y, heading]
        # target_trajectory = np.hstack([
        #     future_positions_ego,
        #     future_headings_relative[:, np.newaxis]
        # ]).astype(np.float32)
        
        # --- NEW: Only XY for Diffusion Model ---
        target_trajectory = future_positions_ego.astype(np.float32)

        if target_trajectory.shape != (80, 2):
            # This is a sanity check that should ideally never fail.
            print(f"Warning: Scenario {scenario_id} produced target with wrong shape {target_trajectory.shape}.")
            return False
            
        # --- 4. Final Assembly and Saving ---
        final_sample = {
            'state': state_dict,
            'target_trajectory': target_trajectory
        }

        output_subdir = os.path.basename(os.path.dirname(npz_path)) # 'training' or 'validation'
        output_dir = os.path.join(CONFIG['data']['featurized_v3_final_waypoint'], output_subdir)
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{scenario_id}.pt")
        torch.save(final_sample, output_path)
        
        return True

    except Exception as e:
        print(f"❌ Error processing {npz_path}: {e}\n{traceback.format_exc()}")
        return False

def main():
    """
    Main orchestrator for the diffusion featurization pipeline.
    """
    config_path = os.path.join(PROJECT_ROOT, 'configs/main_config.yaml')
    global CONFIG
    CONFIG = load_config(config_path)

    output_dir = CONFIG['data']['featurized_v3_final_waypoint']
    npz_base_dir = CONFIG['data']['processed_npz_dir']
    all_npz_paths = glob(os.path.join(npz_base_dir, '*', '*.npz'))

    if not all_npz_paths:
        print(f"❌ Error: No source .npz files found in '{npz_base_dir}'. Please run the parser first.")
        return

    paths_to_process = all_npz_paths

    if os.path.exists(output_dir):
        print(f"\nOutput directory '{output_dir}' already exists.")
        response = input("Choose an action: [d]elete, [c]ontinue (skip existing), or [a]bort? [d/c/a]: ").lower()
        
        if response == 'd':
            print("Deleting existing output directory...")
            shutil.rmtree(output_dir)
            print("Directory deleted.")
        elif response == 'c':
            print("Continuing. Will skip files that have already been processed.")
            existing_ids = {
                os.path.splitext(f)[0] for subdir in ['training', 'validation']
                for f in os.listdir(os.path.join(output_dir, subdir)) if f.endswith('.pt')
            }
            paths_to_process = [
                p for p in all_npz_paths
                if os.path.splitext(os.path.basename(p))[0] not in existing_ids
            ]
            print(f"Found {len(existing_ids)} already featurized scenarios. Skipping them.")
        else:
            print("Aborting.")
            return
    
    if not paths_to_process:
        print("\nAll scenarios have already been featurized. Nothing to do.")
        return

    print(f"\nFound {len(paths_to_process)} new scenarios to featurize (out of {len(all_npz_paths)} total).")
    
    num_workers = cpu_count()
    print(f"Using {num_workers} worker processes.")

    with Pool(processes=num_workers, initializer=init_worker, initargs=(config_path,)) as pool:
        results = list(tqdm(pool.imap_unordered(process_shard, paths_to_process), total=len(paths_to_process)))

    success_count = sum(results)
    print("\n" + "="*50)
    print("Featurization complete!")
    print(f"Successfully processed and saved: {success_count} scenarios.")
    print(f"Failed or skipped during this run: {len(results) - success_count}")
    print(f"Output saved to: {output_dir}")
    print("="*50)

if __name__ == '__main__':
    main()