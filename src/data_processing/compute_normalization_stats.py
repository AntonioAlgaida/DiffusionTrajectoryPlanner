# src/data_processing/compute_normalization_stats.py

import os
import sys
import torch
from glob import glob
from tqdm import tqdm
import numpy as np
import multiprocessing

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config

# Helper function for multiprocessing
def _load_trajectory(path: str) -> np.ndarray | None:
    """Loads just the (x, y) trajectory data from a .pt file."""
    try:
        # Load the sample and immediately slice off the heading
        sample = torch.load(path, map_location='cpu', weights_only=False)
        return sample['target_trajectory'][:, :2] # Return only x, y
    except Exception as e:
        print(f"Warning: Skipping corrupted file {path}. Error: {e}")
        return None

def main():
    print("--- Computing Isotropic (x,y) Normalization Statistics ---")
    config = load_config(os.path.join(PROJECT_ROOT, 'configs/main_config.yaml'))
    featurized_dir = config['data']['featurized_dir']
    
    train_files = glob(os.path.join(featurized_dir, 'training', '*.pt'))
    if not train_files:
        raise FileNotFoundError("No training files found. Please run the featurizer first.")

    # --- Use multiprocessing to load all trajectories in parallel ---
    print(f"Loading {len(train_files)} trajectories for statistics calculation...")
    num_workers = os.cpu_count()
    with multiprocessing.Pool(processes=num_workers) as pool:
        results_iterator = pool.imap_unordered(_load_trajectory, train_files)
        pbar = tqdm(results_iterator, total=len(train_files), desc="Loading trajectories in parallel")
        all_trajectories = [traj for traj in pbar if traj is not None]

    if not all_trajectories:
        raise ValueError("Could not load any valid trajectories.")
        
    # --- Isotropic Scaling Logic ---
    # Concatenate all (80, 2) trajectories into a single large (N * 80, 2) array
    all_waypoints = np.concatenate(all_trajectories, axis=0)
    
    # To find a single min/max for both x and y, we find the min/max of the entire flattened array
    # This ensures that the scaling is uniform and preserves the aspect ratio.
    min_val = np.min(all_waypoints)
    max_val = np.max(all_waypoints)
    
    # Store the single min and max values.
    stats = {
        'min': torch.tensor(min_val, dtype=torch.float32),
        'max': torch.tensor(max_val, dtype=torch.float32)
    }
    
    # Save the stats
    output_path = os.path.join(PROJECT_ROOT, 'models', 'normalization_stats.pt')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(stats, output_path)
    
    print("\n--- Isotropic Normalization Stats ---")
    print(f"Global Min (for both x and y): {stats['min']:.4f}")
    print(f"Global Max (for both x and y): {stats['max']:.4f}")
    print(f"Statistics saved to: {output_path}")

if __name__ == '__main__':
    main()