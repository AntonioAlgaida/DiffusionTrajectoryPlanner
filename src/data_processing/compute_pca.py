# src/data_processing/compute_pca.py

"""
Computes and saves a Principal Component Analysis (PCA) model from the
full set of training trajectories.

This script is a one-time pre-processing step for the "Efficient Virtuoso"
(latent diffusion) project. It learns a low-dimensional subspace that
captures the most significant variance in trajectory shapes. The resulting
fitted PCA model is saved and used later to transform trajectories into
and out of this latent space.
"""

import os
import sys
import torch
from glob import glob
from tqdm import tqdm
import numpy as np
import multiprocessing
import pickle
from sklearn.decomposition import PCA

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config

# ==============================================================================
# === Helper Functions =========================================================
# ==============================================================================

# We need the normalization logic to be consistent with the training pipeline.
# It's crucial to normalize the data BEFORE fitting PCA.
def normalize_trajectory(trajectory, stats):
    """Normalizes an (x, y) trajectory to the [-1, 1] range using isotropic stats."""
    return 2 * (trajectory - stats['min']) / (stats['max'] - stats['min']) - 1

def _load_and_normalize_trajectory(args):
    """
    Helper function for parallel processing. Loads a single trajectory,
    normalizes it, and flattens it.
    """
    path, stats = args
    try:
        sample = torch.load(path, map_location='cpu', weights_only=False)
        trajectory_meters = torch.from_numpy(sample['target_trajectory'][:, :2])
        
        # Normalize the trajectory to [-1, 1] before PCA
        trajectory_normalized = normalize_trajectory(trajectory_meters, stats)
        
        # Flatten from (80, 2) to (160,)
        return trajectory_normalized.flatten().numpy()
    except Exception as e:
        print(f"Warning: Skipping corrupted file {path}. Error: {e}")
        return None

# ==============================================================================
# === Main Script Logic ========================================================
# ==============================================================================

def main():
    print("--- Computing PCA Model for Trajectory Compression ---")
    config = load_config(os.path.join(PROJECT_ROOT, 'configs/main_config.yaml'))
    
    # We will add a new section to the config for PCA parameters.
    pca_cfg = config.get('pca', {'n_components': 16})
    n_components = pca_cfg.get('n_components', 16)
    
    # --- 1. Load Prerequisite Data ---
    featurized_dir = config['data']['featurized_dir_onlyxy']
    stats_path = os.path.join(PROJECT_ROOT, 'models', 'normalization_stats.pt')

    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Normalization stats not found at '{stats_path}'. "
            "Please run `compute_normalization_stats.py` first."
        )
    stats = torch.load(stats_path)
    
    train_files = glob(os.path.join(featurized_dir, 'training', '*.pt'))
    if not train_files:
        raise FileNotFoundError("No training files found. Please run the featurizer first.")

    # --- 2. Load and Prepare All Trajectories in Parallel ---
    print(f"Loading and normalizing {len(train_files)} trajectories...")
    
    # Create argument list for the parallel map function
    map_args = [(path, stats) for path in train_files]

    num_workers = os.cpu_count()
    with multiprocessing.Pool(processes=num_workers) as pool:
        results_iterator = pool.imap_unordered(_load_and_normalize_trajectory, map_args)
        pbar = tqdm(results_iterator, total=len(train_files), desc="Loading trajectories in parallel")
        flattened_trajectories = [traj for traj in pbar if traj is not None]

    if not flattened_trajectories:
        raise ValueError("Could not load any valid trajectories.")
        
    # Stack into a single large NumPy array: (num_samples, 160)
    data_matrix = np.stack(flattened_trajectories, axis=0)
    print(f"Created data matrix of shape: {data_matrix.shape}")

    # --- 3. Fit the PCA Model ---
    print(f"Fitting PCA model with n_components = {n_components}...")
    pca = PCA(n_components=n_components, whiten=True) # `whiten=True` is often beneficial
    pca.fit(data_matrix)
    print("PCA model fitted successfully.")

    # --- 4. Analyze and Report Results ---
    explained_variance = pca.explained_variance_ratio_.sum()
    print(f"\n--- PCA Analysis ---")
    print(f"Number of components: {n_components}")
    print(f"Total Explained Variance: {explained_variance:.4f}")
    print(f"This means our {n_components}-dimensional latent space captures "
          f"{explained_variance:.2%} of the variance of the original data.")

    # --- 5. Save the Fitted PCA Model ---
    output_path = os.path.join(PROJECT_ROOT, 'models', 'trajectory_pca.pkl')
    with open(output_path, 'wb') as f:
        pickle.dump(pca, f)
    
    print(f"\nFitted PCA model saved to: {output_path}")

if __name__ == '__main__':
    main()