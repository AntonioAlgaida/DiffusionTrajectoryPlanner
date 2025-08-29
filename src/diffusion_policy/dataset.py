# src/diffusion_policy/dataset.py

import os
import math
import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
from typing import List, Dict, Any
import numpy as np
from tqdm import tqdm
import multiprocessing

# ==============================================================================
# === Top-Level Helper Functions (for Multiprocessing & DataLoader) ============
# ==============================================================================

def _load_file(path: str) -> Dict[str, Any] | None:
    """Helper function to load a single .pt file. Runs in a worker process."""
    try:
        # The weights_only=False is important for loading older torch files
        # It can be removed if all data is saved with a recent PyTorch version.
        return torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Warning: Skipping corrupted file {path}. Error: {e}")
        return None

def normalize_trajectory(trajectory, stats):
    """Normalizes an (x, y) trajectory to the [-1, 1] range using isotropic stats."""
    return 2 * (trajectory - stats['min']) / (stats['max'] - stats['min']) - 1


def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function that converts numpy arrays in the state_dict to
    tensors and stacks all samples into a single batch.
    """
    collated_batch = {'state': {}, 'target_trajectory': []}
    state_keys = batch[0]['state'].keys()
    for key in state_keys:
        collated_batch['state'][key] = []

    for sample in batch:
        collated_batch['target_trajectory'].append(sample['target_trajectory'])
        for key in state_keys:
            collated_batch['state'][key].append(torch.from_numpy(sample['state'][key]))
    
    collated_batch['target_trajectory'] = torch.stack(collated_batch['target_trajectory'], dim=0)
    for key in state_keys:
        collated_batch['state'][key] = torch.stack(collated_batch['state'][key], dim=0)
        
    return collated_batch

# ==============================================================================
# === Dataset Class ============================================================
# ==============================================================================

class InMemoryDiffusionDataset(Dataset):
    """
    A dataset that pre-loads all samples into RAM for maximum training speed.
    Uses multiprocessing for fast, parallel caching.
    """
    def __init__(self, file_paths: List[str], stats_path: str):
        super().__init__()
        if not file_paths:
            raise ValueError("No file paths provided to the dataset.")
        
        self.stats = torch.load(stats_path)
        print("Loaded normalization stats.")
        
        print(f"Loading {len(file_paths)} samples into memory using parallel workers...")
        num_workers = os.cpu_count()
        with multiprocessing.Pool(processes=num_workers) as pool:
            results_iterator = pool.imap_unordered(_load_file, file_paths)
            pbar = tqdm(results_iterator, total=len(file_paths), desc="Caching data in parallel")
            self.data_cache = [sample for sample in pbar if sample is not None]
        
        print(f"Successfully loaded and cached {len(self.data_cache)} samples.")

    def __len__(self) -> int:
        return len(self.data_cache)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.data_cache[idx]
        normalized_sample = {
            'state': sample['state'],
            'target_trajectory': normalize_trajectory(
                torch.from_numpy(sample['target_trajectory']), self.stats
            ).float()
        }
        return normalized_sample