# src/diffusion_policy/train.py


"""
Main training script for the Conditional Diffusion Policy.

This script orchestrates the entire training pipeline, including:
- Loading the dataset and model.
- Setting up the diffusion noise schedule.
- Running the main training loop with the DDPM loss objective.
- Performing periodic validation.
- Logging metrics to TensorBoard.
- Saving model checkpoints.

To run:
conda activate virtuoso_env
python -m src.diffusion_policy.train
"""

import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from glob import glob
from tqdm import tqdm
from datetime import datetime
import yaml
import itertools
from typing import Dict, Any, Tuple
import argparse
# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config
from src.diffusion_policy.networks import ConditionalUNet
from src.diffusion_policy.dataset import InMemoryDiffusionDataset, custom_collate_fn 

# ==============================================================================
# === Helper Functions =========================================================
# ==============================================================================

def get_beta_schedule(schedule_name: str, num_steps: int, start: float, end: float) -> torch.Tensor:
    if schedule_name == 'linear':
        return torch.linspace(start, end, num_steps)
    elif schedule_name == 'cosine':
        # From the "Improved DDPM" paper
        s = 0.008
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    else:
        raise NotImplementedError(f"Schedule '{schedule_name}' not implemented.")

def extract_into_tensor(arr: torch.Tensor, timesteps: torch.Tensor, broadcast_shape: Tuple) -> torch.Tensor:
    """
    Extracts values from a 1D array `arr` at the indices specified by `timesteps`,
    and reshapes the result to a target `broadcast_shape` for element-wise operations.
    """
    res = arr.to(timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res.unsqueeze(-1)
    return res.expand(broadcast_shape)

def move_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    """Recursively moves a batch of tensors to a specified device."""
    batch['target_trajectory'] = batch['target_trajectory'].to(device)
    for key, value in batch['state'].items():
        batch['state'][key] = value.to(device)
    return batch

# ==============================================================================
# === Main Training Logic ======================================================
# ==============================================================================

def main(args):
    # --- 1. Setup and Configuration ---
    config = load_config(os.path.join(PROJECT_ROOT, 'configs/main_config.yaml'))
    train_cfg = config['training']
    
    device = torch.device(train_cfg['device'] if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create a unique directory for this runc
    if args.resume_from_checkpoint:
        run_dir = os.path.dirname(os.path.dirname(args.resume_from_checkpoint))
    else:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_dir = os.path.join(PROJECT_ROOT, 'runs', 'DiffusionPolicy_Training', timestamp)
        
    os.makedirs(run_dir, exist_ok=True)
    checkpoints_dir = os.path.join(run_dir, 'checkpoints')
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    with open(os.path.join(run_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
        
    writer = SummaryWriter(log_dir=run_dir)
    print(f"Run artifacts will be saved to: {run_dir}")

    # --- 2. Prepare Diffusion Schedule ---
    betas = get_beta_schedule(
        config['diffusion']['beta_schedule'],
        config['diffusion']['num_diffusion_steps'],
        config['diffusion']['beta_start'],
        config['diffusion']['beta_end']
    )
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # --- 3. Load Data using In-Memory Caching ---
    print("--- Loading and Caching Data ---")
    featurized_dir = config['data']['featurized_dir_onlyxy']
    train_files = glob(os.path.join(featurized_dir, 'training', '*.pt'))
    val_files = glob(os.path.join(featurized_dir, 'validation', '*.pt'))
    
    stats_path = os.path.join(PROJECT_ROOT, 'models', 'normalization_stats.pt')
    train_dataset = InMemoryDiffusionDataset(train_files, stats_path=stats_path)
    val_dataset = InMemoryDiffusionDataset(val_files, stats_path=stats_path)
    
    train_loader = DataLoader(
        train_dataset, batch_size=train_cfg['batch_size'], num_workers=train_cfg['num_workers'],
        collate_fn=custom_collate_fn, pin_memory=True, persistent_workers=True,
        shuffle=True # Enable shuffling for standard datasets
    )
    val_loader = DataLoader(
        val_dataset, batch_size=train_cfg['batch_size'], num_workers=train_cfg['num_workers'],
        collate_fn=custom_collate_fn
    )

    # --- 4. Initialize Model, Optimizer, and Scheduler ---
    model = ConditionalUNet(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg['learning_rate'], weight_decay=train_cfg['weight_decay'])
    
    # --- NEW: Instantiate the Scheduler ---
    scheduler_cfg = train_cfg.get('scheduler', {}) # Use .get for backward compatibility
    if scheduler_cfg.get('type') == 'cosine':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=scheduler_cfg.get('T_0', 50000),
            T_mult=scheduler_cfg.get('T_mult', 1),
            eta_min=scheduler_cfg.get('eta_min', 1e-7)
        )
        print("Using CosineAnnealingWarmRestarts scheduler.")
    else:
        scheduler = None # No scheduler
        print("Not using a learning rate scheduler.")
        
    start_step = 0
    best_val_loss = float('inf')
    if args.resume_from_checkpoint:
        print(f"--- Resuming training from checkpoint: {args.resume_from_checkpoint} ---")
        checkpoint = torch.load(args.resume_from_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint['step'] + 1
        best_val_loss = checkpoint.get('loss', float('inf')) # Use .get for old checkpoints
        if scheduler and 'scheduler_state_dict' in checkpoint:
             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print(f"Resuming from step {start_step} with best validation loss {best_val_loss:.6f}")
        
    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters.")

    # --- 5. The Main Training Loop (Corrected for InMemoryDataset) ---
    print("Starting training...")
    
    # Use a progress bar that tracks the total number of steps
    progress_bar = tqdm(total=train_cfg['num_train_steps'], initial=start_step, desc="Training Steps")
    global_step = start_step

    # The outer loop is just to ensure we keep training until we hit the target step count
    while global_step < train_cfg['num_train_steps']:
        # The inner loop iterates through one epoch of the shuffled data
        for batch in train_loader:
            if global_step >= train_cfg['num_train_steps']:
                break
                
            model.train()
            
            # --- 5.1. Get Batch and Move to Device ---
            batch = move_batch_to_device(batch, device)
            clean_trajectory = batch['target_trajectory']
            state_dict = batch['state']
            
            # --- 5.2. DDPM Forward Process ---
            t = torch.randint(0, config['diffusion']['num_diffusion_steps'], (clean_trajectory.shape[0],), device=device).long()
            noise = torch.randn_like(clean_trajectory)
            noisy_trajectory = (
                extract_into_tensor(sqrt_alphas_cumprod, t, clean_trajectory.shape) * clean_trajectory +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, clean_trajectory.shape) * noise
            )
            
            # --- 5.3. Get Model Prediction ---
            predicted_noise = model(noisy_trajectory, t, state_dict)
            
            # --- 5.4. Calculate Loss and Optimize ---
            loss = F.mse_loss(predicted_noise, noise)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()
            
            # --- 5.5. Logging ---
            if global_step % train_cfg['log_interval_steps'] == 0:
                writer.add_scalar('Loss/train', loss.item(), global_step)
                
                if scheduler is not None:
                    writer.add_scalar('LearningRate', scheduler.get_last_lr()[0], global_step)
            
            # --- 5.6. Periodic Validation and Checkpointing ---
            if global_step > 0 and global_step % train_cfg['eval_interval_steps'] == 0:
                model.eval()
                total_val_loss = 0
                val_batches = 0
                with torch.no_grad():
                    for val_batch in val_loader: # Use a smaller progress bar for validation
                        val_batch = move_batch_to_device(val_batch, device)
                        # ... (validation loss calculation is the same as before) ...
                        clean_traj_val, state_dict_val = val_batch['target_trajectory'], val_batch['state']
                        t_val = torch.randint(0, config['diffusion']['num_diffusion_steps'], (clean_traj_val.shape[0],), device=device).long()
                        noise_val = torch.randn_like(clean_traj_val)
                        noisy_traj_val = (
                            extract_into_tensor(sqrt_alphas_cumprod, t_val, clean_traj_val.shape) * clean_traj_val +
                            extract_into_tensor(sqrt_one_minus_alphas_cumprod, t_val, clean_traj_val.shape) * noise_val
                        )
                        pred_noise_val = model(noisy_traj_val, t_val, state_dict_val)
                        val_loss = F.mse_loss(pred_noise_val, noise_val)
                        total_val_loss += val_loss.item()
                        val_batches += 1
                
                avg_val_loss = total_val_loss / val_batches
                writer.add_scalar('Loss/validation', avg_val_loss, global_step)
                progress_bar.set_postfix(val_loss=f"{avg_val_loss:.6f}")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_checkpoint_path = os.path.join(checkpoints_dir, 'best_model.pth')
                    save_dict = {
                        'step': global_step,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'loss': best_val_loss,
                    }
                    if scheduler is not None:
                        save_dict['scheduler_state_dict'] = scheduler.state_dict()
                    
                    torch.save(save_dict, best_checkpoint_path)

                        
                    tqdm.write(f"Step {global_step}: New best model saved with validation loss {avg_val_loss:.6f}")

            # --- 5.7. Update Counters ---
            global_step += 1
            progress_bar.update(1)

    progress_bar.close()
    print("Training finished.")
    writer.close()
    
    # Save the final model state dict
    final_model_path = os.path.join(checkpoints_dir, 'final_model.pth')
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--resume_from_checkpoint', type=str, default=None,
        help='Path to a model checkpoint to resume training from.'
    )
    args = parser.parse_args()
    main(args)