# src/evaluation/evaluate_prediction.py

"""
Main evaluation script for the Conditional Diffusion Policy.

This script loads a trained model checkpoint and evaluates its performance on the
unseen validation set using multi-modal prediction metrics.

It supports both DDPM and the faster DDIM sampling methods.

To run:
conda activate virtuoso_env
python -m src.evaluation.evaluate_prediction --checkpoint runs/DiffusionPolicy_Training/20250829_142341/checkpoints/best_model.pth  --sampler ddim --steps 50
python -m src.evaluation.evaluate_prediction --checkpoint runs/DiffusionPolicy_Training/20250829_142341/checkpoints/best_model.pth  --sampler ddpm

"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

from glob import glob
from tqdm import tqdm
import numpy as np
import json
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config
from src.diffusion_policy.networks import ConditionalUNet
from src.diffusion_policy.dataset import InMemoryDiffusionDataset, custom_collate_fn

# ==============================================================================
# === Helper Functions =========================================================
# ==============================================================================

# In src/evaluation/evaluate_prediction.py
# Replace the old prepare_diffusion_schedule function with this one.

def prepare_diffusion_schedule(config: dict, device: str):
    """
    --- UPGRADED VERSION ---
    Pre-computes the DDPM/DDIM schedule constants, supporting both
    linear and cosine schedules based on the config file.
    """
    schedule_name = config['diffusion']['beta_schedule']
    num_steps = config['diffusion']['num_diffusion_steps']
    
    # --- Generate betas based on the schedule name ---
    if schedule_name == 'linear':
        betas = torch.linspace(
            config['diffusion']['beta_start'], 
            config['diffusion']['beta_end'], 
            num_steps, 
            device=device
        )
    elif schedule_name == 'cosine':
        s = 0.008
        steps = num_steps + 1
        x = torch.linspace(0, num_steps, steps, device=device)
        alphas_cumprod = torch.cos(((x / num_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        betas = torch.clip(betas, 0.0001, 0.9999)
    else:
        raise NotImplementedError(f"Schedule '{schedule_name}' not implemented.")

    # --- Compute the rest of the constants (same for both schedules) ---
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    
    # Required for DDPM sampling
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    # Prevent division by zero for the last step
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
    
    return {
        'betas': betas,
        'alphas_cumprod': alphas_cumprod,
        'sqrt_alphas_cumprod': torch.sqrt(alphas_cumprod),
        'sqrt_one_minus_alphas_cumprod': torch.sqrt(1.0 - alphas_cumprod),
        'posterior_variance': posterior_variance
    }

def move_batch_to_device(batch, device):
    batch['target_trajectory'] = batch['target_trajectory'].to(device)
    for key, value in batch['state'].items():
        batch['state'][key] = value.to(device)
    return batch

# ==============================================================================
# === Samplers (The Core Inference Logic) ======================================
# ==============================================================================

@torch.no_grad()
def sample_ddpm(model, state_dict, shape, schedule, device):
    """
    Performs the original, stochastic DDPM sampling process.
    This uses the full number of diffusion steps.
    """
    batch_size = shape[0]
    num_steps = schedule['betas'].shape[0]
    
    # Start with pure Gaussian noise
    x_t = torch.randn(shape, device=device)
    
    # Loop backwards from T-1 down to 0
    for t in tqdm(reversed(range(0, num_steps)), desc="DDPM Sampling", total=num_steps, leave=False):
        # Create the timestep tensor for the model
        time_cond = torch.full((batch_size,), t, device=device, dtype=torch.long)
        
        # Predict the noise from the model
        pred_noise = model(x_t, time_cond, state_dict)
        
        # Get pre-computed schedule values for this timestep
        alpha_t = schedule['alphas_cumprod'][t]
        alpha_t_prev = schedule['alphas_cumprod'][t-1] if t > 0 else torch.tensor(1.0, device=device)
        beta_t = schedule['betas'][t]
        
        # Calculate the mean of the posterior distribution q(x_{t-1} | x_t, x_0)
        x0_pred = (x_t - torch.sqrt(1. - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
        # x0_pred.clamp_(-1., 1.) # A common practice to stabilize generation
        
        mean_t = (torch.sqrt(alpha_t_prev) * beta_t * x0_pred + torch.sqrt(alpha_t) * (1. - alpha_t_prev) * x_t) / (1. - alpha_t)
        
        # Get the variance (it's fixed in DDPM)
        variance_t = schedule['posterior_variance'][t]
        
        # Sample from the distribution: x_{t-1} ~ N(mean_t, variance_t)
        noise = torch.randn_like(x_t)
        
        # Don't add noise at the last step (t=0)
        mask = torch.tensor(1.0 if t > 0 else 0.0, device=device).view(-1, *([1] * (len(x_t.shape) - 1)))
        
        x_t = mean_t + mask * torch.sqrt(variance_t) * noise
        
    return x_t

@torch.no_grad()
def sample_ddim(model, state_dict, shape, num_steps, schedule, device):
    """
    Performs the fast, deterministic DDIM sampling process.
    --- UPGRADED: Uses a more numerically stable update step. ---
    """
    batch_size = shape[0]
    total_steps = schedule['betas'].shape[0]
    
    # Create the DDIM timestep sequence
    times = torch.linspace(-1, total_steps - 1, steps=num_steps + 1).long()
    times = list(reversed(times.tolist()))
    time_pairs = list(zip(times[:-1], times[1:]))
    
    x_t = torch.randn(shape, device=device)

    for time, time_next in time_pairs:
        # Prepare the timestep tensor
        time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
        
        # Predict the noise `epsilon` for the current step `t`
        pred_noise = model(x_t, time_cond, state_dict)
        
        # Get the schedule constants for the current and next timesteps
        alpha_t = schedule['alphas_cumprod'][time]
        alpha_next = schedule['alphas_cumprod'][time_next] if time_next >= 0 else torch.tensor(1.0, device=device)

        # --- THIS IS THE ROBUST DDIM UPDATE EQUATION ---
        # 1. Predict the final clean sample x_0 using the noise prediction
        pred_x0 = (x_t - torch.sqrt(1. - alpha_t) * pred_noise) / torch.sqrt(alpha_t)

        # Optional: Clip the predicted x_0 to be in the valid data range
        # This is a key technique for stabilizing DDIM.
        pred_x0 = torch.clamp(pred_x0, -1.0, 1.0)
        
        # 2. Calculate the final x_{t-1} using the predicted x_0
        # This formulation is less prone to exploding.
        x_t = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1. - alpha_next) * pred_noise
        
    return x_t

# ==============================================================================
# === Metric Calculation =======================================================
# ==============================================================================

def compute_multimodal_metrics(predictions, ground_truth):
    """
    Calculates minADE and minFDE for a set of K predictions.
    
    Args:
        predictions (Tensor): Shape (K, H, 2) - K predictions of H waypoints in 2D.
        ground_truth (Tensor): Shape (H, 2) - The single ground truth trajectory.
    """
    # Calculate displacement errors (L2 distance) for all waypoints
    # Broadcasting takes care of the K dimension. Shape -> (K, H)
    displacement_errors = torch.linalg.norm(predictions - ground_truth.unsqueeze(0), dim=-1)
    
    # ADE: Average over the horizon. Shape -> (K,)
    ade = torch.mean(displacement_errors, dim=-1)
    
    # FDE: Final displacement error. Shape -> (K,)
    fde = displacement_errors[:, -1]
    
    # Find the minimum over the K predictions
    min_ade, _ = torch.min(ade, dim=0)
    min_fde, _ = torch.min(fde, dim=0)
    
    return min_ade.item(), min_fde.item()

def denormalize_trajectory(trajectory, stats):
    """De-normalizes an (x, y) trajectory from [-1, 1] back to original scale."""
    return ((trajectory + 1) / 2) * (stats['max'] - stats['min']) + stats['min']

# ==============================================================================
# === Main Evaluation Script ===================================================
# ==============================================================================

def main(args):
    print("--- Starting Evaluation ---")
    config = load_config(args.config)
    eval_cfg = config['evaluation']
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Load Model ---
    model = ConditionalUNet(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()
    print(f"Loaded model checkpoint from: {args.checkpoint}")
    
    stats_path = os.path.join(PROJECT_ROOT, 'models', 'normalization_stats.pt')
    stats = torch.load(stats_path, map_location=device)
    
    # --- Load Data ---
    val_files = glob(os.path.join(config['data']['featurized_dir_onlyxy'], 'validation', '*.pt'))
    val_dataset = InMemoryDiffusionDataset(val_files[:100], stats_path=stats_path)  # <-- Limiting to 10 samples for quick testing
    val_loader = DataLoader(val_dataset, batch_size=1, collate_fn=custom_collate_fn)
    
    # --- Prepare for Evaluation ---
    schedule = prepare_diffusion_schedule(config, device)
    results = []
    
    # --- Main Evaluation Loop ---
    for batch in tqdm(val_loader, desc="Evaluating on validation set"):
        batch = move_batch_to_device(batch, device)
        state_dict = batch['state']
        gt_trajectory_normalized  = batch['target_trajectory']
        gt_trajectory_meters = denormalize_trajectory(gt_trajectory_normalized, stats)

        # Generate K predictions for this scene
        predictions_k = []
        for _ in range(eval_cfg['k_samples']):
            if args.sampler == 'ddim':
                pred_traj_normalized = sample_ddim(
                    model, state_dict, gt_trajectory_normalized.shape, args.steps, schedule, device
                )
            elif args.sampler == 'ddpm':
                # DDPM uses the full schedule, so the `steps` argument is ignored.
                pred_traj_normalized = sample_ddpm(
                    model, state_dict, gt_trajectory_normalized.shape, schedule, device
                )
            else:
                raise ValueError(f"Unknown sampler: {args.sampler}")
            predictions_k.append(pred_traj_normalized)
            
        predictions_k = torch.cat(predictions_k, dim=0)
        predictions_k_meters = denormalize_trajectory(predictions_k, stats)

        min_ade, min_fde = compute_multimodal_metrics(
            predictions_k_meters[..., :2], # Use de-normalized predictions
            gt_trajectory_meters.squeeze(0)[..., :2] # Use de-normalized ground truth
        )
        miss_rate = 1.0 if min_fde > 2.0 else 0.0

        results.append({
            'minADE': min_ade,
            'minFDE': min_fde,
            'MissRate': miss_rate
        })
    
    # --- Aggregate and Report Results ---
    avg_min_ade = np.mean([r['minADE'] for r in results])
    avg_min_fde = np.mean([r['minFDE'] for r in results])
    avg_miss_rate = np.mean([r['MissRate'] for r in results])
    
    report = {
        'model': args.checkpoint,
        'sampler': args.sampler,
        'sampling_steps': args.steps if args.sampler == 'ddim' else config['diffusion']['num_diffusion_steps'],
        'k_samples': eval_cfg['k_samples'],
        'num_scenarios': len(val_dataset),
        'metrics': {
            'minADE': avg_min_ade,
            'minFDE': avg_min_fde,
            'MissRate@2m': avg_miss_rate
        }
    }

    print("\n--- Evaluation Complete ---")
    print(json.dumps(report['metrics'], indent=4))
    
    output_filename = f"eval_results_{args.sampler}"
    if args.sampler == 'ddim':
        output_filename += f"_{args.steps}steps"
    output_filename += ".json"
    
    output_path = os.path.join(os.path.dirname(args.checkpoint), output_filename)
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=4)
    print(f"Results saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--checkpoint', type=str, required=True, 
        help='Path to the trained model .pth checkpoint file.'
    )
    parser.add_argument(
        '--config', type=str, default=os.path.join(PROJECT_ROOT, 'configs/main_config.yaml'),
        help='Path to the project configuration file.'
    )
    parser.add_argument(
        '--sampler', type=str, default='ddim', choices=['ddim', 'ddpm'], # <-- Added 'ddpm'
        help='Which sampling method to use.'
    )
    parser.add_argument(
        '--steps', type=int, default=50,
        help='Number of steps for the sampler.'
    )
    args = parser.parse_args()
    main(args)