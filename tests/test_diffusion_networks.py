# tests/test_diffusion_networks.py

"""
Unit test for the ConditionalUNet model defined in src/diffusion_policy/networks.py.

This test verifies the architectural integrity of the model by performing a
forward pass with a real data sample. It ensures that all sub-modules are
correctly connected and that the output tensor shape matches the input tensor shape.

To run this test from the project root:
conda activate virtuoso_env
python -m tests.test_diffusion_networks
"""

import os
import sys
import torch
from glob import glob
from typing import Dict, Any

# Add project root to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(PROJECT_ROOT)

from src.utils.config import load_config
from src.diffusion_policy.networks import ConditionalUNet

def numpy_dict_to_torch_dict(np_dict: Dict[str, Any], device: str) -> Dict[str, Any]:
    """Converts a dictionary of NumPy arrays to a dictionary of PyTorch tensors."""
    torch_dict = {}
    for key, value in np_dict.items():
        # Add a batch dimension and move to the specified device
        torch_dict[key] = torch.from_numpy(value).unsqueeze(0).to(device)
    return torch_dict

def test_conditional_unet_forward_pass():
    """
    Tests the forward pass of the ConditionalUNet with a real data sample.
    """
    print("\n--- Running Test: ConditionalUNet Forward Pass ---")

    # --- 1. Load Configuration and Find a Real Data Sample ---
    print("Loading configuration...")
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'main_config.yaml')
    config = load_config(config_path)

    featurized_dir = config['data']['featurized_dir_onlyxy']
    pt_files = glob(os.path.join(featurized_dir, '*', '*.pt'))
    if not pt_files:
        raise FileNotFoundError(
            f"No featurized data found in {featurized_dir}. "
            "Please run the featurizer script first."
        )
    
    sample_path = pt_files[0]
    print(f"Loading a real data sample from: {sample_path}")

    # --- 2. Prepare a Batch of Input Data ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Load the sample and add a batch dimension to everything
    sample_data = torch.load(sample_path, map_location='cpu', weights_only=False)
    
    # The state_dict contains numpy arrays. We need to convert them to a batch of torch tensors.
    state_dict_np = sample_data['state']
    state_dict_torch = numpy_dict_to_torch_dict(state_dict_np, device)

    # Create other inputs for the model
    batch_size = 1
    future_steps = int(config['trajectory']['future_sec'] / config['trajectory']['dt'])
    trajectory_dim = config['model']['trajectory_dim']
    num_diffusion_steps = config['diffusion']['num_diffusion_steps']

    # Use the real trajectory to create a noisy version
    clean_trajectory = torch.from_numpy(sample_data['target_trajectory']).unsqueeze(0).to(device)
    noise = torch.randn_like(clean_trajectory)
    noisy_trajectory = clean_trajectory + noise # Simplification for test purposes

    # Create a random timestep tensor
    timestep = torch.randint(1, num_diffusion_steps, (batch_size,), device=device)

    print(f"Input noisy_trajectory shape: {noisy_trajectory.shape}")
    print(f"Input timestep shape: {timestep.shape}")
    
    # --- 3. Instantiate the Model ---
    print("Instantiating ConditionalUNet model...")
    try:
        model = ConditionalUNet(config).to(device)
        model.eval() # Set to evaluation mode
        print("Model instantiated successfully.")
    except Exception as e:
        print("❌ FAILED: Could not instantiate model.")
        raise e

    # --- 4. Perform Forward Pass ---
    print("Performing a forward pass...")
    try:
        with torch.no_grad(): # No need to compute gradients for this test
            predicted_noise = model(noisy_trajectory, timestep, state_dict_torch)
        print("Forward pass completed successfully.")
    except Exception as e:
        print("❌ FAILED: Error during model forward pass.")
        raise e

    # --- 5. Assert Correctness ---
    print("Asserting output shape...")
    expected_shape = noisy_trajectory.shape
    actual_shape = predicted_noise.shape

    assert actual_shape == expected_shape, \
        f"Shape mismatch! Expected {expected_shape}, but got {actual_shape}."
    
    print(f"Output predicted_noise shape: {actual_shape}")
    print("✅ PASSED: Output shape is correct.")
    print("--- Test Complete ---")


if __name__ == '__main__':
    test_conditional_unet_forward_pass()