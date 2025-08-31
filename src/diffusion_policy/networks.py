# src/diffusion_policy/networks.py

"""
Defines the neural network architectures for the conditional diffusion policy.

This file contains the core components:
1. SinusoidalTimeEmbedding: Encodes the diffusion timestep.
2. FiLMConditionedResidualBlock: The main building block for the U-Net,
   which injects conditioning information via FiLM layers.
3. StateEncoder: A powerful, attention-based module that processes the
   structured `state_dict` into a single scene embedding vector.
4. ConditionalUNet: The final, top-level model that assembles all components
   and performs the denoising task.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict

# ==============================================================================
# === Component 1: Time Embedding ==============================================
# ==============================================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Standard sinusoidal time embedding module from the DDPM paper, followed by
    an MLP to make it more expressive.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t is shape [B]
        device = t.device
        half_dim = self.embed_dim // 2
        
        # Standard sinusoidal formula
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat([embeddings.sin(), embeddings.cos()], dim=-1)
        
        return embeddings

# ==============================================================================
# === Component 2: The Core U-Net Block ========================================
# ==============================================================================

class FiLMConditionedResidualBlock(nn.Module):
    """
    A residual block for the 1D U-Net that is conditioned via FiLM
    (Feature-wise Linear Modulation).
    """
    def __init__(self, in_channels: int, out_channels: int, cond_embed_dim: int):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # If the number of channels changes, we need a simple 1x1 conv for the residual connection
        self.residual_conv = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # The "FiLM Generator" is a small MLP that takes the conditioning embedding
        # and predicts the scale (gamma) and shift (beta) parameters.
        self.film_generator = nn.Linear(cond_embed_dim, out_channels * 2)
        
        self.activation = nn.Mish()

    def forward(self, x: torch.Tensor, cond_embedding: torch.Tensor) -> torch.Tensor:
        # x is shape [B, C_in, L]
        # cond_embedding is shape [B, D_cond]
        
        # Generate FiLM parameters
        film_params = self.film_generator(cond_embedding)
        gamma, beta = torch.chunk(film_params, 2, dim=-1) # Split into two tensors
        # Reshape gamma and beta to be broadcastable with x for the FiLM operation
        gamma = gamma.unsqueeze(-1) # -> [B, C_out, 1]
        beta = beta.unsqueeze(-1)   # -> [B, C_out, 1]

        # Main path
        h = self.conv1(x)
        h = self.norm1(h)
        
        # Apply FiLM modulation
        h = h * gamma + beta
        
        h = self.activation(h)
        h = self.conv2(h)
        h = self.norm2(h)
        
        # Add residual connection and return
        return self.activation(h + self.residual_conv(x))

# ==============================================================================
# === Component 3: The Context Encoder =========================================
# ==============================================================================

class StateEncoder(nn.Module):
    """
    Processes the structured state_dict into a single scene embedding vector
    using dedicated encoders for each entity and a Transformer for fusion.
    """
    def __init__(self, embed_dim: int, num_attn_heads: int = 4):
        super().__init__()
        
        # --- Define dedicated encoders for each entity type ---
        self.ego_history_encoder = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=32, kernel_size=3, padding=1), nn.Mish(),
            nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.Mish(),
            nn.AdaptiveAvgPool1d(1) # Pool along the time dimension to get a single vector
        )
        self.ego_out_proj = nn.Linear(64, embed_dim)
        
        self.agent_encoder = nn.Linear(11, embed_dim) # From (B, 16, 11) to (B, 16, embed_dim)
        self.map_encoder = nn.Linear(20, embed_dim)   # From (B, 64, 10, 2) -> (B, 64, 20) to (B, 64, embed_dim)
        self.goal_encoder = nn.Linear(2, embed_dim)    # From (B, 5, 2) to (B, 5, embed_dim)
        self.tl_encoder = nn.Linear(2, embed_dim)      # From (B, 2) to (B, embed_dim)
        
        # --- The [CLS] token, a learnable parameter ---
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        
        # --- Transformer Encoder for fusing all entity embeddings ---
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_attn_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=2)

    def forward(self, state_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
        # --- 1. Encode each entity independently ---
        # Note the permutation for Conv1d: (B, C, L)
        ego_hist = state_dict['ego_history'].permute(0, 2, 1)
        ego_embedding = self.ego_history_encoder(ego_hist).squeeze(-1)
        ego_embedding = self.ego_out_proj(ego_embedding).unsqueeze(1) # (B, 1, embed_dim)

        # Flatten map features for the MLP
        map_geom = state_dict['map'].flatten(start_dim=2) # (B, 64, 20)
        
        agent_embeddings = self.agent_encoder(state_dict['agents'])     # (B, 16, embed_dim)
        map_embeddings = self.map_encoder(map_geom)                     # (B, 64, embed_dim)
        goal_embeddings = self.goal_encoder(state_dict['goal'])         # (B, 5, embed_dim)
        tl_embedding = self.tl_encoder(state_dict['traffic_lights']).unsqueeze(1) # (B, 1, embed_dim)
        
        # --- 2. Build the full sequence of tokens for the Transformer ---
        batch_size = ego_embedding.shape[0]
        
        # Use masks to zero out embeddings for non-existent entities before concatenation
        # This prevents the model from attending to padded garbage.
        agent_embeddings = agent_embeddings * state_dict['agents_mask'].unsqueeze(-1)
        map_embeddings = map_embeddings * state_dict['map_mask'].unsqueeze(-1)
        goal_embeddings = goal_embeddings * state_dict['goal_mask'].unsqueeze(-1)
        # tl_mask is always True, so no need to apply it
        
        # Prepend the [CLS] token
        cls_token = self.cls_token.expand(batch_size, -1, -1)
        
        full_sequence = torch.cat([
            cls_token,
            ego_embedding,
            agent_embeddings,
            map_embeddings,
            goal_embeddings,
            tl_embedding
        ], dim=1)
        
        # --- 3. Build the attention mask for the Transformer ---
        # This tells the Transformer which tokens are real and which are padding.
        # [CLS] and ego are always real.
        cls_mask = torch.ones(batch_size, 1, device=full_sequence.device, dtype=torch.bool)
        ego_mask = torch.ones(batch_size, 1, device=full_sequence.device, dtype=torch.bool)
        
        # We need to flatten the goal mask for this to work
        full_mask = torch.cat([
            cls_mask,
            ego_mask,
            state_dict['agents_mask'],
            state_dict['map_mask'],
            state_dict['goal_mask'],
            state_dict['traffic_lights_mask'] # This is shape (B, 1)
        ], dim=1)
        
        # Sequence length of `full_sequence` should now match `full_mask`.
        assert full_sequence.shape[1] == full_mask.shape[1], \
            f"Shape mismatch! Sequence length {full_sequence.shape[1]} != Mask length {full_mask.shape[1]}"

        # --- 4. Pass through the Transformer ---
        transformer_output = self.transformer(src=full_sequence, src_key_padding_mask=~full_mask)
        
        # --- 5. Extract the [CLS] token's output ---
        # The [CLS] token is the first token in the sequence (index 0).
        # Its final hidden state is our holistic scene embedding.
        scene_embedding = transformer_output[:, 0, :]
        
        return scene_embedding

# ==============================================================================
# === Component 4: The Final Assembled U-Net ===================================
# ==============================================================================

class ConditionalUNet(nn.Module):
    """
    The main model. It takes a noisy trajectory, a timestep, and the scene context,
    and predicts the noise that was added to the trajectory.
    """
    def __init__(self, config: Dict):
        super().__init__()
        
        trajectory_dim = config['model']['trajectory_dim']
        time_embed_dim = config['model']['time_embed_dim']
        scene_embed_dim = config['model']['scene_embed_dim']
        down_dims = config['model']['down_dims']
        
        # The total dimension of the conditioning vector
        cond_embed_dim = time_embed_dim + scene_embed_dim
        
        # --- Instantiate all sub-modules ---
        self.time_encoder = nn.Sequential(
            SinusoidalTimeEmbedding(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim), nn.Mish(),
            nn.Linear(time_embed_dim, time_embed_dim)
        )
        self.state_encoder = StateEncoder(embed_dim=scene_embed_dim)
        
        # --- U-Net Architecture ---
        # The input to the U-Net is a sequence of (x, y, heading)
        # We need to transpose it to (B, C, L) for Conv1d
        
        self.initial_conv = nn.Conv1d(trajectory_dim, down_dims[0], kernel_size=1)
        
        # Downsampling path
        self.down_blocks = nn.ModuleList()
        for i in range(len(down_dims) - 1):
            self.down_blocks.append(nn.ModuleList([
                FiLMConditionedResidualBlock(down_dims[i], down_dims[i], cond_embed_dim),
                FiLMConditionedResidualBlock(down_dims[i], down_dims[i+1], cond_embed_dim),
                nn.Conv1d(down_dims[i+1], down_dims[i+1], kernel_size=3, stride=2, padding=1) # Downsample
            ]))
        
        # Middle block
        self.middle_block1 = FiLMConditionedResidualBlock(down_dims[-1], down_dims[-1], cond_embed_dim)
        self.middle_block2 = FiLMConditionedResidualBlock(down_dims[-1], down_dims[-1], cond_embed_dim)
        
        self.up_blocks = nn.ModuleList()
        up_dims = down_dims[::-1]
        for i in range(len(up_dims) - 1):
            # Note the order of layers in the ModuleList for clarity
            self.up_blocks.append(nn.ModuleList([
                nn.ConvTranspose1d(up_dims[i], up_dims[i], kernel_size=4, stride=2, padding=1),
                FiLMConditionedResidualBlock(up_dims[i] * 2, up_dims[i+1], cond_embed_dim),
                FiLMConditionedResidualBlock(up_dims[i+1], up_dims[i+1], cond_embed_dim),
            ]))
            
        self.final_conv = nn.Conv1d(down_dims[0], trajectory_dim, kernel_size=1)
        
    def forward(
        self, 
        noisy_trajectory: torch.Tensor, # (B, L, C_traj)
        timestep: torch.Tensor,         # (B,)
        state_dict: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        
        # --- 1. Prepare Conditioning Vector ---
        time_embedding = self.time_encoder(timestep)
        scene_embedding = self.state_encoder(state_dict)
        cond_embedding = torch.cat([time_embedding, scene_embedding], dim=1)
        
        # --- 2. U-Net Forward Pass ---
        # Input shape for Conv1d is (B, C, L), so we need to transpose
        x = noisy_trajectory.permute(0, 2, 1)
        
        x = self.initial_conv(x)
        
        skip_connections = []
        # Downsampling
        for res1, res2, downsample in self.down_blocks:
            x = res1(x, cond_embedding)
            x = res2(x, cond_embedding)
            skip_connections.append(x)
            x = downsample(x)
            
        # Middle
        x = self.middle_block1(x, cond_embedding)
        x = self.middle_block2(x, cond_embedding)
                
        # Upsampling
        for upsample, res1, res2 in self.up_blocks:
            # First, upsample the feature map from the lower level
            x = upsample(x)
            
            # Get the corresponding skip connection (pop from the end)
            skip = skip_connections.pop()
            
            # The defensive crop is now for its intended purpose: handling minor off-by-one errors
            if x.shape[-1] != skip.shape[-1]:
                # This should NOT print with our current config, but is good practice
                print(f"Cropping skip connection from {skip.shape} to {x.shape}")
                diff = skip.shape[-1] - x.shape[-1]
                skip = skip[..., diff//2 : -(diff - diff//2)]

            # Concatenate the upsampled feature map and the skip connection
            x = torch.cat([x, skip], dim=1)
            
            # Process through the residual blocks
            x = res1(x, cond_embedding)
            x = res2(x, cond_embedding)
            
        # Final projection
        x = self.final_conv(x)
        
        # Transpose back to (B, L, C_traj) to match the noise input
        predicted_noise = x.permute(0, 2, 1)
        
        return predicted_noise