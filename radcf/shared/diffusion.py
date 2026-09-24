"""
Diffusion utilities for flow matching.

Handles timestep sampling, interpolation, and target computation.
"""

import torch


class DiffusionEngine:
    """
    Handles all diffusion-related computations.

    This encapsulates the flow matching logic that was previously
    duplicated across trainers.
    """

    def __init__(self, transformer):
        """
        Args:
            transformer: SiT transformer model with interpolant() method
        """
        self.transformer = transformer

    def compute_flow_targets(self, clean_latent, noise, t):
        """
        Compute noisy latents and velocity targets for flow matching.

        Args:
            clean_latent: z0 [B, C, H, W] - clean data
            noise: Gaussian noise [B, C, H, W]
            t: Timesteps [B] in [0, 1]

        Returns:
            tuple: (noisy_latent, velocity_target)
                - noisy_latent: x_t [B, C, H, W]
                - velocity_target: target velocity [B, C, H, W]
        """
        B = clean_latent.shape[0]

        # Get interpolation coefficients
        # t needs shape [B, 1, 1, 1] for broadcasting
        alpha, sigma, d_alpha, d_sigma = self.transformer.interpolant(
            t.view(B, 1, 1, 1), path_type='linear'
        )

        # Compute noisy latent: x_t = alpha * z0 + sigma * noise
        noisy_latent = alpha * clean_latent + sigma * noise

        # Compute velocity target: v = d_alpha * z0 + d_sigma * noise
        velocity_target = d_alpha * clean_latent + d_sigma * noise

        return noisy_latent, velocity_target

    def sample_timesteps(self, batch_size, device):
        """
        Sample random timesteps for training.

        Args:
            batch_size: Number of timesteps to sample
            device: Target device

        Returns:
            Timesteps [batch_size] uniformly sampled from [0, 1]
        """
        return torch.rand(batch_size, device=device)

    def sample_noise(self, shape, device):
        """
        Sample Gaussian noise.

        Args:
            shape: Shape of noise tensor
            device: Target device

        Returns:
            Gaussian noise with given shape
        """
        return torch.randn(shape, device=device)
