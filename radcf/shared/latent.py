"""
Latent encoding/decoding utilities.

Handles VAE operations for converting between image and latent space.
"""

import torch


class LatentEncoder:
    """
    Handles VAE encoding and decoding.

    Encapsulates the logic for converting images to/from latent space,
    including bias and scale normalization.
    """

    def __init__(self, vae, latents_bias, latents_scale):
        """
        Args:
            vae: VAE model (frozen)
            latents_bias: Bias tensor for latent normalization
            latents_scale: Scale tensor for latent normalization
        """
        self.vae = vae
        self.latents_bias = latents_bias
        self.latents_scale = latents_scale

    @torch.no_grad()
    def encode(self, images):
        """
        Encode images to latent space.

        Args:
            images: Images [B, 3, H, W]

        Returns:
            Latents [B, C, H//8, W//8] (normalized)
        """
        posterior = self.vae.encode(images)

        # Extract mode from posterior
        try:
            z_raw = posterior.mode()
        except AttributeError:
            z_raw = posterior.latent_dist.mode()

        # Normalize: z = (z_raw - bias) * scale
        z_normalized = (z_raw - self.latents_bias.to(z_raw)) * self.latents_scale.to(z_raw)

        return z_normalized

    @torch.no_grad()
    def decode(self, latents):
        """
        Decode latents to image space.

        Args:
            latents: Latents [B, C, H, W] (normalized)

        Returns:
            Images [B, 3, H*8, W*8]
        """
        # Denormalize: z_raw = z / scale + bias
        z_denormalized = latents / self.latents_scale.to(latents) + self.latents_bias.to(latents)

        # Decode
        decoded = self.vae.decode(z_denormalized)

        return decoded.sample
