"""
ConditionedUNet: Wraps a UNet2DConditionModel (or PEFT-wrapped variant) to
inject metadata conditioning via cross-attention.

Instead of text embeddings from CLIP, the cross-attention layers receive
projected metadata embeddings. This puts the conditioning signal directly
through the LoRA'd to_q/k/v/to_out.0 layers.

    Old (time embedding hook):
        emb = time_embedding(t) + metadata_emb   →  ResNet blocks (no LoRA)
        cross-attn keys/values = null text        →  LoRA'd layers (wasted)

    New (cross-attention injection):
        cross-attn keys/values = projected metadata  →  LoRA'd layers (direct)
"""

import torch
import torch.nn as nn
from typing import Optional


class ConditionedUNet(nn.Module):
    """
    Thin wrapper around a UNet2DConditionModel that replaces
    encoder_hidden_states (text embeddings) with projected metadata
    for cross-attention conditioning.

    Supports two inference modes:

    Single mode (inversion): all UNet calls use the same cross-attn.
        conditioned_unet.set_metadata_cross_attn(cross_attn)

    CFG mode (denoising): alternates between uncond and cond cross-attn.
        The RADEdit denoising loop calls the UNet twice per timestep:
        first for unconditional prediction, then for conditional.
        conditioned_unet.set_cfg_cross_attn(null_cross_attn, cond_cross_attn)

    Args:
        unet: UNet2DConditionModel, possibly PEFT-wrapped.
    """

    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet
        self._metadata_cross_attn: Optional[torch.Tensor] = None
        # CFG mode: paired uncond/cond cross-attention with toggle
        self._uncond_cross_attn: Optional[torch.Tensor] = None
        self._cond_cross_attn: Optional[torch.Tensor] = None
        self._next_is_uncond: bool = True

    def set_metadata_cross_attn(self, cross_attn: Optional[torch.Tensor]) -> None:
        """Single mode: all UNet calls use the same cross-attn (e.g. inversion)."""
        self._metadata_cross_attn = cross_attn
        self._uncond_cross_attn = None
        self._cond_cross_attn = None

    def set_cfg_cross_attn(
        self,
        uncond_cross_attn: torch.Tensor,
        cond_cross_attn: torch.Tensor,
    ) -> None:
        """CFG mode: alternate between uncond and cond cross-attn per timestep.

        The RADEdit denoising loop calls UNet twice per timestep:
          1st call (unconditional) → uses uncond_cross_attn
          2nd call (conditional)   → uses cond_cross_attn
        """
        self._uncond_cross_attn = uncond_cross_attn
        self._cond_cross_attn = cond_cross_attn
        self._metadata_cross_attn = None
        self._next_is_uncond = True

    def clear_metadata_cross_attn(self) -> None:
        """Clear all stored cross-attn state."""
        self._metadata_cross_attn = None
        self._uncond_cross_attn = None
        self._cond_cross_attn = None
        self._next_is_uncond = True

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        metadata_cross_attn: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """
        Forward pass. Replaces encoder_hidden_states with metadata
        cross-attention based on the current mode:

        - Training: metadata_cross_attn arg is passed directly.
        - Single mode: uses stored _metadata_cross_attn.
        - CFG mode: alternates between _uncond_cross_attn and _cond_cross_attn.
        """
        if metadata_cross_attn is not None:
            # Training: explicit cross-attn provided
            cross_attn = metadata_cross_attn
        elif self._uncond_cross_attn is not None:
            # CFG mode: alternate uncond/cond
            if self._next_is_uncond:
                cross_attn = self._uncond_cross_attn
            else:
                cross_attn = self._cond_cross_attn
            self._next_is_uncond = not self._next_is_uncond
        elif self._metadata_cross_attn is not None:
            # Single mode
            cross_attn = self._metadata_cross_attn
        else:
            cross_attn = None

        if cross_attn is not None:
            # Expand to match batch size (e.g. CFG doubles the batch)
            if cross_attn.shape[0] != sample.shape[0]:
                cross_attn = cross_attn.expand(sample.shape[0], -1, -1)
            encoder_hidden_states = cross_attn
        return self.unet(sample, timestep, encoder_hidden_states, **kwargs)

    # ------------------------------------------------------------------
    # Attribute delegation
    # ------------------------------------------------------------------
    @property
    def config(self):
        unet = self.unet
        if hasattr(unet, "config"):
            return unet.config
        return unet.base_model.model.config

    @property
    def in_channels(self):
        return self.config.in_channels

    @property
    def sample_size(self):
        return self.config.sample_size

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.unet, name)
