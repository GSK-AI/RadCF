"""
ConditionedRadEditPipeline: Inference pipeline for LoRARadEditModel.

This file is modified from https://huggingface.co/microsoft/radedit/blob/main/pipeline.py.
See ./LICENSE_originals/LICENSE-RadEdit-pipeline for the original license.

Wraps a RadEditPipeline (or StableDiffusionPipeline) so that every UNet
call inside the DDPM inversion and denoising loops uses projected metadata
as cross-attention input instead of text embeddings.
"""

from __future__ import annotations

import os
import sys
import importlib
from typing import Dict, Optional

import torch
from diffusers import StableDiffusionPipeline
from diffusers.image_processor import PipelineImageInput

from unet_conditioned import ConditionedUNet


def _load_radedit_module():
    """
    Load the RadEdit pipeline module from the HuggingFace cached source.

    Returns the module object containing RadEditPipeline class and
    standalone functions (inversion_forward_process, etc.).
    """
    import glob
    import importlib.util

    # Find the cached pipeline.py from microsoft/radedit
    patterns = [
        os.path.expanduser(
            "~/.cache/huggingface/modules/diffusers_modules/local/"
            "microsoft--radedit/*/pipeline.py"
        ),
    ]
    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            spec = importlib.util.spec_from_file_location("radedit_pipeline", matches[0])
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod

    return None


def _load_radedit_pipeline_cls():
    """Load just the RadEditPipeline class."""
    mod = _load_radedit_module()
    if mod is not None:
        return mod.RadEditPipeline
    print("Warning: RadEditPipeline not found, falling back to StableDiffusionPipeline")
    return StableDiffusionPipeline


class ConditionedRadEditPipeline:
    """
    Wraps a loaded RadEditPipeline (or StableDiffusionPipeline) so that
    every UNet call uses projected metadata for cross-attention.

    Args:
        pipeline:  A loaded pipeline whose .unet is a ConditionedUNet.
        embedder:  The trained AttrEmbedder.
        meta_proj: The trained MetadataProjector.
    """

    def __init__(self, pipeline, embedder: torch.nn.Module, meta_proj: torch.nn.Module):
        assert isinstance(pipeline.unet, ConditionedUNet), (
            "pipeline.unet must be a ConditionedUNet. "
            "Replace it before constructing ConditionedRadEditPipeline."
        )
        self.pipeline = pipeline
        self.embedder = embedder
        self.meta_proj = meta_proj

    @classmethod
    def from_model(
        cls,
        lora_radedit_model,
        model_id: str = "microsoft/radedit",
    ) -> "ConditionedRadEditPipeline":
        """
        Build a ConditionedRadEditPipeline from a trained LoRARadEditModel.

        Loads the official RadEdit pipeline (correct VAE, scheduler, etc.),
        swaps in the LoRA'd conditioned UNet, and uses the RadEditPipeline
        class (which adds DDPM inversion + mask editing).
        """
        from diffusers import DiffusionPipeline, DDPMScheduler

        # Load official RadEdit pipeline (has correct VAE and scheduler)
        pipeline = DiffusionPipeline.from_pretrained(
            model_id, trust_remote_code=True,
        )

        # RadEdit requires DDPMScheduler for DDPM inversion (not DDIM)
        pipeline.scheduler = DDPMScheduler.from_config(pipeline.scheduler.config)

        # Swap in the LoRA'd conditioned UNet
        pipeline.unet = lora_radedit_model.conditioned_unet

        device = next(lora_radedit_model.conditioned_unet.parameters()).device
        pipeline = pipeline.to(device)

        return cls(
            pipeline=pipeline,
            embedder=lora_radedit_model.embedder,
            meta_proj=lora_radedit_model.meta_proj,
        )

    def __call__(
        self,
        prompt: str | list[str],
        image: PipelineImageInput,
        metadata: Optional[Dict[str, torch.Tensor]] = None,
        edit_mask: Optional[PipelineImageInput] = None,
        keep_mask: Optional[PipelineImageInput] = None,
        invert_prompt: str = "",
        weights: float | list[float] = 1.0,
        num_inference_steps: int = 50,
        skip_ratio: float = 0.5,
        eta: float = 1.0,
        prog_bar: bool = True,
        output_type: str | None = "np",
    ):
        """
        Edit an image with metadata conditioning via cross-attention.

        Splits the RadEdit pipeline into two explicit phases so that
        each phase uses the correct metadata conditioning:

        1. Inversion (forward process): uses null metadata conditioning.
           Only one UNet call per timestep (unconditional), so single mode.
        2. Denoising (reverse process): uses CFG with null metadata (uncond)
           vs target metadata (cond). Two UNet calls per timestep, so
           the ConditionedUNet alternates between null and target cross-attn.
        """
        conditioned_unet: ConditionedUNet = self.pipeline.unet
        device = next(self.embedder.parameters()).device
        pipeline = self.pipeline

        # --- Compute cross-attention embeddings ---
        with torch.no_grad():
            if metadata is not None:
                metadata_on_device = {
                    k: v.to(device) if torch.is_tensor(v) else v
                    for k, v in metadata.items()
                }
                y_emb = self.embedder(metadata_on_device, training=False)
                cond_cross_attn = self.meta_proj(y_emb)
            else:
                cond_cross_attn = self.meta_proj(
                    self.embedder.get_null_embedding_sum(device, 1)
                )

            B = cond_cross_attn.shape[0]
            null_cross_attn = self.meta_proj(
                self.embedder.get_null_embedding_sum(device, B)
            )

        # --- Load RADEdit functions ---
        radedit_mod = _load_radedit_module()
        inversion_forward_process = radedit_mod.inversion_forward_process
        inversion_reverse_process_two_masks = radedit_mod.inversion_reverse_process_two_masks

        # --- Setup ---
        skip = int(num_inference_steps * skip_ratio)
        pipeline.scheduler.set_timesteps(num_inference_steps)
        pipeline.scheduler.num_inference_steps = num_inference_steps

        # Preprocess image → latents
        processed_images = pipeline.image_processor.preprocess(image)
        latents = pipeline.encode_images_vae(processed_images)

        # Preprocess masks
        if edit_mask is not None:
            edit_mask = pipeline.image_processor.preprocess(edit_mask)
            edit_mask = (edit_mask > 0).to(dtype=edit_mask.dtype)
            edit_mask = pipeline.downsample_mask(edit_mask, latents)
            edit_mask = edit_mask.to(pipeline.device)

        if keep_mask is not None:
            keep_mask = pipeline.image_processor.preprocess(keep_mask)
            keep_mask = (keep_mask > 0).to(dtype=keep_mask.dtype)
            keep_mask = pipeline.downsample_mask(keep_mask, latents)
            keep_mask = keep_mask.to(pipeline.device)

        try:
            # --- Phase 1: Inversion (unconditional, single mode) ---
            # invert_prompt="" → only one UNet call per timestep (unconditional)
            conditioned_unet.set_metadata_cross_attn(null_cross_attn)

            _, noises, timestep_to_latents = inversion_forward_process(
                pipeline,
                latents,
                etas=1.0,
                prompt=invert_prompt,
                cfg_scale=weights,
                prog_bar=prog_bar,
                num_inference_steps=num_inference_steps,
            )

            batch_size = latents.size(0)

            # --- Phase 2: Denoising with CFG (paired mode) ---
            # Two UNet calls per timestep: 1st = uncond (null), 2nd = cond (target)
            conditioned_unet.set_cfg_cross_attn(null_cross_attn, cond_cross_attn)

            denoised_latents, _ = inversion_reverse_process_two_masks(
                pipeline,
                xT=timestep_to_latents[:, num_inference_steps - skip],
                timestep_to_latents=timestep_to_latents,
                edit_mask=edit_mask,
                keep_mask=keep_mask,
                etas=eta,
                prompts=[prompt] * batch_size,
                cfg_scales=weights,
                prog_bar=prog_bar,
                zs=noises[:, : (num_inference_steps - skip)],
            )
        finally:
            conditioned_unet.clear_metadata_cross_attn()

        # --- Decode ---
        with torch.no_grad():
            denoised_latents = denoised_latents / pipeline.vae.config.scaling_factor
            edited_image = pipeline.vae.decode(
                denoised_latents, return_dict=False
            )[0]
        do_denormalize = [True] * edited_image.shape[0]
        edited_image = pipeline.image_processor.postprocess(
            edited_image, output_type=output_type, do_denormalize=do_denormalize,
        )
        return edited_image

    def __getattr__(self, name):
        return getattr(self.pipeline, name)
