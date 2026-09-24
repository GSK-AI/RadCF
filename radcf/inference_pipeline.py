"""
InferencePipeline: Unified inference for all modes.

Supports:
- Batch inference: run_batch_counterfactual()
- Single image inference: run_single_counterfactual()
- Classifier-Free Guidance (CFG)
- Image preprocessing
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from radcf.shared import euler_sample, euler_invert, LatentEncoder, DiffusionEngine
from radcf.model_wrappers import BaseModel, ScratchModel, LoRAModel, FullFTModel
from radcf.loaders import load_base_checkpoint, load_full_checkpoint, load_lora_checkpoint


class InferencePipeline:
    """
    Universal inference pipeline for counterfactual generation.

    Supports modes: lora, full, scratch, base.

    Args:
        model: Model wrapper instance
        vae: VAE for encoding/decoding images
        latents_bias: Latent bias tensor
        latents_scale: Latent scale tensor
        schema: Schema list for metadata
        device: Device (cuda/cpu)
        null_token: Null token for unconditional generation
    """

    def __init__(
        self,
        model: nn.Module,
        vae: nn.Module,
        latents_bias: torch.Tensor,
        latents_scale: torch.Tensor,
        schema: list,
        device: str = "cuda",
        null_token: int = 0,
    ):
        self.model = model
        self.vae = vae
        self.latents_bias = latents_bias.to(device)
        self.latents_scale = latents_scale.to(device)
        self.schema = schema
        self.device = device
        self.null_token = null_token

        # Determine mode from model type
        if isinstance(model, LoRAModel):
            self.mode = "lora"
        elif isinstance(model, FullFTModel):
            self.mode = "full"
        elif isinstance(model, ScratchModel):
            self.mode = "scratch"
        elif isinstance(model, BaseModel):
            self.mode = "base"
        else:
            raise ValueError(f"Unknown model type: {type(model)}")

        # Latent encoder for convenience
        self.latent_encoder = LatentEncoder(vae, latents_bias, latents_scale)

        # Set to eval mode
        self.model.eval()
        self.vae.eval()

    @torch.no_grad()
    def run_batch_counterfactual(
        self,
        images_tensor: torch.Tensor,
        source_metas: Dict[str, torch.Tensor],
        target_metas: Dict[str, torch.Tensor],
        num_steps: int = 50,
        cfg_scale: float = 1.0,
        guidance_low: float = 0.0,
        guidance_high: float = 1.0,
    ):
        """
        Generate counterfactuals for a batch of images.

        Args:
            images_tensor: Input images [B, 3, H, W]
            source_metas: Source metadata dict
            target_metas: Target metadata dict
            num_steps: Number of ODE steps (default: 50)
            cfg_scale: Classifier-Free Guidance scale (default: 1.0)
            guidance_low: CFG start time (default: 0.0)
            guidance_high: CFG end time (default: 1.0)

        Returns:
            img_null_intervention: Reconstructed original images [B, 3, H, W]
            img_cf: Counterfactual images [B, 3, H, W]
        """
        B = images_tensor.shape[0]
        images_tensor = images_tensor.to(self.device)

        z0 = self.latent_encoder.encode(images_tensor)

        velocity_fn_source_inv = self._build_velocity_fn(
            source_metas,
            cfg_scale=1.0, #never use cfg in inversion
            guidance_low=guidance_low,
            guidance_high=guidance_high,
        )
        zT = euler_invert(velocity_fn_source_inv, z0, num_steps=num_steps)

        velocity_fn_null = self._build_velocity_fn(
            source_metas,
            cfg_scale=cfg_scale,  #use cfg in null intervention
            guidance_low=guidance_low,
            guidance_high=guidance_high,
        )
        z0_null = euler_sample(velocity_fn_null, zT, num_steps=num_steps)

        velocity_fn_target = self._build_velocity_fn(
            target_metas,
            cfg_scale=cfg_scale,
            guidance_low=guidance_low,
            guidance_high=guidance_high,
        )
        z0_cf = euler_sample(velocity_fn_target, zT, num_steps=num_steps)

        # Decode to images
        img_null = self.latent_encoder.decode(z0_null)
        img_cf = self.latent_encoder.decode(z0_cf)

        return img_null, img_cf

    def _build_velocity_fn(
        self,
        metadata: Dict[str, torch.Tensor],
        cfg_scale: float,
        guidance_low: float,
        guidance_high: float,
    ):
        """
        Build velocity function for ODE integration.

        Args:
            metadata: Metadata dict
            cfg_scale: CFG scale
            guidance_low: CFG start time
            guidance_high: CFG end time

        Returns:
            velocity_fn: Callable (x, t) -> v
        """
        def velocity_fn(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
            # Ensure t is tensor
            if not isinstance(t, torch.Tensor):
                t = torch.tensor([t], device=self.device, dtype=x.dtype)
            if t.ndim == 0:
                t = t.unsqueeze(0)

            # Expand t to batch size
            B = x.shape[0]
            t_batch = t.expand(B)

            # Apply CFG if requested
            if cfg_scale != 1.0 and guidance_low <= t.item() <= guidance_high:
                # Conditional velocity
                v_cond = self.model.inference_forward(x, t_batch, metadata)

                # Unconditional velocity
                v_uncond = self.model.inference_forward(x, t_batch, None)

                # CFG combination
                v = v_uncond + cfg_scale * (v_cond - v_uncond)
            else:
                # No CFG
                v = self.model.inference_forward(x, t_batch, metadata)

            return v

        return velocity_fn

    @classmethod
    def from_checkpoint(
        cls,
        mode: str,
        schema: list,
        device: str = "cuda",
        null_token: int = 0,
        # base mode
        base_model: Optional[str] = None,
        # lora mode
        lora_ckpt_path: Optional[str] = None,
        # full mode
        full_model_ckpt_path: Optional[str] = None,
        # scratch mode
        scratch_ckpt_path: Optional[str] = None,
    ) -> "InferencePipeline":
        """
        Load inference pipeline from checkpoint.

        Args:
            mode: "lora", "full", "scratch", or "base"
            schema: Metadata schema
            device: Device
            null_token: Null token
            base_model: Model zoo key or path (for base mode)
            lora_ckpt_path: Path to LoRA checkpoint dir (for lora mode)
            full_model_ckpt_path: Path to full model checkpoint (for full mode)
            scratch_ckpt_path: Path to scratch checkpoint (for scratch mode)

        Returns:
            pipeline: InferencePipeline instance
        """

        if mode == "lora":
            if not lora_ckpt_path:
                raise ValueError("lora mode requires lora_ckpt_path")

            vae, peft_transformer, latents_scale, latents_bias, _ = (
                load_lora_checkpoint(lora_ckpt_path, schema, device)
            )
            latent_encoder = LatentEncoder(vae, latents_bias, latents_scale)
            diffusion = DiffusionEngine(peft_transformer)
            model = LoRAModel(
                transformer=peft_transformer,
                embedder=peft_transformer.y_embedder,
                diffusion_engine=diffusion,
                latent_encoder=latent_encoder,
                lora_config={},
                apply_lora=False,
            )
            model.eval()

        elif mode in ("full", "scratch"):
            ckpt_path = full_model_ckpt_path if mode == "full" else scratch_ckpt_path
            if not ckpt_path:
                raise ValueError(f"{mode} mode requires "
                                 f"{'full_model_ckpt_path' if mode == 'full' else 'scratch_ckpt_path'}")

            vae, transformer, latents_scale, latents_bias, _ = (
                load_full_checkpoint(ckpt_path, schema, device)
            )
            latent_encoder = LatentEncoder(vae, latents_bias, latents_scale)
            diffusion = DiffusionEngine(transformer)

            if mode == "full":
                model = FullFTModel(
                    transformer=transformer,
                    embedder=transformer.y_embedder,
                    diffusion_engine=diffusion,
                    latent_encoder=latent_encoder,
                )
            else:
                model = ScratchModel(
                    transformer=transformer,
                    latent_encoder=latent_encoder,
                )
            model.eval()

        elif mode == "base":
            if not base_model:
                raise ValueError("base mode requires base_model")
            vae, transformer, latents_scale, latents_bias, _ = load_base_checkpoint(
                base_model, device
            )
            latent_encoder = LatentEncoder(vae, latents_bias, latents_scale)
            model = BaseModel(
                transformer=transformer,
                latent_encoder=latent_encoder,
                null_token=null_token,
            )

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # Create pipeline
        return cls(
            model=model,
            vae=vae,
            latents_bias=latents_bias,
            latents_scale=latents_scale,
            schema=schema,
            device=device,
            null_token=null_token,
        )
