"""
Model wrappers for radcf inference and training.

Each wrapper provides a uniform interface (inference_forward, forward,
get_trainable_parameters) around a SiT transformer + metadata embedder.
Loading logic lives in loaders.py and inference_pipeline.py, not here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional


class BaseModel(nn.Module):
    """
    Wrapper for the untouched pretrained base transformer.

    Used for true base-model inference, e.g. zT inversion with the original
    pretrained velocity field, without any fine-tuning effects.
    """

    def __init__(
        self,
        transformer: nn.Module,
        latent_encoder,
        null_token: int = 0,
    ):
        super().__init__()
        self.transformer = transformer
        self.latent_encoder = latent_encoder
        self.null_token = null_token

        self.transformer.requires_grad_(False)
        self.transformer.eval()

    def inference_forward(
        self,
        noisy_latent: torch.Tensor,
        t: torch.Tensor,
        metadata=None,
    ) -> torch.Tensor:
        B = noisy_latent.shape[0]
        device = noisy_latent.device
        null_token = torch.full((B,), self.null_token, device=device, dtype=torch.long)

        with torch.no_grad():
            v_base = self.transformer.inference(noisy_latent, t, y=null_token)

        return v_base


class ScratchModel(nn.Module):
    """
    Inference wrapper for scratch-trained SiT models (produced by run_train.py).
    """

    def __init__(self, transformer: nn.Module, latent_encoder):
        super().__init__()
        self.transformer = transformer
        self.latent_encoder = latent_encoder
        self.transformer.requires_grad_(False)
        self.transformer.eval()

    def inference_forward(
        self,
        noisy_latent: torch.Tensor,
        t: torch.Tensor,
        metadata: Optional[Dict[str, torch.Tensor]],
    ) -> torch.Tensor:
        with torch.no_grad():
            sit = self.transformer
            B = noisy_latent.shape[0]
            device = noisy_latent.device

            x = sit.x_embedder(noisy_latent) + sit.pos_embed
            t_emb = sit.t_embedder(t)

            if isinstance(metadata, dict):
                y_emb = sit.y_embedder(metadata, training=False)
            else:
                y_emb = sit.y_embedder.get_null_embedding_sum(device, B)

            c = t_emb + y_emb

            for block in sit.blocks:
                x = block(x, c)

            x = sit.final_layer(x, c)
            return sit.unpatchify(x)


class LoRAModel(nn.Module):
    """Self-contained LoRA training/inference model."""

    def __init__(
        self,
        transformer: nn.Module,
        embedder: nn.Module,
        diffusion_engine,
        latent_encoder,
        lora_config: Dict,
        apply_lora: bool = True,
    ):
        super().__init__()

        self.diffusion_engine = diffusion_engine
        self.latent_encoder = latent_encoder

        transformer.y_embedder = embedder

        if apply_lora:
            self.transformer = self._apply_lora(transformer, lora_config)
        else:
            self.transformer = transformer

    def _apply_lora(self, transformer, lora_config):
        from peft import LoraConfig, get_peft_model

        peft_config = LoraConfig(
            r=lora_config.get('rank', 16),
            lora_alpha=lora_config.get('alpha', 32),
            target_modules=lora_config.get('target_modules', ["qkv", "proj", "fc1", "fc2"]),
            bias=lora_config.get('bias', "none"),
            modules_to_save=lora_config.get('modules_to_save', ["y_embedder"]),
        )

        model = get_peft_model(transformer, peft_config)
        print("LoRA configuration:")
        model.print_trainable_parameters()
        return model

    def forward(self, images: torch.Tensor, metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.training_forward(images, metadata)

    def training_forward(
        self,
        images: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        B = images.shape[0]
        device = images.device

        clean_latent = self.latent_encoder.encode(images)

        t = self.diffusion_engine.sample_timesteps(B, device)
        noise = self.diffusion_engine.sample_noise(clean_latent.shape, device)

        noisy_latent, velocity_target = self.diffusion_engine.compute_flow_targets(
            clean_latent, noise, t
        )

        v_pred = self._manual_forward(noisy_latent, t, metadata, training=True)

        loss = F.mse_loss(v_pred, velocity_target)

        return loss

    def training_forward_from_latents(
        self,
        clean_latent: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Training forward from pre-encoded, normalized latents (for vae_update)."""
        B = clean_latent.shape[0]
        device = clean_latent.device
        t = self.diffusion_engine.sample_timesteps(B, device)
        noise = self.diffusion_engine.sample_noise(clean_latent.shape, device)
        noisy_latent, velocity_target = self.diffusion_engine.compute_flow_targets(
            clean_latent, noise, t
        )
        v_pred = self._manual_forward(noisy_latent, t, metadata, training=True)
        return F.mse_loss(v_pred, velocity_target)

    def inference_forward(
        self,
        noisy_latent: torch.Tensor,
        t: torch.Tensor,
        metadata: Dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self._manual_forward(
                noisy_latent, t, metadata, training=False
            )

    def _manual_forward(
        self,
        noisy_latent: torch.Tensor,
        t: torch.Tensor,
        metadata: Dict[str, torch.Tensor] | None,
        training: bool = False,
    ) -> torch.Tensor:
        sit = self._get_base_sit()
        B = noisy_latent.shape[0]
        device = noisy_latent.device

        x = sit.x_embedder(noisy_latent) + sit.pos_embed

        t_emb = sit.t_embedder(t)

        y_embedder = sit.y_embedder
        if isinstance(metadata, dict):
            y_emb = y_embedder(metadata, training=training)
        else:
            y_emb = y_embedder.get_null_embedding_sum(device, B)

        c = t_emb + y_emb

        for block in sit.blocks:
            x = block(x, c)

        x = sit.final_layer(x, c)
        return sit.unpatchify(x)

    def _get_base_sit(self):
        if hasattr(self.transformer, 'base_model'):
            if hasattr(self.transformer.base_model, 'model'):
                return self.transformer.base_model.model
            return self.transformer.base_model
        return self.transformer

    def get_trainable_parameters(self):
        return (p for p in self.transformer.parameters() if p.requires_grad)


class FullFTModel(nn.Module):
    """
    Full fine-tuning model: Train entire transformer.

    Includes y-embedding normalization specific to this mode.
    """

    def __init__(
        self,
        transformer: nn.Module,
        embedder: nn.Module,
        diffusion_engine,
        latent_encoder,
    ):
        super().__init__()

        self.transformer = transformer
        self.embedder = embedder
        self.diffusion_engine = diffusion_engine
        self.latent_encoder = latent_encoder

        self.transformer.y_embedder = self.embedder

        self.transformer.requires_grad_(True)

        # Freeze REPA components not used in our manual forward pass.
        if hasattr(self.transformer, 'projectors'):
            self.transformer.projectors.requires_grad_(False)
        if hasattr(self.transformer, 'bn'):
            self.transformer.bn.requires_grad_(False)

        trainable_params = sum(p.numel() for p in self.transformer.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.transformer.parameters())
        print(f"Full Fine-tuning: {trainable_params:,} / {total_params:,} parameters are trainable")

    def forward(self, images: torch.Tensor, metadata: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.training_forward(images, metadata)

    def training_forward(
        self,
        images: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        B = images.shape[0]
        device = images.device

        clean_latent = self.latent_encoder.encode(images)

        t = self.diffusion_engine.sample_timesteps(B, device)
        noise = self.diffusion_engine.sample_noise(clean_latent.shape, device)

        noisy_latent, velocity_target = self.diffusion_engine.compute_flow_targets(
            clean_latent, noise, t
        )

        v_pred = self._manual_forward_with_normalization(
            noisy_latent, t, metadata, training=True
        )

        loss = F.mse_loss(v_pred, velocity_target)

        return loss

    def training_forward_from_latents(
        self,
        clean_latent: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Training forward from pre-encoded, normalized latents (for vae_update)."""
        B = clean_latent.shape[0]
        device = clean_latent.device
        t = self.diffusion_engine.sample_timesteps(B, device)
        noise = self.diffusion_engine.sample_noise(clean_latent.shape, device)
        noisy_latent, velocity_target = self.diffusion_engine.compute_flow_targets(
            clean_latent, noise, t
        )
        v_pred = self._manual_forward_with_normalization(
            noisy_latent, t, metadata, training=True
        )
        return F.mse_loss(v_pred, velocity_target)

    def inference_forward(
        self,
        noisy_latent: torch.Tensor,
        t: torch.Tensor,
        metadata: Dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        with torch.no_grad():
            return self._manual_forward_with_normalization(
                noisy_latent, t, metadata, training=False
            )

    def _manual_forward_with_normalization(
        self,
        noisy_latent: torch.Tensor,
        t: torch.Tensor,
        metadata: Dict[str, torch.Tensor] | None,
        training: bool = False,
    ) -> torch.Tensor:
        B = noisy_latent.shape[0]
        device = noisy_latent.device

        x = self.transformer.x_embedder(noisy_latent) + self.transformer.pos_embed

        t_emb = self.transformer.t_embedder(t)

        if isinstance(metadata, dict):
            y_emb = self.embedder(metadata, training=training)
        else:
            y_emb = self.embedder.get_null_embedding_sum(device, B)

        y_emb = self._normalize_y_embedding(y_emb, t_emb)

        c = t_emb + y_emb

        for block in self.transformer.blocks:
            x = block(x, c)

        x = self.transformer.final_layer(x, c)
        return self.transformer.unpatchify(x)

    def _normalize_y_embedding(
        self,
        y_emb: torch.Tensor,
        t_emb: torch.Tensor,
    ) -> torch.Tensor:
        y_norm = y_emb.norm(dim=-1, keepdim=True) + 1e-6
        t_norm = t_emb.norm(dim=-1, keepdim=True) + 1e-6
        return y_emb / y_norm * t_norm

    def get_trainable_parameters(self):
        return (p for p in self.transformer.parameters() if p.requires_grad)
