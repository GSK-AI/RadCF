"""
LoRARadEditModel: LoRA fine-tuning of a UNet with schema-driven metadata conditioning.

Supports any SD1.x-compatible backbone (RADEdit, SD1.5, etc.).

Metadata conditioning flows through cross-attention (replacing text embeddings),
so LoRA adapters on to_q/k/v/to_out.0 directly learn metadata-conditioned attention.

    Conditioning path:
        metadata → AttrEmbedder [B, D] → MetadataProjector [B, T, 768] → cross-attention

    Trainable:  LoRA adapters (to_q/k/v/to_out.0) + AttrEmbedder + MetadataProjector
    Frozen:     UNet base weights, VAE

time_embed_dim is derived from the UNet config: block_out_channels[0] * 4.
All AttrEmbedder schema dims must equal this value for sum-mode compatibility.
  - RADEdit:  128 * 4 = 512
  - SD1.5:    320 * 4 = 1280
"""

import os
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DiffusionPipeline, UNet2DConditionModel
from peft import LoraConfig, PeftModel, get_peft_model

from radcf.shared.embedder import AttrEmbedder
from unet_conditioned import ConditionedUNet


def _load_radedit_components(
    model_id: str = "microsoft/radedit",
    torch_dtype: torch.dtype = torch.float32,
):
    """Load UNet, VAE, and scheduler from the official RadEdit pipeline.

    This ensures we use the same VAE and scheduler that the RadEdit model
    card specifies, rather than substituting components from a different
    pipeline (e.g. SD 1.5 inpainting).

    Args:
        model_id: HuggingFace model ID (must be ``microsoft/radedit`` or a
                  local path to the same model).
        torch_dtype: Weight dtype.

    Returns:
        (unet, vae, scheduler) extracted from the loaded pipeline.
    """
    print(f"Loading RadEdit pipeline from {model_id}...")
    pipe = DiffusionPipeline.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=torch_dtype,
    )
    return pipe.unet, pipe.vae, pipe.scheduler


class MetadataProjector(nn.Module):
    """Project metadata embedding [B, meta_dim] → [B, num_tokens, cross_attn_dim]."""

    def __init__(self, meta_dim: int, cross_attn_dim: int, num_tokens: int = 4):
        super().__init__()
        self.num_tokens = num_tokens
        self.cross_attn_dim = cross_attn_dim
        self.proj = nn.Linear(meta_dim, num_tokens * cross_attn_dim)

    def forward(self, meta_emb: torch.Tensor) -> torch.Tensor:
        # meta_emb: [B, meta_dim] → [B, num_tokens, cross_attn_dim]
        return self.proj(meta_emb).view(-1, self.num_tokens, self.cross_attn_dim)


class LoRARadEditModel(nn.Module):
    """
    RADEdit UNet with LoRA adapters and schema-driven metadata conditioning
    via cross-attention.

    Args:
        conditioned_unet:  ConditionedUNet wrapping the PEFT-LoRA UNet
        embedder:          AttrEmbedder (output_mode='sum', dim=time_embed_dim)
        meta_proj:         MetadataProjector (time_embed_dim → cross_attn tokens)
        vae:               Frozen VAE
        scheduler:         Noise scheduler (DDPM/DDIM)
    """

    def __init__(
        self,
        conditioned_unet: ConditionedUNet,
        embedder: AttrEmbedder,
        meta_proj: MetadataProjector,
        vae: nn.Module,
        scheduler,
    ):
        super().__init__()
        self.conditioned_unet = conditioned_unet
        self.embedder = embedder
        self.meta_proj = meta_proj
        self.vae = vae
        self.scheduler = scheduler

        self.vae.requires_grad_(False)
        self.vae.eval()

        self.embedder.requires_grad_(True)
        self.meta_proj.requires_grad_(True)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_pretrained(
        cls,
        model_id: str = "microsoft/radedit",
        schema: list = None,
        lora_config: dict = None,
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float32,
    ) -> "LoRARadEditModel":
        """
        Build a fresh LoRARadEditModel from RADEdit pretrained weights.

        All components (UNet, VAE, scheduler) are loaded from the official
        RadEdit pipeline to ensure the correct latent / noising system.

        Args:
            model_id:    HF model ID for the RadEdit pipeline
                         (default: ``microsoft/radedit``; gated — requires
                         ``huggingface-cli login`` and accepted access terms).
            schema:      AttrEmbedder schema list with dim=time_embed_dim.
            lora_config: Dict: rank, alpha, target_modules, bias,
                         cfg_dropout_prob, num_meta_tokens.
            device:      Target device string.
            torch_dtype: Dtype for model weights.

        Returns:
            LoRARadEditModel ready for training.
        """
        schema = schema or []
        lora_config = lora_config or {}

        unet, vae, scheduler = _load_radedit_components(model_id, torch_dtype)

        # Derive dims from UNet config
        unet_cfg = unet.config
        time_embed_dim = getattr(unet_cfg, "time_embedding_dim", None) or (
            unet_cfg.block_out_channels[0] * 4
        )
        cross_attn_dim = unet_cfg.cross_attention_dim  # 768 for SD 1.x
        print(f"  UNet time_embed_dim: {time_embed_dim}, cross_attn_dim: {cross_attn_dim}")

        # Validate schema dims against time_embed_dim
        for item in schema:
            if item.get("dim") != time_embed_dim:
                raise ValueError(
                    f"Schema item '{item['name']}' has dim={item.get('dim')} "
                    f"but UNet time_embed_dim={time_embed_dim}. "
                    f"Set dim={time_embed_dim} for all schema items in LoRA mode."
                )

        # Apply PEFT LoRA to UNet attention layers
        peft_cfg = LoraConfig(
            r=lora_config.get("rank", 16),
            lora_alpha=lora_config.get("alpha", 32),
            target_modules=lora_config.get(
                "target_modules", ["to_q", "to_k", "to_v", "to_out.0"]
            ),
            bias=lora_config.get("bias", "none"),
        )
        unet = get_peft_model(unet, peft_cfg)
        print("LoRA configuration:")
        unet.print_trainable_parameters()

        # Wrap UNet
        conditioned_unet = ConditionedUNet(unet).to(device)

        # Build schema-driven metadata embedder
        embedder = AttrEmbedder(
            schema,
            dropout_prob=lora_config.get("cfg_dropout_prob", 0.1),
            output_mode="sum",
        ).to(device)

        # Metadata projector: [B, time_embed_dim] → [B, num_tokens, cross_attn_dim]
        num_meta_tokens = lora_config.get("num_meta_tokens", 4)
        meta_proj = MetadataProjector(
            meta_dim=time_embed_dim,
            cross_attn_dim=cross_attn_dim,
            num_tokens=num_meta_tokens,
        ).to(device)
        print(f"  MetadataProjector: {time_embed_dim} → {num_meta_tokens} × {cross_attn_dim}")

        vae.to(device)

        return cls(
            conditioned_unet=conditioned_unet,
            embedder=embedder,
            meta_proj=meta_proj,
            vae=vae,
            scheduler=scheduler,
        )

    @classmethod
    def from_checkpoint(
        cls,
        lora_ckpt_dir: str,
        schema: list,
        model_id: str = "microsoft/radedit",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float32,
    ) -> "LoRARadEditModel":
        """
        Load a trained LoRARadEditModel from a saved checkpoint directory.

        Expects:
            {lora_ckpt_dir}/adapter_model/   — PEFT LoRA adapter
            {lora_ckpt_dir}/embedder.pt       — AttrEmbedder state dict
            {lora_ckpt_dir}/meta_proj.pt      — MetadataProjector state dict

        Returns:
            LoRARadEditModel in eval mode.
        """
        unet, vae, scheduler = _load_radedit_components(model_id, torch_dtype)

        unet_cfg = unet.config
        time_embed_dim = getattr(unet_cfg, "time_embedding_dim", None) or (
            unet_cfg.block_out_channels[0] * 4
        )
        cross_attn_dim = unet_cfg.cross_attention_dim

        adapter_dir = os.path.join(lora_ckpt_dir, "adapter_model")
        print(f"Loading LoRA weights from {adapter_dir}...")
        peft_unet = PeftModel.from_pretrained(unet, adapter_dir)
        peft_unet.eval()

        conditioned_unet = ConditionedUNet(peft_unet).to(device)

        # Override schema dims to match UNet time_embed_dim
        for item in schema:
            item["dim"] = time_embed_dim

        embedder = AttrEmbedder(schema, dropout_prob=0.0, output_mode="sum").to(device)
        emb_path = os.path.join(lora_ckpt_dir, "embedder.pt")
        print(f"Loading embedder from {emb_path}...")
        embedder.load_state_dict(torch.load(emb_path, map_location=device))
        embedder.eval()

        # Load projector — infer num_tokens from saved weights
        proj_path = os.path.join(lora_ckpt_dir, "meta_proj.pt")
        print(f"Loading meta_proj from {proj_path}...")
        proj_state = torch.load(proj_path, map_location=device)
        out_features = proj_state["proj.weight"].shape[0]
        num_tokens = out_features // cross_attn_dim
        meta_proj = MetadataProjector(
            meta_dim=time_embed_dim,
            cross_attn_dim=cross_attn_dim,
            num_tokens=num_tokens,
        ).to(device)
        meta_proj.load_state_dict(proj_state)
        meta_proj.eval()

        vae.to(device)

        model = cls(
            conditioned_unet=conditioned_unet,
            embedder=embedder,
            meta_proj=meta_proj,
            vae=vae,
            scheduler=scheduler,
        )
        model.eval()
        return model

    @classmethod
    def from_lightning_checkpoint(
        cls,
        ckpt_path: str,
        schema: list,
        model_id: str = "microsoft/radedit",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.float32,
    ) -> "LoRARadEditModel":
        """
        Load a trained LoRARadEditModel from a PyTorch Lightning .ckpt file.

        Lightning saves the full state_dict with keys prefixed by 'model.'
        (from LoRARadEditTrainer.model). This method builds a fresh model
        and loads the trained weights from the checkpoint.

        Returns:
            LoRARadEditModel in eval mode.
        """
        print(f"Loading Lightning checkpoint from {ckpt_path}...")
        ckpt = torch.load(ckpt_path, map_location=device)
        state_dict = ckpt["state_dict"]

        # Strip 'model.' prefix added by Lightning
        state_dict = {
            k.removeprefix("model."): v for k, v in state_dict.items()
            if k.startswith("model.")
        }

        unet, vae, scheduler = _load_radedit_components(model_id, torch_dtype)

        unet_cfg = unet.config
        time_embed_dim = getattr(unet_cfg, "time_embedding_dim", None) or (
            unet_cfg.block_out_channels[0] * 4
        )
        cross_attn_dim = unet_cfg.cross_attention_dim

        # Infer LoRA config from saved keys
        lora_keys = [k for k in state_dict if "lora_A" in k]
        if lora_keys:
            rank = state_dict[lora_keys[0]].shape[0]
        else:
            rank = 16

        peft_cfg = LoraConfig(
            r=rank,
            lora_alpha=rank * 2,
            target_modules=["to_q", "to_k", "to_v", "to_out.0"],
            bias="none",
        )
        unet = get_peft_model(unet, peft_cfg)
        conditioned_unet = ConditionedUNet(unet).to(device)

        embedder = AttrEmbedder(schema, dropout_prob=0.0, output_mode="sum").to(device)

        # Infer num_tokens from saved projector weights
        proj_weight = state_dict["meta_proj.proj.weight"]
        num_tokens = proj_weight.shape[0] // cross_attn_dim
        meta_proj = MetadataProjector(
            meta_dim=time_embed_dim,
            cross_attn_dim=cross_attn_dim,
            num_tokens=num_tokens,
        ).to(device)

        model = cls(
            conditioned_unet=conditioned_unet,
            embedder=embedder,
            meta_proj=meta_proj,
            vae=vae,
            scheduler=scheduler,
        )

        # Load trained weights
        model.load_state_dict(state_dict, strict=False)
        model.to(device)
        model.eval()
        print("Model loaded from Lightning checkpoint.")
        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images [B,3,H,W] → latents [B,4,H/8,W/8] via frozen VAE."""
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
            latents = latents * self.vae.config.scaling_factor
        return latents

    # ------------------------------------------------------------------
    # Training / inference
    # ------------------------------------------------------------------

    def training_forward(
        self,
        images: torch.Tensor,
        metadata: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """
        Compute training loss: standard DDPM noise-prediction objective.

        Steps:
          1. VAE-encode images → z0
          2. Sample timestep t and noise ε
          3. z_t = add_noise(z0, ε, t)
          4. Metadata embedding → project to cross-attention tokens
          5. ε_pred = UNet(z_t, t, cross_attn=projected_metadata)
          6. loss = MSE(ε_pred, target)

        Args:
            images:    [B, 3, H, W], normalized to [-1, 1]
            metadata:  Dict of tensors matching the AttrEmbedder schema

        Returns:
            Scalar loss tensor.
        """
        B = images.shape[0]
        device = images.device

        z0 = self._encode_images(images)
        t = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (B,), device=device
        )
        noise = torch.randn_like(z0)
        z_t = self.scheduler.add_noise(z0, noise, t)

        y_emb = self.embedder(metadata, training=True)    # [B, 512]
        cross_attn = self.meta_proj(y_emb)                # [B, T, 768]

        model_output = self.conditioned_unet(
            z_t, t, cross_attn
        ).sample

        pred_type = self.scheduler.config.prediction_type
        if pred_type == "epsilon":
            target = noise
        elif pred_type == "v_prediction":
            target = self.scheduler.get_velocity(z0, noise, t)
        else:
            raise ValueError(f"Unsupported prediction_type: {pred_type!r}")

        return F.mse_loss(model_output, target)

    def get_trainable_parameters(self):
        """Return all trainable parameters: LoRA adapters + embedder + projector."""
        return [
            p for p in self.conditioned_unet.parameters() if p.requires_grad
        ] + list(self.embedder.parameters()) + list(self.meta_proj.parameters())

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save_checkpoint(self, output_dir: str) -> None:
        """
        Save LoRA adapters, embedder, and projector to output_dir.

        Layout:
            {output_dir}/adapter_model/   — PEFT LoRA adapter files
            {output_dir}/embedder.pt      — AttrEmbedder state dict
            {output_dir}/meta_proj.pt     — MetadataProjector state dict
        """
        os.makedirs(output_dir, exist_ok=True)

        adapter_dir = os.path.join(output_dir, "adapter_model")
        self.conditioned_unet.unet.save_pretrained(adapter_dir)
        print(f"LoRA adapter saved → {adapter_dir}")

        emb_path = os.path.join(output_dir, "embedder.pt")
        torch.save(self.embedder.state_dict(), emb_path)
        print(f"Embedder saved      → {emb_path}")

        proj_path = os.path.join(output_dir, "meta_proj.pt")
        torch.save(self.meta_proj.state_dict(), proj_path)
        print(f"MetaProj saved      → {proj_path}")
