"""
Shared checkpoint loading and saving utilities for radcf.

Two checkpoint formats:
  Full-parameter:  Single .pt file with model_config, transformer, vae,
                   latents_scale, latents_bias.  Used by both run_train.py
                   and run_finetune.py (full mode).
  LoRA:            Directory with model_config.pt, vae.pt, lora_weights/.

Loading:
  load_base_checkpoint()   – load a pretrained base for fine-tuning
  load_full_checkpoint()   – load any full-parameter checkpoint for inference
  load_lora_checkpoint()   – load a LoRA checkpoint for inference

Saving:
  save_full_checkpoint()   – save full-parameter checkpoint
  save_lora_checkpoint()   – save LoRA checkpoint
"""

import gc
import json
import logging
import warnings
from pathlib import Path

import torch

from radcf.configs import get_checkpoint_path, get_checkpoint_info

logger = logging.getLogger(__name__)

# Backward compat: old checkpoints do not store z_dims in model_config.
_ENC_DIMS = {
    "dinov2-vit-s": 384,
    "dinov2-vit-b": 768,
    "dinov2-vit-l": 1024,
    "dinov2-vit-g": 1536,
    "clip-vit-b":   512,
    "clip-vit-l":   768,
    "dinov1-vit-b": 768,
}


def _enc_z_dims(enc_type: str) -> list[int]:
    for key, dim in _ENC_DIMS.items():
        if key in enc_type:
            return [dim]
    raise ValueError(
        f"Cannot infer z_dims for enc_type '{enc_type}'. "
        f"Known keys: {list(_ENC_DIMS)}"
    )


def _load_self_describing_base(state: dict, device: str):
    """Load transformer + VAE from a self-describing checkpoint (run_train.py output)."""
    from radcf.models.sit import SiT_models
    from radcf.models.autoencoder import vae_models

    cfg         = state["model_config"]
    in_channels = cfg["in_channels"]
    latent_size = cfg["resolution"] // (8 if cfg["vae"] == "f8d4" else 16)

    transformer = SiT_models[cfg["model"]](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=cfg["num_classes"],
        class_dropout_prob=cfg["cfg_prob"],
        z_dims=cfg["z_dims"],
        encoder_depth=cfg["encoder_depth"],
        fused_attn=cfg["fused_attn"],
        qk_norm=cfg["qk_norm"],
        label_type=cfg.get("label_type", "class"),
    ).to(device)

    weights = state.get("transformer") or state.get("ema") or state.get("model")
    result = transformer.load_state_dict(weights, strict=False)
    if result.missing_keys or result.unexpected_keys:
        logging.getLogger(__name__).warning(
            f"Non-strict load: missing={result.missing_keys}, "
            f"unexpected={result.unexpected_keys}"
        )
    transformer.eval()
    transformer.requires_grad_(False)

    vae = vae_models[cfg["vae"]]().to(device)
    vae.load_state_dict(state["vae"])
    vae.eval()
    vae.requires_grad_(False)

    latents_scale = state["latents_scale"].to(device)
    latents_bias  = state["latents_bias"].to(device)

    return vae, transformer, latents_scale, latents_bias, dict(cfg)


def load_base_checkpoint(base_model: str, device: str):
    """
    Load a base model checkpoint.

    Handles two formats:
    - New format: single self-describing .pt (from run_train.py)
    - Old format: experiment dir with args.json + checkpoints/<step>.pt

    Args:
        base_model: Model zoo key (e.g. "e2ev2_mixview") or direct path to a .pt file.
        device: torch device string.

    Returns:
        vae, transformer, latents_scale, latents_bias, model_config (dict)
    """
    from radcf.utils import get_models
    from dictdot import dictdot

    try:
        ckpt_path = get_checkpoint_path(base_model)
        info      = get_checkpoint_info(base_model)
        train_steps = info.get("train_steps", 400000)
        null_token  = info.get("null_token", 0)
    except ValueError:
        ckpt_path   = base_model
        train_steps = 0
        null_token  = 0

    ckpt_p = Path(ckpt_path)

    if ckpt_p.is_file() and ckpt_p.suffix == ".pt":
        state = torch.load(ckpt_p, map_location=device, weights_only=False)
        if "model_config" in state:
            logger.info(f"Loading self-describing checkpoint: {ckpt_p}")
            return _load_self_describing_base(state, device)

    warnings.warn(
        f"Loading old-format checkpoint (args.json + checkpoints/): {ckpt_path}. "
        "Re-save with run_train.py to produce a self-describing checkpoint. "
        "Old-format support will be removed in a future version.",
        DeprecationWarning,
        stacklevel=2,
    )
    vae, transformer, latents_scale, latents_bias = get_models(
        ckpt_path,
        device,
        train_steps=train_steps,
        label_type="class",
        null_token=null_token,
    )
    vae.eval()
    transformer.eval()

    exp_p    = Path(ckpt_path)
    exp_root = exp_p.parent.parent if exp_p.is_file() else exp_p
    with open(exp_root / "args.json") as f:
        cfg = dictdot(json.load(f))

    vae_type  = cfg.vae
    in_ch     = 4 if vae_type == "f8d4" else 32
    enc_type  = cfg.get("enc_type", "dinov2-vit-b")
    model_cfg = {
        "model":         cfg.model,
        "vae":           vae_type,
        "in_channels":   in_ch,
        "resolution":    cfg.resolution,
        "enc_type":      enc_type,
        "encoder_depth": cfg.get("encoder_depth", 8),
        "z_dims":        cfg.get("z_dims") or _enc_z_dims(enc_type),
        "num_classes":   cfg.get("num_classes", 1000),
        "cfg_prob":      cfg.get("cfg_prob", 0.1),
        "fused_attn":    cfg.get("fused_attn", True),
        "qk_norm":       cfg.get("qk_norm", False),
    }

    return vae, transformer, latents_scale, latents_bias, model_cfg


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------

def _build_checkpoint_dict(
    model_config: dict,
    vae_state_dict: dict,
    latents_scale: torch.Tensor,
    latents_bias: torch.Tensor,
) -> dict:
    """Build the shared checkpoint payload used by both full and LoRA saves."""
    return {
        "model_config": model_config,
        "vae": vae_state_dict,
        "latents_scale": latents_scale.cpu(),
        "latents_bias": latents_bias.cpu(),
    }


def save_full_checkpoint(
    transformer_state_dict: dict,
    vae_state_dict: dict,
    latents_scale: torch.Tensor,
    latents_bias: torch.Tensor,
    model_config: dict,
    save_path,
    training_state: dict | None = None,
):
    """Save a full-parameter checkpoint.

    Produces the same format from both run_train.py and run_finetune.py.
    Pass *training_state* (opt, steps, model_raw, etc.) to make the
    checkpoint resumable.
    """
    checkpoint = _build_checkpoint_dict(
        model_config, vae_state_dict, latents_scale, latents_bias,
    )
    checkpoint["transformer"] = transformer_state_dict
    if training_state:
        checkpoint["training_state"] = training_state
    torch.save(checkpoint, save_path)
    logger.info(f"Saved full checkpoint → {Path(save_path).name}")


def save_lora_checkpoint(
    model,
    vae: torch.nn.Module,
    latents_scale: torch.Tensor,
    latents_bias: torch.Tensor,
    model_config: dict,
    ckpt_dir,
    step: int,
):
    """Save a LoRA checkpoint as a directory.

    Layout::

        {ckpt_dir}/{step:07d}/
            checkpoint.pt        # model_config, vae, latents_scale, latents_bias
            lora_weights/        # PEFT save_pretrained output
    """
    ckpt_step = Path(ckpt_dir) / f"{step:07d}"
    ckpt_step.mkdir(parents=True, exist_ok=True)

    # Shared payload — same keys as save_full_checkpoint
    checkpoint = _build_checkpoint_dict(
        model_config, vae.state_dict(), latents_scale, latents_bias,
    )
    torch.save(checkpoint, ckpt_step / "checkpoint.pt")

    # LoRA adapter weights (model is a PeftModel wrapping the transformer)
    lora_dir = ckpt_step / "lora_weights"
    if hasattr(model, "save_pretrained"):
        model.save_pretrained(lora_dir)
    else:
        lora_dir.mkdir(exist_ok=True)
        torch.save(model.state_dict(), lora_dir / "adapter_model.bin")

    logger.info(f"Saved LoRA checkpoint → {ckpt_step.name}/")


# ---------------------------------------------------------------------------
# Loading (inference)
# ---------------------------------------------------------------------------

def load_full_checkpoint(ckpt_path, schema: list, device: str):
    """Load any full-parameter checkpoint for inference.

    Handles checkpoints from both run_train.py and run_finetune.py.
    Key lookup order: ``transformer`` (new) then ``ema`` (old scratch format).
    If given a directory, the latest checkpoint inside ``checkpoints/`` is
    selected automatically.

    Returns:
        (vae, transformer, latents_scale, latents_bias, model_config)
    """
    from radcf.models.sit import SiT_models
    from radcf.models.autoencoder import vae_models
    from radcf.shared import AttrEmbedder

    ckpt_path = Path(ckpt_path)

    if ckpt_path.is_dir():
        candidates = sorted((ckpt_path / "checkpoints").glob("*.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"No checkpoints found under {ckpt_path / 'checkpoints'}"
            )
        ckpt_path = candidates[-1]
        logger.info(f"Auto-selected checkpoint: {ckpt_path.name}")

    logger.info(f"Loading full checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)

    if "model_config" not in state:
        raise KeyError(
            f"Checkpoint {ckpt_path} has no 'model_config' key. "
            "This checkpoint format is not supported — re-save with model_config embedded."
        )

    cfg = state["model_config"]
    in_channels = cfg["in_channels"]
    latent_size = cfg["resolution"] // (8 if cfg["vae"] == "f8d4" else 16)

    embedder = AttrEmbedder(schema, dropout_prob=0.0, output_mode="sum").to(device)
    transformer = SiT_models[cfg["model"]](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=cfg["num_classes"],
        class_dropout_prob=cfg["cfg_prob"],
        z_dims=cfg["z_dims"],
        encoder_depth=cfg["encoder_depth"],
        fused_attn=cfg["fused_attn"],
        qk_norm=cfg["qk_norm"],
        label_type="custom",
        y_embedder=embedder,
    ).to(device)

    # Unified key is "transformer"; fall back to "ema" for old scratch checkpoints.
    weights = state.get("transformer")
    if weights is None:
        weights = state.get("ema")
        if weights is None:
            raise KeyError("Checkpoint has neither 'transformer' nor 'ema' key.")
        warnings.warn(
            f"Checkpoint {ckpt_path} uses old 'ema' key instead of 'transformer'. "
            "Re-save with run_train.py to update. "
            "Old key support will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
    transformer.load_state_dict(weights, strict=False)
    transformer.eval()
    transformer.requires_grad_(False)

    vae = vae_models[cfg["vae"]]().to(device)
    if "vae" in state:
        vae.load_state_dict(state["vae"])
    else:
        warnings.warn(
            f"Checkpoint {ckpt_path} has no embedded VAE — loading from external path. "
            "Re-save with run_train.py to embed the VAE. "
            "External-VAE support will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2,
        )
        args = state.get("args")
        vae_ckpt_path = getattr(args, "vae_ckpt", None) if args else None
        if vae_ckpt_path is None:
            raise KeyError(
                "Checkpoint has no 'vae' state dict and no vae_ckpt path in args."
            )
        vae_state = torch.load(vae_ckpt_path, map_location=device)
        vae.load_state_dict(vae_state)
    vae.eval()
    vae.requires_grad_(False)

    latents_scale = state["latents_scale"].to(device)
    latents_bias = state["latents_bias"].to(device)

    del state
    gc.collect()

    return vae, transformer, latents_scale, latents_bias, dict(cfg)


def load_lora_checkpoint(lora_dir, schema: list, device: str):
    """Load a LoRA checkpoint for inference.

    The LoRA checkpoint only stores adapter weights, not the frozen base
    transformer.  We must reload the pretrained base (via ``base_model_ref``
    stored in ``model_config``) and then apply the LoRA adapter on top.

    The SA-SPEC VAE is loaded from the checkpoint (it may have been
    co-trained), so we do NOT reuse the base checkpoint's VAE.

    Returns:
        (vae, peft_transformer, latents_scale, latents_bias, model_config)
    """
    from radcf.models.autoencoder import vae_models
    from radcf.shared import AttrEmbedder
    from peft import PeftModel

    lora_dir = Path(lora_dir)
    state = torch.load(
        lora_dir / "checkpoint.pt", map_location=device, weights_only=False
    )

    cfg = state["model_config"]

    # --- Restore the pretrained base transformer ---
    base_ref = cfg.get("base_model_ref")
    if base_ref is None:
        raise KeyError(
            "LoRA checkpoint has no 'base_model_ref' in model_config. "
            "Re-save the checkpoint with the base model reference."
        )

    # load_base_checkpoint returns (vae, transformer, scale, bias, base_cfg).
    # We only need the transformer; VAE / latent stats come from the LoRA
    # checkpoint (SA-SPEC may have co-trained the VAE).
    _, transformer, _, _, _ = load_base_checkpoint(base_ref, device)

    # Replace the base class-conditional y_embedder with the schema-driven
    # AttrEmbedder that LoRA was trained with.
    hidden_size = transformer.pos_embed.shape[-1]
    embedder = AttrEmbedder(
        schema, dropout_prob=0.0, output_mode="sum", hidden_size=hidden_size,
    ).to(device)
    transformer.y_embedder = embedder
    transformer.label_type = "custom"

    # Apply LoRA adapter on top of the pretrained base
    peft_transformer = PeftModel.from_pretrained(
        transformer, lora_dir / "lora_weights"
    )
    peft_transformer.eval()

    # --- VAE from checkpoint (may be SA-SPEC co-trained) ---
    vae = vae_models[cfg["vae"]]().to(device)
    vae.load_state_dict(state["vae"])
    vae.eval()
    vae.requires_grad_(False)

    latents_scale = state["latents_scale"].to(device)
    latents_bias = state["latents_bias"].to(device)

    return vae, peft_transformer, latents_scale, latents_bias, dict(cfg)
