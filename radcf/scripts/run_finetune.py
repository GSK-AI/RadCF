"""Fine-tune a pretrained base model (flow matching with REPA alignment).

This file was modified from: https://github.com/End2End-Diffusion/REPA-E/blob/main/train_repae.py
See ./LICENSE_originals/LICENSE-REPA-E for the original license.

Loads a pretrained SiT checkpoint and continues training with the same
loop as run_train.py — flow-matching denoising loss plus REPA projection
alignment against frozen vision-encoder features. The only difference
from scratch training is checkpoint initialisation.

Two modes (--mode):
  full   All transformer weights trainable.
  lora   Freeze base weights, apply PEFT LoRA adapters.

Config:  radcf/configs/training_{lora,full}.yaml  (merged with CLI dotlist overrides)

VAE mode (override):
  vae_update.enabled=false  (default)  LDM-only — VAE frozen, transformer trained alone
  vae_update.enabled=true              REPA-E  — VAE + discriminator + transformer updated jointly

Loss:
  loss = denoising_loss + proj_coeff * proj_loss   (+ VAE/disc losses when vae_update)

Checkpoint format:
  full:  {exp_dir}/checkpoints/{N:07d}.pt   (self-describing, same as run_train.py)
  lora:  {exp_dir}/checkpoints/{N:07d}/     (checkpoint.pt + lora_weights/)

Usage:
    accelerate launch radcf/scripts/run_finetune.py \\
        --mode lora --dataset chex8 --base-model e2ev2_mixview \\
        --save-dir out  lora.rank=32

    accelerate launch radcf/scripts/run_finetune.py \\
        --mode full --dataset chex8 --base-model e2ev2_mixview \\
        --save-dir out  vae_update.enabled=true
"""
import argparse
import copy
import logging
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from custom_datasets import get_schema, load_custom_dataset_config, build_dataset
from radcf.configs import load_training_config, TrainingConfig
from radcf.loaders import load_base_checkpoint, save_full_checkpoint, save_lora_checkpoint
from radcf.naming import build_experiment_name
from radcf.shared import AttrEmbedder, LatentEncoder
from radcf.utils import (
    load_encoders, build_vocabs, count_trainable_params,
    preprocess_raw_image, update_ema, denormalize_latents,
)
from radcf.visualization import log_cf_visualization


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    return logging.getLogger(__name__)

logger = logging.getLogger(__name__)


def _get_base_sit(model):
    """Extract the raw SiT from a PEFT-wrapped or plain model."""
    if hasattr(model, 'base_model'):
        if hasattr(model.base_model, 'model'):
            return model.base_model.model
        return model.base_model
    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main(args, overrides=None):
    if overrides is None:
        overrides = []

    # --------------------------------------------------------------------------
    # Config & schema
    # --------------------------------------------------------------------------
    training_config = load_training_config(args.mode)
    if overrides:
        merged = OmegaConf.merge(
            OmegaConf.create(dict(training_config)),
            OmegaConf.from_dotlist(overrides),
        )
        training_config = TrainingConfig(OmegaConf.to_container(merged))

    dataset_config = load_custom_dataset_config(args.dataset)

    # Auto-generate exp_name if not provided
    if not args.exp_name:
        args.exp_name = build_experiment_name(
            dataset_name=args.dataset,
            mode=args.mode,
            base_model_name=args.base_model,
            dataset_ratio=args.ratio,
            overrides=overrides,
        )

    # --------------------------------------------------------------------------
    # Accelerator
    # --------------------------------------------------------------------------
    grad_accum_steps = training_config.training.gradient_accumulation_steps
    mixed_precision  = training_config.training.mixed_precision
    exp_dir = Path(args.save_dir) / args.exp_name
    report_to = training_config.training.get("report_to", "tensorboard")
    accelerator = Accelerator(
        gradient_accumulation_steps=grad_accum_steps,
        mixed_precision=mixed_precision,
        log_with=report_to,
        project_config=ProjectConfiguration(project_dir=str(exp_dir)),
    )
    device = accelerator.device
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # --------------------------------------------------------------------------
    # Experiment directory
    # --------------------------------------------------------------------------
    if accelerator.is_main_process:
        exp_dir.mkdir(parents=True, exist_ok=True)
        logger = create_logger(exp_dir)
        logger.info(f"Experiment dir: {exp_dir}")
        if overrides:
            logger.info(f"Config overrides: {overrides}")

    # --------------------------------------------------------------------------
    # Data
    # --------------------------------------------------------------------------
    train_dataset = build_dataset(dataset_config, split="train")
    if args.ratio < 1.0:
        n = int(len(train_dataset) * args.ratio)
        train_dataset = Subset(train_dataset, range(n))
    local_batch_size = int(training_config.training.batch_size // accelerator.num_processes)
    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=training_config.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if accelerator.is_main_process:
        logger.info(f"Train samples: {len(train_dataset)}")

    # --------------------------------------------------------------------------
    # Load base checkpoint
    # --------------------------------------------------------------------------
    vae, transformer, latents_scale, latents_bias, model_config = load_base_checkpoint(
        args.base_model, device
    )
    latent_encoder = LatentEncoder(vae, latents_bias, latents_scale)  # for visualization

    # --------------------------------------------------------------------------
    # Vision encoders (for REPA alignment)
    # --------------------------------------------------------------------------
    enc_type   = model_config["enc_type"]
    resolution = model_config["resolution"]
    encoders, encoder_types, architectures = load_encoders(enc_type, device, resolution)

    # --------------------------------------------------------------------------
    # Schema & y_embedder for target dataset
    # --------------------------------------------------------------------------
    hidden_size = transformer.pos_embed.shape[-1]
    schema = get_schema(dataset_config, args.mode)
    if accelerator.is_main_process:
        logger.info(f"Schema ({len(schema)} attributes): {[s['name'] for s in schema]}")

    if args.mode == "lora":
        cfg_dropout = training_config.lora.get("cfg_dropout_prob", 0.1)
    else:
        cfg_dropout = training_config.get("full", {}).get("cfg_dropout_prob", 0.1)

    embedder = AttrEmbedder(schema, dropout_prob=cfg_dropout, output_mode="sum", hidden_size=hidden_size).to(device)
    transformer.y_embedder = embedder
    transformer.requires_grad_(True)  # unfreeze (load_base_checkpoint freezes everything)

    # --------------------------------------------------------------------------
    # Apply LoRA or prepare full fine-tuning
    # --------------------------------------------------------------------------
    if args.mode == "lora":
        from peft import LoraConfig, get_peft_model

        lora_config = {
            "rank":            training_config.lora.rank,
            "alpha":           training_config.lora.alpha,
            "target_modules":  training_config.lora.target_modules,
            "bias":            training_config.lora.get("bias", "none"),
            "modules_to_save": training_config.lora.get("modules_to_save", ["y_embedder"]),
        }
        peft_config = LoraConfig(
            r=lora_config["rank"],
            lora_alpha=lora_config["alpha"],
            target_modules=lora_config["target_modules"],
            bias=lora_config["bias"],
            modules_to_save=lora_config["modules_to_save"],
        )
        model = get_peft_model(transformer, peft_config)
        model_config = {
            **model_config,
            "base_model_ref":      args.base_model,
            "lora_rank":           lora_config["rank"],
            "lora_alpha":          lora_config["alpha"],
            "lora_target_modules": lora_config["target_modules"],
        }
        if accelerator.is_main_process:
            model.print_trainable_parameters()
    else:
        model = transformer

    # EMA
    ema = copy.deepcopy(model).to(device)
    ema.requires_grad_(False)
    update_ema(ema, model, decay=0)  # sync EMA to initial weights
    model.eval()
    ema.eval()

    vae_update_enabled = training_config.get("vae_update", {}).get("enabled", False)

    # SA-SPEC: unfreeze VAE so alignment/reconstruction gradients flow
    if vae_update_enabled:
        vae.requires_grad_(True)
        vae.train()

    if accelerator.is_main_process:
        logger.info(f"Model: {args.mode} fine-tuning")
        logger.info(f"Trainable params: {count_trainable_params(model):,}")
        logger.info(f"VAE co-training: {vae_update_enabled}")
        if vae_update_enabled:
            vae_trainable = sum(p.numel() for p in vae.parameters() if p.requires_grad)
            logger.info(f"VAE trainable params: {vae_trainable:,}")

    # --------------------------------------------------------------------------
    # VAE co-training infrastructure
    # --------------------------------------------------------------------------
    if vae_update_enabled:
        from radcf.loss.losses import ReconstructionLoss_Single_Stage

        vae_update_cfg = training_config.vae_update
        loss_cfg = OmegaConf.load(vae_update_cfg.loss_cfg_path)
        vae_loss_fn = ReconstructionLoss_Single_Stage(loss_cfg).to(device)

        if vae_update_cfg.get("disc_pretrained_ckpt") is not None:
            disc_ckpt = torch.load(vae_update_cfg.disc_pretrained_ckpt, map_location="cpu")
            vae_loss_fn.discriminator.load_state_dict(disc_ckpt)
            if accelerator.is_main_process:
                logger.info(f"Loaded discriminator from {vae_update_cfg.disc_pretrained_ckpt}")

        trainable_vae_params = [p for p in vae.parameters() if p.requires_grad]
        if not trainable_vae_params:
            raise RuntimeError(
                "SA-SPEC (vae_update.enabled=true) is set, but no VAE parameters "
                "are trainable. Check that vae.requires_grad_(True) was called."
            )

        optimizer_vae = torch.optim.AdamW(
            trainable_vae_params,
            lr=vae_update_cfg.learning_rate,
            weight_decay=training_config.training.get("weight_decay", 1e-2),
        )
        optimizer_disc = torch.optim.AdamW(
            vae_loss_fn.parameters(),
            lr=vae_update_cfg.disc_learning_rate,
            weight_decay=training_config.training.get("weight_decay", 1e-2),
        )

    # --------------------------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=training_config.training.learning_rate,
        weight_decay=training_config.training.get("weight_decay", 1e-2),
    )

    # Flow / loss config
    flow_cfg   = training_config.flow
    proj_coeff = flow_cfg.proj_coeff
    loss_kwargs_base = dict(
        path_type=flow_cfg.path_type,
        prediction=flow_cfg.prediction,
        weighting=flow_cfg.weighting,
    )
    align_proj_coeff = training_config.get("vae_update", {}).get("align_proj_coeff", 1.5)

    vocabs        = build_vocabs(schema)
    gradient_clip = training_config.training.get("gradient_clip", 1.0)
    log_every     = training_config.training.get("log_every_n_steps", 100)
    vis_every     = training_config.training.get("log_visualization_every_n_steps", 5000)
    ckpt_every    = training_config.training.checkpoint_every_n_steps
    ckpt_dir      = exp_dir / "checkpoints"
    in_channels   = model_config["in_channels"]

    # Save trainable param references for freeze/unfreeze across DDP.
    # Using parameter objects (not names) avoids the DDP "module." prefix
    # mismatch that breaks name-based restore on multi-GPU.
    trainable_model_params = [p for p in model.parameters() if p.requires_grad]

    # --------------------------------------------------------------------------
    # Accelerator prepare
    # --------------------------------------------------------------------------
    if vae_update_enabled:
        model, vae, vae_loss_fn, optimizer, optimizer_vae, optimizer_disc, train_loader = (
            accelerator.prepare(
                model, vae, vae_loss_fn, optimizer, optimizer_vae, optimizer_disc, train_loader
            )
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    if accelerator.is_main_process:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        raw_config = {**vars(copy.deepcopy(args)), **dict(training_config)}
        tracker_config = {
            k: v for k, v in raw_config.items()
            if isinstance(v, (int, float, str, bool))
        }
        accelerator.init_trackers(
            project_name="finetune",
            config=tracker_config,
        )

    # --------------------------------------------------------------------------
    # Training loop
    # --------------------------------------------------------------------------
    global_step = 0
    num_epochs = training_config.training.num_epochs

    for epoch in range(num_epochs):
        model.train()

        for raw_image, meta, _ in tqdm(
            train_loader,
            desc=f"Epoch {epoch + 1}",
            disable=not accelerator.is_local_main_process,
        ):
            raw_image = raw_image.to(device)
            meta = {k: v.to(device) if torch.is_tensor(v) else v for k, v in meta.items()}

            # Extract vision-encoder features (always frozen)
            with torch.no_grad():
                zs = []
                with accelerator.autocast():
                    for encoder, enc_t, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, enc_t)
                        z_feat = encoder.forward_features(raw_image_)
                        if 'mocov3' in enc_t:
                            z_feat = z_feat[:, 1:]
                        if 'dinov2' in enc_t:
                            z_feat = z_feat['x_norm_patchtokens']
                        zs.append(z_feat)

            if vae_update_enabled:
                # ----------------------------------------------------------
                # Three-phase: VAE → discriminator → transformer (same as run_train.py)
                # ----------------------------------------------------------
                vae.train()
                model.train()
                with accelerator.accumulate(model, vae, vae_loss_fn), accelerator.autocast():
                    processed_image = raw_image.float()
                    posterior, z, recon_image = vae(processed_image)

                    # Phase 1: VAE generator + REPA alignment
                    for p in trainable_model_params:
                        p.requires_grad_(False)
                    model.eval()

                    vae_loss, vae_loss_dict = vae_loss_fn(
                        processed_image, recon_image, posterior, global_step, "generator"
                    )
                    vae_loss = vae_loss.mean()

                    vae_align_outputs = model(
                        x=z, y=meta, zs=zs,
                        loss_kwargs={**loss_kwargs_base, "align_only": True},
                    )
                    vae_loss = vae_loss + align_proj_coeff * vae_align_outputs["proj_loss"].mean()

                    accelerator.backward(vae_loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(vae.parameters(), gradient_clip)
                    optimizer_vae.step()
                    optimizer_vae.zero_grad(set_to_none=True)

                    # Phase 2: Discriminator
                    d_loss, d_loss_dict = vae_loss_fn(
                        processed_image, recon_image, posterior, global_step, "discriminator"
                    )
                    d_loss = d_loss.mean()
                    accelerator.backward(d_loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(vae_loss_fn.parameters(), gradient_clip)
                    optimizer_disc.step()
                    optimizer_disc.zero_grad(set_to_none=True)

                    # Phase 3: Transformer (detached z)
                    model.train()
                    for p in trainable_model_params:
                        p.requires_grad_(True)

                    sit_outputs = model(
                        x=z.detach(), y=meta, zs=zs,
                        loss_kwargs={**loss_kwargs_base, "align_only": False},
                    )
                    loss = sit_outputs["denoising_loss"].mean() + proj_coeff * sit_outputs["proj_loss"].mean()
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], gradient_clip
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if accelerator.sync_gradients:
                        unwrapped = accelerator.unwrap_model(model)
                        update_ema(ema, getattr(unwrapped, '_orig_mod', unwrapped))

            else:
                # ----------------------------------------------------------
                # LDM-only: frozen VAE, single transformer update
                # ----------------------------------------------------------
                with torch.no_grad(), accelerator.autocast():
                    processed_image = raw_image.float()
                    posterior, z, _ = vae(processed_image, return_recon=False)

                with accelerator.accumulate(model), accelerator.autocast():
                    model.train()
                    # Keep BN running stats frozen when VAE is not being updated
                    base_sit = _get_base_sit(accelerator.unwrap_model(model))
                    base_sit.bn.eval()

                    sit_outputs = model(
                        x=z, y=meta, zs=zs,
                        loss_kwargs={**loss_kwargs_base, "align_only": False},
                    )
                    loss = sit_outputs["denoising_loss"].mean() + proj_coeff * sit_outputs["proj_loss"].mean()

                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], gradient_clip
                        )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if accelerator.sync_gradients:
                        unwrapped = accelerator.unwrap_model(model)
                        update_ema(ema, getattr(unwrapped, '_orig_mod', unwrapped))

            # ------------------------------------------------------------------
            # Logging
            # ------------------------------------------------------------------
            if accelerator.sync_gradients:
                global_step += 1

                if global_step % log_every == 0:
                    logs = {
                        "loss": accelerator.gather(loss).mean().detach().item(),
                        "denoising_loss": accelerator.gather(sit_outputs["denoising_loss"]).mean().detach().item(),
                        "proj_loss": accelerator.gather(sit_outputs["proj_loss"]).mean().detach().item(),
                    }
                    if vae_update_enabled:
                        logs.update({
                            "vae_loss": accelerator.gather(vae_loss).mean().detach().item(),
                            "d_loss": accelerator.gather(d_loss).mean().detach().item(),
                            "reconstruction_loss": accelerator.gather(vae_loss_dict["reconstruction_loss"].mean()).mean().detach().item(),
                            "perceptual_loss": accelerator.gather(vae_loss_dict["perceptual_loss"].mean()).mean().detach().item(),
                            "kl_loss": accelerator.gather(vae_loss_dict["kl_loss"].mean()).mean().detach().item(),
                        })
                    accelerator.log(logs, step=global_step)
                    if accelerator.is_main_process:
                        logger.info(
                            f"Step {global_step} | loss: {logs['loss']:.4f} "
                            f"| denoise: {logs['denoising_loss']:.4f} "
                            f"| proj: {logs['proj_loss']:.4f}"
                        )

            # ------------------------------------------------------------------
            # Visualization
            # ------------------------------------------------------------------
            if global_step == 1 or (global_step % vis_every == 0 and global_step > 0):
                model.eval()
                if vae_update_enabled:
                    vae.eval()

                unwrapped = accelerator.unwrap_model(model)
                vis_vae = accelerator.unwrap_model(vae) if vae_update_enabled else vae

                try:
                    log_cf_visualization(
                        model=unwrapped,
                        images=raw_image,
                        metas=meta,
                        iteration=global_step,
                        schema=schema,
                        exp_dir=str(exp_dir),
                        latent_encoder=latent_encoder,
                        vae=vis_vae,
                        latents_bias=latent_encoder.latents_bias,
                        latents_scale=latent_encoder.latents_scale,
                        null_token=getattr(_get_base_sit(unwrapped), "null_token", 0),
                        vocabs=vocabs,
                        num_steps=20,
                    )
                except Exception as e:
                    if accelerator.is_main_process:
                        logger.warning(f"Visualization failed: {e}")

            # ------------------------------------------------------------------
            # Checkpoint
            # ------------------------------------------------------------------
            if global_step > 0 and global_step % ckpt_every == 0:
                if accelerator.is_main_process:
                    unwrapped = accelerator.unwrap_model(model)
                    ckpt_vae = accelerator.unwrap_model(vae) if vae_update_enabled else vae

                    # Derive latent stats from EMA BN running statistics
                    ema_sit = _get_base_sit(ema)
                    ema_latents_scale = ema_sit.bn.running_var.rsqrt().view(1, in_channels, 1, 1).cpu()
                    ema_latents_bias = ema_sit.bn.running_mean.view(1, in_channels, 1, 1).cpu()

                    if args.mode == "lora":
                        save_lora_checkpoint(
                            unwrapped, ckpt_vae, ema_latents_scale, ema_latents_bias,
                            model_config, ckpt_dir, global_step,
                        )
                    else:
                        save_full_checkpoint(
                            transformer_state_dict=ema.state_dict(),
                            vae_state_dict=ckpt_vae.state_dict(),
                            latents_scale=ema_latents_scale,
                            latents_bias=ema_latents_bias,
                            model_config=model_config,
                            save_path=ckpt_dir / f"{global_step:07d}.pt",
                            training_state={
                                "model": unwrapped.state_dict(),
                                "opt": optimizer.state_dict(),
                                "steps": global_step,
                            },
                        )

    if accelerator.is_main_process:
        logger.info("Training complete!")
    accelerator.wait_for_everyone()
    accelerator.end_training()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Fine-tune a pretrained base model (lora or full).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--mode",       required=True, choices=["full", "lora"],
                        help="Fine-tuning strategy: 'full' trains all weights; 'lora' injects PEFT LoRA adapters.")
    parser.add_argument("--dataset",    required=True,
                        help="Dataset name (e.g. chex8, chexpert).")
    parser.add_argument("--base-model", required=True, dest="base_model",
                        help="Base model zoo key (e.g. e2ev2_mixview) or path to a self-describing .pt checkpoint.")
    parser.add_argument("--save-dir",   required=True, dest="save_dir",
                        help="Root directory for experiment outputs.")
    parser.add_argument("--exp-name",   type=str, default=None, dest="exp_name",
                        help="Experiment name (auto-generated from naming convention if omitted).")
    parser.add_argument("--ratio",      type=float, default=1.0,
                        help="Fraction of training dataset to use (1.0 = full dataset).")
    parser.add_argument("--seed",       type=int, default=0,
                        help="Random seed for reproducibility.")
    return parser.parse_known_args(input_args)


if __name__ == "__main__":
    args, overrides = parse_args()
    main(args, overrides)
