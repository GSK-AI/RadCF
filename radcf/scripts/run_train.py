"""From-scratch training for SiT + VAE (flow matching with REPA alignment).

This file is modified from: https://github.com/End2End-Diffusion/REPA-E/blob/main/train_repae.py
See ./LICENSE_originals/LICENSE-REPA-E for the original license.

Initialises a SiT transformer and VAE from random/pretrained weights and
trains with the standard flow-matching objective plus REPA projection
alignment against frozen vision-encoder features.

The training loop is shared with run_finetune.py — the only difference
is that this script initialises from scratch, whereas run_finetune.py
loads a pretrained checkpoint.

Config:  radcf/configs/training_scratch.yaml  (merged with CLI dotlist overrides)

VAE mode (override):
  vae_update.enabled=false  (default)  LDM-only — VAE frozen, SiT trained alone
  vae_update.enabled=true              REPA-E  — VAE + discriminator + SiT updated jointly

Loss:
  loss = denoising_loss + proj_coeff * proj_loss   (+ VAE/disc losses when vae_update)

Checkpoints are self-describing: they embed model_config, latents_scale,
and latents_bias so run_inference.py can load them without args.json.

Usage:
    accelerate launch radcf/scripts/run_train.py \\
        --dataset chexpert --label-type custom --save-dir out \\
        vae_update.enabled=true
"""
import argparse
import copy
import logging
import math
import os
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from torch.utils.data import DataLoader, Subset

from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed

from custom_datasets import load_custom_dataset_config, build_dataset, get_schema
from radcf.configs import load_training_config, TrainingConfig
from radcf.models.autoencoder import vae_models
from radcf.models.sit import SiT_models, get_hidden_size
from radcf.naming import build_experiment_name
from radcf.shared.sampling import euler_sample
from radcf.shared.embedder import AttrEmbedder
from radcf.loaders import save_full_checkpoint
from radcf.utils import (
    load_encoders, denormalize_latents, count_trainable_params, build_vocabs,
    preprocess_raw_image, update_ema,
)


def create_logger(logging_dir):
    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
    )
    return logging.getLogger(__name__)


def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def main(args, overrides=None):
    if overrides is None:
        overrides = []

    # --------------------------------------------------------------------------
    # Config
    # --------------------------------------------------------------------------
    config = load_training_config("scratch")
    if overrides:
        merged = OmegaConf.merge(
            OmegaConf.create(dict(config)),
            OmegaConf.from_dotlist(overrides),
        )
        config = TrainingConfig(OmegaConf.to_container(merged))

    # Auto-generate exp_name if not provided
    if not args.exp_name:
        mode = "scratch_cond" if args.label_type == "custom" else "scratch_uncond"
        args.exp_name = build_experiment_name(
            dataset_name=args.dataset,
            mode=mode,
            dataset_ratio=args.ratio,
            overrides=overrides,
        )

    # --------------------------------------------------------------------------
    # Accelerator
    # --------------------------------------------------------------------------
    exp_dir = Path(args.save_dir) / args.exp_name
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision,
        log_with=config.training.report_to,
        project_config=ProjectConfiguration(project_dir=str(exp_dir)),
    )

    if accelerator.is_main_process:
        exp_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = exp_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(exp_dir / "args.json", 'w') as f:
            json.dump(vars(args), f, indent=4)
        with open(exp_dir / "config.json", 'w') as f:
            json.dump(dict(config), f, indent=4)
        logger = create_logger(exp_dir)
        logger.info(f"Experiment dir: {exp_dir}")
    device = accelerator.device
    if torch.backends.mps.is_available():
        accelerator.native_amp = False
    if args.seed is not None:
        set_seed(args.seed + accelerator.process_index)

    # --------------------------------------------------------------------------
    # VAE geometry
    # --------------------------------------------------------------------------
    if config.vae.type == "f8d4":
        assert config.training.resolution % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
        latent_size = config.training.resolution // 8
        in_channels = 4
    elif config.vae.type == "f16d32":
        assert config.training.resolution % 16 == 0, "Image size must be divisible by 16 (for the VAE encoder)."
        latent_size = config.training.resolution // 16
        in_channels = 32
    else:
        raise NotImplementedError()

    # --------------------------------------------------------------------------
    # Config & schema
    # --------------------------------------------------------------------------
    dataset_config = load_custom_dataset_config(args.dataset)
    if args.label_type == "custom":
        schema = get_schema(dataset_config, mode='lora')
        vocabs = build_vocabs(schema)

    # --------------------------------------------------------------------------
    # Data
    # --------------------------------------------------------------------------
    train_dataset = build_dataset(dataset_config, split="train")
    if args.ratio < 1.0:
        n = int(len(train_dataset) * args.ratio)
        train_dataset = Subset(train_dataset, range(n))
    local_batch_size = int(config.training.batch_size // accelerator.num_processes)
    train_loader = DataLoader(
        train_dataset,
        batch_size=local_batch_size,
        shuffle=True,
        num_workers=config.training.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    if accelerator.is_main_process:
        logger.info(f"Dataset contains {len(train_dataset):,} images ({args.dataset})")

    # --------------------------------------------------------------------------
    # Encoders, embedder, SiT
    # --------------------------------------------------------------------------
    encoders, encoder_types, architectures = load_encoders(config.model.enc_type, device, config.training.resolution)
    z_dims = [encoder.embed_dim for encoder in encoders]

    block_kwargs = {"fused_attn": config.model.fused_attn, "qk_norm": config.model.qk_norm}
    if args.label_type == "custom":
        hidden_size = get_hidden_size(config.model.backbone)
        embedder = AttrEmbedder(schema, dropout_prob=config.flow.cfg_prob, output_mode='sum', hidden_size=hidden_size).to(device)
        model_kwargs = dict(
            input_size=latent_size,
            in_channels=in_channels,
            num_classes=1,
            class_dropout_prob=config.flow.cfg_prob,
            z_dims=z_dims,
            encoder_depth=config.model.encoder_depth,
            label_type='custom',
            y_embedder=embedder,
            **block_kwargs,
        )
    else:
        model_kwargs = dict(
            input_size=latent_size,
            in_channels=in_channels,
            num_classes=1,
            class_dropout_prob=config.flow.cfg_prob,
            z_dims=z_dims,
            encoder_depth=config.model.encoder_depth,
            label_type='class',
            **block_kwargs,
        )
    if config.vae_update.enabled:
        # bn_momentum only matters when the VAE is being co-trained (BN stats are live)
        model_kwargs["bn_momentum"] = config.model.bn_momentum

    model = SiT_models[config.model.backbone](**model_kwargs).to(device)
    ema = copy.deepcopy(model).to(device)
    requires_grad(ema, False)

    # --------------------------------------------------------------------------
    # VAE
    # --------------------------------------------------------------------------
    vae = vae_models[config.vae.type]().to(device)
    vae_ckpt = torch.load(config.vae.ckpt, map_location=device)
    # strict=False when vae_update because the projection layer may be absent in the base ckpt
    vae.load_state_dict(vae_ckpt, strict=not config.vae_update.enabled)
    del vae_ckpt

    # Initialise SiT BN normalisation layer from the pre-computed latent stats file.
    latents_stats = torch.load(config.vae.ckpt.replace(".pt", "-latents-stats.pt"))
    latents_scale_init = latents_stats["latents_scale"].squeeze().to(device)
    latents_bias_init = latents_stats["latents_bias"].squeeze().to(device)
    model.init_bn(latents_bias=latents_bias_init, latents_scale=latents_scale_init)

    if accelerator.use_distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    if accelerator.is_main_process:
        logger.info(f"SiT Parameters: {sum(p.numel() for p in model.parameters()):,}")
        if config.vae_update.enabled:
            logger.info(f"Total trainable params in VAE: {count_trainable_params(vae)}")

    # --------------------------------------------------------------------------
    # VAE co-training infrastructure (REPA-E only)
    # --------------------------------------------------------------------------
    if config.vae_update.enabled:
        from radcf.loss.losses import ReconstructionLoss_Single_Stage

        loss_cfg = OmegaConf.load(config.vae_update.loss_cfg_path)
        vae_loss_fn = ReconstructionLoss_Single_Stage(loss_cfg).to(device)

        if config.vae_update.disc_pretrained_ckpt is not None:
            disc_ckpt = torch.load(config.vae_update.disc_pretrained_ckpt, map_location=device)
            vae_loss_fn.discriminator.load_state_dict(disc_ckpt)
            if accelerator.is_main_process:
                logger.info(f"Loaded discriminator from {config.vae_update.disc_pretrained_ckpt}")

    # --------------------------------------------------------------------------
    # Optimizers
    # --------------------------------------------------------------------------
    if config.training.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        betas=(config.training.adam_beta1, config.training.adam_beta2),
        weight_decay=config.training.adam_weight_decay,
        eps=config.training.adam_epsilon,
    )
    if config.vae_update.enabled:
        optimizer_vae = torch.optim.AdamW(
            vae.parameters(),
            lr=config.vae_update.learning_rate,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            weight_decay=config.training.adam_weight_decay,
            eps=config.training.adam_epsilon,
        )
        optimizer_loss_fn = torch.optim.AdamW(
            vae_loss_fn.parameters(),
            lr=config.vae_update.disc_learning_rate,
            betas=(config.training.adam_beta1, config.training.adam_beta2),
            weight_decay=config.training.adam_weight_decay,
            eps=config.training.adam_epsilon,
        )

    # --------------------------------------------------------------------------
    # EMA init + optional resume
    # --------------------------------------------------------------------------
    update_ema(ema, model, decay=0)  # sync EMA to initial weights
    model.eval()
    ema.eval()
    if config.vae_update.enabled:
        vae.eval()

    global_step = 0
    if args.resume_step > 0:
        ckpt_name = str(args.resume_step).zfill(7) + '.pt'
        ckpt_path = f'{args.continue_train_exp_dir}/checkpoints/{ckpt_name}'
        ckpt = torch.load(ckpt_path, map_location='cpu')
        ts = ckpt['training_state']
        model.load_state_dict(ts['model'])
        ema.load_state_dict(ckpt['transformer'])
        optimizer.load_state_dict(ts['opt'])
        if config.vae_update.enabled:
            vae.load_state_dict(ckpt['vae'])
            vae_loss_fn.discriminator.load_state_dict(ts['discriminator'])
            optimizer_vae.load_state_dict(ts['opt_vae'])
            optimizer_loss_fn.load_state_dict(ts['opt_disc'])
        global_step = ts['steps']

    # --------------------------------------------------------------------------
    # Model compilation
    # --------------------------------------------------------------------------
    torch._dynamo.config.cache_size_limit = 64
    torch._dynamo.config.accumulated_cache_size_limit = 512

    if config.model.compile:
        model = torch.compile(model, backend="inductor", mode="default")
        if config.vae_update.enabled:
            vae = torch.compile(vae, backend="inductor", mode="default")
            vae_loss_fn = torch.compile(vae_loss_fn, backend="inductor", mode="default")

    # --------------------------------------------------------------------------
    # Accelerator prepare
    # --------------------------------------------------------------------------
    if config.vae_update.enabled:
        model, vae, vae_loss_fn, optimizer, optimizer_vae, optimizer_loss_fn, train_loader = accelerator.prepare(
            model, vae, vae_loss_fn, optimizer, optimizer_vae, optimizer_loss_fn, train_loader
        )
    else:
        model, optimizer, train_loader = accelerator.prepare(
            model, optimizer, train_loader
        )

    if accelerator.is_main_process:
        raw_config = {**vars(copy.deepcopy(args)), **dict(config)}
        tracker_config = {
            k: v for k, v in raw_config.items()
            if isinstance(v, (int, float, str, bool))
        }
        accelerator.init_trackers(
            project_name="gradient-pass-through",
            config=tracker_config,
            init_kwargs={"wandb": {"name": f"{args.exp_name}"}},
        )

    # --------------------------------------------------------------------------
    # model_config embedded in every checkpoint (self-describing)
    # --------------------------------------------------------------------------
    model_config = {
        "model": config.model.backbone,
        "vae": config.vae.type,
        "in_channels": in_channels,
        "resolution": config.training.resolution,
        "enc_type": config.model.enc_type,
        "encoder_depth": config.model.encoder_depth,
        "z_dims": z_dims,
        "num_classes": 1,
        "label_type": args.label_type,
        "cfg_prob": config.flow.cfg_prob,
        "fused_attn": config.model.fused_attn,
        "qk_norm": config.model.qk_norm,
    }

    # --------------------------------------------------------------------------
    # Training loop
    # --------------------------------------------------------------------------
    progress_bar = tqdm(
        range(0, config.training.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )
    xT = torch.randn((local_batch_size, in_channels, latent_size, latent_size), device=device)

    for epoch in range(config.training.num_epochs):
        model.train()

        for raw_image, meta, _ in train_loader:
            raw_image = raw_image.to(device)
            meta = {k: v.to(device) if torch.is_tensor(v) else v for k, v in meta.items()}
            y = meta if args.label_type == "custom" else torch.zeros(raw_image.shape[0], dtype=torch.long, device=device)

            # Extract vision-foundation features (always frozen)
            with torch.no_grad():
                zs = []
                with accelerator.autocast():
                    for encoder, encoder_type, arch in zip(encoders, encoder_types, architectures):
                        raw_image_ = preprocess_raw_image(raw_image, encoder_type)
                        z = encoder.forward_features(raw_image_)
                        if 'mocov3' in encoder_type:
                            z = z[:, 1:]
                        if 'dinov2' in encoder_type:
                            z = z['x_norm_patchtokens']
                        zs.append(z)

            if config.vae_update.enabled:
                # --------------------------------------------------------------
                # REPA-E: three-phase update (VAE → discriminator → SiT)
                # --------------------------------------------------------------
                vae.train()
                model.train()
                with accelerator.accumulate([model, vae, vae_loss_fn]), accelerator.autocast():
                    processed_image = raw_image.float()
                    posterior, z, recon_image = vae(processed_image)

                    loss_kwargs = dict(
                        path_type=config.flow.path_type,
                        prediction=config.flow.prediction,
                        weighting=config.flow.weighting,
                    )
                    time_input = None
                    noises = None

                    # Phase 1: VAE generator update
                    # Block REPA gradient from flowing into SiT during VAE step.
                    requires_grad(model, False)
                    model.eval()

                    vae_loss, vae_loss_dict = vae_loss_fn(
                        processed_image, recon_image, posterior, global_step, "generator"
                    )
                    vae_loss = vae_loss.mean()

                    loss_kwargs["align_only"] = True
                    vae_align_outputs = model(
                        x=z, y=y, zs=zs, loss_kwargs=loss_kwargs,
                        time_input=time_input, noises=noises,
                    )
                    vae_loss = vae_loss + config.vae_update.align_proj_coeff * vae_align_outputs["proj_loss"].mean()
                    # Reuse the sampled time/noise in the SiT step below.
                    time_input = vae_align_outputs["time_input"]
                    noises = vae_align_outputs["noises"]

                    accelerator.backward(vae_loss)
                    if accelerator.sync_gradients:
                        grad_norm_vae = accelerator.clip_grad_norm_(vae.parameters(), config.training.max_grad_norm)
                    optimizer_vae.step()
                    optimizer_vae.zero_grad(set_to_none=True)

                    # Phase 2: discriminator update
                    d_loss, d_loss_dict = vae_loss_fn(
                        processed_image, recon_image, posterior, global_step, "discriminator"
                    )
                    d_loss = d_loss.mean()
                    accelerator.backward(d_loss)
                    if accelerator.sync_gradients:
                        grad_norm_disc = accelerator.clip_grad_norm_(vae_loss_fn.parameters(), config.training.max_grad_norm)
                    optimizer_loss_fn.step()
                    optimizer_loss_fn.zero_grad(set_to_none=True)

                    # Phase 3: SiT update
                    # Detach z so diffusion loss does not back-propagate into the VAE.
                    requires_grad(model, True)
                    model.train()

                    loss_kwargs["align_only"] = False
                    sit_outputs = model(
                        x=z.detach(), y=y, zs=zs, loss_kwargs=loss_kwargs,
                        time_input=time_input, noises=noises,
                    )
                    sit_loss = sit_outputs["denoising_loss"].mean() + config.flow.proj_coeff * sit_outputs["proj_loss"].mean()
                    accelerator.backward(sit_loss)
                    if accelerator.sync_gradients:
                        grad_norm_sit = accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if accelerator.sync_gradients:
                        unwrapped_model = accelerator.unwrap_model(model)
                        update_ema(ema, getattr(unwrapped_model, '_orig_mod', unwrapped_model))

            else:
                # --------------------------------------------------------------
                # LDM-only: frozen VAE, single SiT update
                # --------------------------------------------------------------
                with torch.no_grad(), accelerator.autocast():
                    processed_image = raw_image.float()
                    posterior, z, _ = vae(processed_image, return_recon=False)

                with accelerator.accumulate(model), accelerator.autocast():
                    model.train()
                    # Keep BN running stats frozen — VAE is not being updated.
                    accelerator.unwrap_model(model).bn.eval()

                    loss_kwargs = dict(
                        weighting=config.flow.weighting,
                        path_type=config.flow.path_type,
                        prediction=config.flow.prediction,
                        align_only=False,
                    )
                    sit_outputs = model(x=z, y=y, zs=zs, loss_kwargs=loss_kwargs)

                    sit_loss = sit_outputs["denoising_loss"].mean() + config.flow.proj_coeff * sit_outputs["proj_loss"].mean()
                    accelerator.backward(sit_loss)
                    if accelerator.sync_gradients:
                        grad_norm_sit = accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    if accelerator.sync_gradients:
                        unwrapped_model = accelerator.unwrap_model(model)
                        update_ema(ema, getattr(unwrapped_model, '_orig_mod', unwrapped_model))

            # ------------------------------------------------------------------
            # Logging
            # ------------------------------------------------------------------
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                logs = {
                    "sit_loss": accelerator.gather(sit_loss).mean().detach().item(),
                    "denoising_loss": accelerator.gather(sit_outputs["denoising_loss"]).mean().detach().item(),
                    "proj_loss": accelerator.gather(sit_outputs["proj_loss"]).mean().detach().item(),
                    "grad_norm_sit": accelerator.gather(grad_norm_sit).mean().detach().item(),
                    "epoch": epoch,
                }
                if config.vae_update.enabled:
                    logs.update({
                        "vae_loss": accelerator.gather(vae_loss).mean().detach().item(),
                        "reconstruction_loss": accelerator.gather(vae_loss_dict["reconstruction_loss"].mean()).mean().detach().item(),
                        "perceptual_loss": accelerator.gather(vae_loss_dict["perceptual_loss"].mean()).mean().detach().item(),
                        "kl_loss": accelerator.gather(vae_loss_dict["kl_loss"].mean()).mean().detach().item(),
                        "weighted_gan_loss": accelerator.gather(vae_loss_dict["weighted_gan_loss"].mean()).mean().detach().item(),
                        "discriminator_factor": accelerator.gather(vae_loss_dict["discriminator_factor"].mean()).mean().detach().item(),
                        "gan_loss": accelerator.gather(vae_loss_dict["gan_loss"].mean()).mean().detach().item(),
                        "d_weight": accelerator.gather(vae_loss_dict["d_weight"].mean()).mean().detach().item(),
                        "grad_norm_vae": accelerator.gather(grad_norm_vae).mean().detach().item(),
                        "vae_align_loss": accelerator.gather(vae_align_outputs["proj_loss"].mean()).mean().detach().item(),
                        "d_loss": accelerator.gather(d_loss).mean().detach().item(),
                        "grad_norm_disc": accelerator.gather(grad_norm_disc).mean().detach().item(),
                        "logits_real": accelerator.gather(d_loss_dict["logits_real"].mean()).mean().detach().item(),
                        "logits_fake": accelerator.gather(d_loss_dict["logits_fake"].mean()).mean().detach().item(),
                        "lecam_loss": accelerator.gather(d_loss_dict["lecam_loss"].mean()).mean().detach().item(),
                    })
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            # ------------------------------------------------------------------
            # Checkpoint
            # ------------------------------------------------------------------
            if global_step % config.training.checkpoint_every_n_steps == 0 and global_step > 0:
                if accelerator.is_main_process:
                    unwrapped_model = getattr(accelerator.unwrap_model(model), '_orig_mod', accelerator.unwrap_model(model))
                    unwrapped_vae = getattr(accelerator.unwrap_model(vae), '_orig_mod', accelerator.unwrap_model(vae))

                    # Derive latent normalisation stats from EMA BN running statistics.
                    latents_scale = ema.bn.running_var.rsqrt().view(1, in_channels, 1, 1).cpu()
                    latents_bias = ema.bn.running_mean.view(1, in_channels, 1, 1).cpu()

                    training_state = {
                        "model": unwrapped_model.state_dict(),
                        "opt": optimizer.state_dict(),
                        "steps": global_step,
                        "args": args,
                    }
                    if config.vae_update.enabled:
                        unwrapped_disc = getattr(accelerator.unwrap_model(vae_loss_fn), '_orig_mod', accelerator.unwrap_model(vae_loss_fn)).discriminator
                        training_state.update({
                            "discriminator": unwrapped_disc.state_dict(),
                            "opt_vae": optimizer_vae.state_dict(),
                            "opt_disc": optimizer_loss_fn.state_dict(),
                        })

                    save_full_checkpoint(
                        transformer_state_dict=ema.state_dict(),
                        vae_state_dict=unwrapped_vae.state_dict(),
                        latents_scale=latents_scale,
                        latents_bias=latents_bias,
                        model_config=model_config,
                        save_path=f"{checkpoint_dir}/{global_step:07d}.pt",
                        training_state=training_state,
                    )

            # ------------------------------------------------------------------
            # Periodic sample visualisation
            # ------------------------------------------------------------------
            if global_step == 1 or (global_step % config.training.log_visualization_every_n_steps == 0 and global_step > 0):
                model.eval()
                if config.vae_update.enabled:
                    vae.eval()

                if args.label_type == "custom":
                    # Counterfactual visualization for custom labels
                    from radcf.visualization import log_cf_visualization
                    from radcf.shared import LatentEncoder

                    unwrapped_model = getattr(accelerator.unwrap_model(model), '_orig_mod', accelerator.unwrap_model(model))
                    unwrapped_vae = getattr(accelerator.unwrap_model(vae), '_orig_mod', accelerator.unwrap_model(vae))

                    # Build a fresh LatentEncoder from live BN stats
                    vis_stats = unwrapped_model.extract_latents_stats()
                    vis_scale = vis_stats['latents_scale'].view(1, in_channels, 1, 1)
                    vis_bias = vis_stats['latents_bias'].view(1, in_channels, 1, 1)
                    latent_encoder = LatentEncoder(unwrapped_vae, vis_bias, vis_scale)

                    try:
                        log_cf_visualization(
                            model=unwrapped_model,
                            images=raw_image,
                            metas=meta,
                            iteration=global_step,
                            schema=schema,
                            exp_dir=str(exp_dir),
                            latent_encoder=latent_encoder,
                            vae=latent_encoder.vae,
                            latents_bias=latent_encoder.latents_bias,
                            latents_scale=latent_encoder.latents_scale,
                            null_token=getattr(unwrapped_model, "null_token", 0),
                            vocabs=vocabs,
                            num_steps=100,
                        )
                    except Exception as e:
                        if accelerator.is_main_process:
                            logger.warning(f"CF visualization failed: {e}")
                else:
                    # Unconditional: euler sample + save_image
                    with torch.no_grad():
                        unwrapped_model = accelerator.unwrap_model(model)
                        sample_y = torch.zeros(xT.shape[0], dtype=torch.long, device=device)
                        velocity_fn = lambda x, t: unwrapped_model.inference(
                            x, t.unsqueeze(0).expand(x.shape[0]), y=sample_y
                        )
                        samples = euler_sample(velocity_fn, xT, num_steps=100).to(torch.float32)
                        latents_stats_sample = unwrapped_model.extract_latents_stats()
                        sample_scale = latents_stats_sample['latents_scale'].view(1, in_channels, 1, 1)
                        sample_bias = latents_stats_sample['latents_bias'].view(1, in_channels, 1, 1)
                        samples = accelerator.unwrap_model(vae).decode(
                            denormalize_latents(samples, sample_scale, sample_bias)
                        ).sample
                        samples = (samples + 1) / 2.
                    out_samples = accelerator.gather(samples.to(torch.float32))
                    if accelerator.is_main_process:
                        from torchvision.utils import save_image
                        sample_dir = str(exp_dir / "samples")
                        os.makedirs(sample_dir, exist_ok=True)
                        save_image(
                            out_samples.clamp(0, 1),
                            os.path.join(sample_dir, f"{global_step:07d}.png"),
                            nrow=round(math.sqrt(out_samples.shape[0])),
                        )
                        logger.info("Generating EMA samples done.")

            if global_step >= config.training.max_train_steps:
                break
        if global_step >= config.training.max_train_steps:
            break

    model.eval()
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logger.info("Done!")
    accelerator.end_training()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="From-scratch SiT training (REPA-E or LDM-only).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",    type=str, required=True,
                        help="Dataset config name.")
    parser.add_argument("--label-type", type=str, required=True,
                        choices=["custom", "unconditional"],
                        help="'custom' uses AttrEmbedder schema; 'unconditional' uses null class.")
    parser.add_argument("--save-dir",   type=str, default="exps",
                        help="Root directory for experiment outputs.")
    parser.add_argument("--exp-name",   type=str, default=None, dest="exp_name",
                        help="Experiment name (auto-generated from naming convention if omitted).")
    parser.add_argument("--ratio",      type=float, default=1.0,
                        help="Fraction of training dataset to use (1.0 = full dataset).")
    parser.add_argument("--seed",       type=int, default=0,
                        help="Random seed for reproducibility.")
    parser.add_argument("--resume-step",            type=int, default=0,
                        help="Step to resume training from.")
    parser.add_argument("--continue-train-exp-dir", type=str, default=None,
                        help="Experiment directory to resume from.")
    return parser.parse_known_args(input_args)


if __name__ == "__main__":
    args, overrides = parse_args()
    main(args, overrides)
