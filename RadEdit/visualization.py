"""
Training visualization for LoRA RADEdit.

4-panel output per schema item:
    Original | Null Intervention | Counterfactual | Diff Heatmap

Reuses radcf's build_counterfactual_meta and save_batch_heatmaps directly.
"""

import os
import random
import torch

from radcf.visualization import build_counterfactual_meta
from radcf.utils import save_batch_heatmaps


@torch.no_grad()
def log_radedit_visualization(
    model,
    images: torch.Tensor,
    metas: dict,
    iteration: int,
    schema: list,
    exp_dir: str,
    vocabs: dict,
    noise_timestep: int = 500,
    num_ddim_steps: int = 20,
):
    """
    Generate and save counterfactual visualizations during training.

    For each schema item:
      1. Encode the first batch image → z0
      2. Add noise at `noise_timestep` → z_t
      3. DDIM-denoise z_t with original metadata → null intervention
      4. DDIM-denoise z_t with flipped metadata  → counterfactual
      5. Save 4-panel: Original | Null | CF | Heatmap
    """
    model.eval()
    device = images.device
    save_dir = os.path.join(exp_dir, "training_samples")
    os.makedirs(save_dir, exist_ok=True)

    # Work on the first sample only
    img0 = images[:1]
    meta0 = {
        k: v[:1].clone() if isinstance(v, torch.Tensor) else v
        for k, v in metas.items()
    }

    # 1. Encode to latent
    z0 = model.vae.encode(img0).latent_dist.sample()
    z0 = z0 * model.vae.config.scaling_factor

    # 2. Add noise at noise_timestep → z_t
    noise = torch.randn_like(z0)
    t_tensor = torch.tensor([noise_timestep], device=device, dtype=torch.long)
    z_t = model.scheduler.add_noise(z0, noise, t_tensor)

    def ddim_denoise(z_start, y_emb, meta_dict=None):
        """Run DDIM denoising from z_start (at noise_timestep) back to z0."""
        cross_attn = model.meta_proj(y_emb)

        model.scheduler.set_timesteps(num_ddim_steps)
        start_idx = 0
        for i, t in enumerate(model.scheduler.timesteps):
            if int(t) <= noise_timestep:
                start_idx = i
                break
        timesteps = model.scheduler.timesteps[start_idx:]

        z = z_start.clone()
        for t in timesteps:
            t_batch = torch.tensor([int(t)], device=device, dtype=torch.long)
            noise_pred = model.conditioned_unet(
                z, t_batch, cross_attn
            ).sample
            z = model.scheduler.step(noise_pred, t, z).prev_sample
        return z

    def decode(z):
        z_dec = z / model.vae.config.scaling_factor
        img = model.vae.decode(z_dec).sample
        return img.clamp(-1, 1)

    # 3. Null intervention: denoise with original metadata
    y_orig = model.embedder(meta0, training=False)
    z0_null = ddim_denoise(z_t, y_orig, meta0)
    img_null = decode(z0_null)

    # 4. Per-schema-item counterfactuals
    for item in schema:
        itype = item["type"]
        name = item["name"]

        if itype in ["categorical", "categorical_with_unknown"]:
            intervention = {name: {"strategy": "flip"}}
            target_key = name

        elif itype == "continuous":
            intervention = {name: {"strategy": "fixed_delta", "delta": 0.5}}
            target_key = name

        elif itype in ["group_categorical", "group_categorical_with_unknown"]:
            valid_keys = [k for k in item.get("keys", []) if k in meta0]
            if not valid_keys:
                continue
            target_key = random.choice(valid_keys)
            intervention = {target_key: {"strategy": "flip"}}

        else:
            continue

        cf_meta = build_counterfactual_meta(
            source_meta=meta0,
            schema=schema,
            intervention_plan=intervention,
        )

        y_cf = model.embedder(cf_meta, training=False)
        z0_cf = ddim_denoise(z_t, y_cf, cf_meta)
        img_cf = decode(z0_cf)

        # Scalar meta values for heatmap titles
        orig_vals = {
            k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else 0
            for k, v in meta0.items()
        }
        cf_vals = {
            k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else 0
            for k, v in cf_meta.items()
        }

        save_batch_heatmaps(
            orig_imgs=img0,
            null_intervention_imgs=img_null,
            cf_imgs=img_cf,
            orig_metas=[orig_vals],
            cf_metas=[cf_vals],
            save_dir=save_dir,
            steps=f"{iteration}_{target_key}",
            do=target_key,
            vocabs=vocabs,
        )

    model.train()
