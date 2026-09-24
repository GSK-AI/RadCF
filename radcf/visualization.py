"""
Visualization and intervention utilities for radcf.

Provides counterfactual metadata construction and training-time visualization.
"""

import torch
import os
import random
from typing import Any, Dict, Mapping

from radcf.shared import euler_sample, euler_invert


def build_counterfactual_meta(
    source_meta: Mapping[str, Any],
    schema: list[dict],
    intervention_plan: Mapping[str, dict],
) -> Dict[str, Any]:
    """
    Build a counterfactual metadata dictionary from source metadata.

    Args:
        source_meta:
            Original metadata dict, e.g. {"Age": tensor([0.2]), "View": tensor([1])}
        schema:
            List of schema items.
        intervention_plan:
            Dict specifying what to intervene on.

            Expected format:
            {
                "View": {"strategy": "flip"},
                "Age": {"strategy": "fixed_delta", "delta": 0.5},
                "BMI": {"strategy": "random_delta", "random_range": (-0.2, 0.2)},
            }

            For grouped schema items, you can still target the actual sub-key:
            {
                "Edema": {"strategy": "flip"},
                "Age": {"strategy": "fixed_delta", "delta": -0.5},
            }

    Returns:
        cf_meta:
            New metadata dict with interventions applied.
    """
    cf_meta: Dict[str, Any] = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v)
        for k, v in source_meta.items()
    }

    key_to_schema = build_key_to_schema(schema)

    for key, spec in intervention_plan.items():
        if key not in cf_meta:
            continue
        if key not in key_to_schema:
            continue

        schema_item = key_to_schema[key]
        strategy = spec.get("strategy", "flip")
        delta = spec.get("delta", None)
        random_range = spec.get("random_range", None)
        target_class = spec.get("target_class", None)

        cf_meta[key] = intervene_tensor_by_schema(
            val_tensor=cf_meta[key],
            schema_item=schema_item,
            strategy=strategy,
            delta=delta,
            random_range=random_range,
            target_class=target_class,
        )

    return cf_meta


def build_key_to_schema(schema: list[dict]) -> Dict[str, dict]:
    """
    Build mapping from metadata key -> schema item.

    For normal items:
        {"name": "View", ...} -> key "View"

    For grouped items:
        {"name": "Disease", "keys": [...], ...} -> each sub-key points to the same schema item
    """
    key_to_schema: Dict[str, dict] = {}

    for item in schema:
        name = item["name"]
        itype = item["type"]

        if itype in [
            "categorical",
            "categorical_with_unknown",
            "continuous",
        ]:
            key_to_schema[name] = item

        elif itype in [
            "group_categorical",
            "group_categorical_with_unknown",
            "group_continuous",
        ]:
            for k in item.get("keys", []):
                key_to_schema[k] = item

    return key_to_schema


def intervene_tensor_by_schema(
    val_tensor: torch.Tensor,
    schema_item: dict,
    strategy: str = "flip",
    delta: float | None = None,
    random_range: tuple[float, float] | None = None,
    target_class: int | None = None,
) -> torch.Tensor:
    """
    Intervene on one metadata tensor according to schema.

    Categorical:
    - flip:        (val + 1) % num_classes
    - fixed_class: set all values to target_class

    Continuous:
    - flip:        normalized +0.5 with wraparound
    - fixed_delta: val + delta, wrapped into range
    - random_delta: val + U(low, high), wrapped into range
    """
    val = val_tensor.clone()
    itype = schema_item["type"]

    # Categorical
    if "categorical" in itype:
        num_classes = int(schema_item.get("num_classes", 2))
        if num_classes <= 1:
            return val

        if strategy == "flip":
            out = val.long()
            out = (out + 1) % num_classes
        elif strategy == "fixed_class":
            if target_class is None:
                raise ValueError("strategy='fixed_class' requires `target_class`")
            if target_class < 0 or target_class >= num_classes:
                raise ValueError(f"fixed_class target {target_class} out of range [0, {num_classes})")
            out = torch.full_like(val.long(), target_class)
        else:
            raise ValueError(
                f"Categorical type supports 'flip' or 'fixed_class', got '{strategy}'"
            )

        return out.to(val.dtype) if val.is_floating_point() else out

    # Continuous
    if "continuous" in itype:
        min_v, max_v = schema_item.get("range", [0.0, 1.0])
        min_v = float(min_v)
        max_v = float(max_v)
        span = max_v - min_v

        if span <= 0:
            raise ValueError(f"Invalid continuous range: {schema_item.get('range')}")

        val_f = val.to(torch.float32)

        if strategy == "flip":
            val_norm = (val_f - min_v) / span
            out = ((val_norm + 0.5) % 1.0) * span + min_v

        elif strategy == "fixed_delta":
            if delta is None:
                raise ValueError("strategy='fixed_delta' requires `delta`")
            out = _wrap_to_range(val_f + float(delta), min_v, max_v)

        elif strategy == "random_delta":
            if random_range is None:
                raise ValueError(
                    "strategy='random_delta' requires `random_range=(low, high)`"
                )
            low, high = random_range
            rand_delta = torch.empty_like(val_f).uniform_(float(low), float(high))
            out = _wrap_to_range(val_f + rand_delta, min_v, max_v)

        else:
            raise ValueError(f"Unsupported strategy '{strategy}' for continuous type")

        return out.to(val_tensor.dtype)

    return val


def _wrap_to_range(x: torch.Tensor, min_v: float, max_v: float) -> torch.Tensor:
    """
    Wrap x into [min_v, max_v).

    Example for [0,1]:
      0.2 - 0.5 -> 0.7
      0.8 + 0.5 -> 0.3
    """
    width = max_v - min_v
    return ((x - min_v) % width) + min_v


@torch.no_grad()
def log_cf_visualization(
    model,
    images: torch.Tensor,
    metas: Dict[str, torch.Tensor],
    iteration: int,
    schema: list,
    exp_dir: str,
    latent_encoder,
    vae,
    latents_bias: torch.Tensor,
    latents_scale: torch.Tensor,
    null_token: int,
    vocabs: dict,
    num_steps: int = 50,
):
    """
    Visualize counterfactuals during training.

    Logic:
    - Categorical: Flip category
    - Group Categorical: Randomly choose ONE sub-key and flip it
    - Continuous: Add value (+0.5)
    - Group Continuous: Randomly choose ONE sub-key and add value (+0.5)

    Args:
        model: Model instance (AdapterModel, LoRAModel, or FullFTModel)
        images: Input images [B, 3, H, W]
        metas: Metadata dict
        iteration: Training iteration number
        schema: Schema list
        exp_dir: Experiment directory for saving images
        latent_encoder: LatentEncoder instance
        vae: VAE model
        latents_bias: Latent bias tensor
        latents_scale: Latent scale tensor
        null_token: Null token value
        vocabs: Vocabulary dict
        num_steps: Number of ODE steps (default: 50)
    """
    from radcf.utils import save_batch_heatmaps

    model.eval()
    device = images.device

    # 1. Setup Base Sample (First item in batch)
    idx0 = 0
    img0 = images[idx0:idx0 + 1]

    # Clone metas for the single sample
    meta0 = {}
    for k, v in metas.items():
        if isinstance(v, torch.Tensor):
            meta0[k] = v[idx0:idx0 + 1].clone()
        else:
            meta0[k] = v  # scalars/strings

    # 2. Inversion of Original
    x0 = latent_encoder.encode(img0)

    # Build velocity function for inversion
    def velocity_fn_inv(x, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=device, dtype=x.dtype)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        return model.inference_forward(x, t, meta0)

    xT = euler_invert(velocity_fn_inv, x0, num_steps=num_steps)

    # Reconstruct Original (Null Intervention)
    def velocity_fn_null(x, t):
        if not isinstance(t, torch.Tensor):
            t = torch.tensor([t], device=device, dtype=x.dtype)
        if t.ndim == 0:
            t = t.unsqueeze(0)
        return model.inference_forward(x, t, meta0)

    z0_null = euler_sample(velocity_fn_null, xT, num_steps=num_steps)
    null_intervention_imgs = latent_encoder.decode(z0_null)

    save_dir = f"{exp_dir}/training_samples"
    os.makedirs(save_dir, exist_ok=True)

    # 3. Iterate Schema to generate specific Counterfactuals
    for item in schema:
        name = item["name"]
        itype = item["type"]

        if itype in ["categorical", "categorical_with_unknown"]:
            intervention_plan = {
                name: {"strategy": "flip"}
            }
            target_key = name

        elif itype == "continuous":
            intervention_plan = {
                name: {"strategy": "fixed_delta", "delta": 0.5}
            }
            target_key = name

        elif itype in ["group_categorical", "group_categorical_with_unknown"]:
            valid_keys = [k for k in item.get("keys", []) if k in meta0]
            if not valid_keys:
                continue
            target_key = random.choice(valid_keys)
            intervention_plan = {
                target_key: {"strategy": "flip"}
            }

        elif itype == "group_continuous":
            valid_keys = [k for k in item.get("keys", []) if k in meta0]
            if not valid_keys:
                continue
            target_key = random.choice(valid_keys)
            intervention_plan = {
                target_key: {"strategy": "fixed_delta", "delta": 0.5}
            }

        else:
            continue

        cf_metas = build_counterfactual_meta(
            source_meta=meta0,
            schema=schema,
            intervention_plan=intervention_plan,
        )

        # --- Generate CF Image ---
        def velocity_fn_cf(x, t):
            if not isinstance(t, torch.Tensor):
                t = torch.tensor([t], device=device, dtype=x.dtype)
            if t.ndim == 0:
                t = t.unsqueeze(0)
            return model.inference_forward(x, t, cf_metas)

        z0_cf = euler_sample(velocity_fn_cf, xT, num_steps=num_steps)
        cf_img = latent_encoder.decode(z0_cf)

        # --- Save ---
        # Extract scalar values for logging text
        orig_meta_vals = {
            k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else 0
            for k, v in meta0.items()
        }
        cf_meta_vals = {
            k: v.item() if isinstance(v, torch.Tensor) and v.numel() == 1 else 0
            for k, v in cf_metas.items()
        }

        save_batch_heatmaps(
            orig_imgs=img0,
            cf_imgs=cf_img,
            null_intervention_imgs=null_intervention_imgs,
            orig_metas=[orig_meta_vals],
            cf_metas=[cf_meta_vals],
            save_dir=save_dir,
            steps=f"{iteration}_{target_key}",
            do=target_key,
            vocabs=vocabs,
        )

    model.train()
