#!/usr/bin/env python3
"""
Batch inference for LoRA RADEdit counterfactual generation (M-SPEC).

Produces the same output structure as run_inference.py
(originals/, null_interventions/, counterfactuals/, reverse_cfs/,
interventions_shard*.csv) but uses ConditionedRadEditPipeline with
DDPM inversion instead of the SiT flow-matching pipeline.

Example:
    python batch_inference.py \
      --dataset chex8 \
      --checkpoint out/chex8_lora/.../epoch=99-step=625000.ckpt \
      --flip_keys "Pleural Effusion" \
      --num_inference_steps 200 \
      --skip_ratio 0.0 \
      --weights 3.0 \
      --shard_rank 0 \
      --num_shards 10 \
      --out_dir results_lora_radedit \
      --save_originals \
      --skip_existing
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Any, Dict, List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

_RADEDIT_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_ROOT = os.environ.get(
    "RESEARCH_ROOT",
    os.path.dirname(_RADEDIT_DIR),  # defaults to opensource/
)
for _p in (_RESEARCH_ROOT, _RADEDIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from custom_datasets import load_custom_dataset_config, build_dataset
from radcf.visualization import build_counterfactual_meta
from radcf.naming import intervention_plan_to_str

from model import LoRARadEditModel
from pipeline import ConditionedRadEditPipeline
from utils import get_radedit_schema


# ---------------------------------------------------------------------------
# Helpers (matching run_inference.py conventions)
# ---------------------------------------------------------------------------

def shard_indices(n: int, shard_rank: int, num_shards: int) -> range:
    per = int(math.ceil(n / num_shards))
    start = shard_rank * per
    end = min(n, start + per)
    return range(start, end)


def build_cli_intervention_plan(
    flip_keys: List[str] | None = None,
    fixed_deltas: List[str] | None = None,
    random_deltas: List[str] | None = None,
) -> Dict[str, Dict[str, Any]]:
    plan: Dict[str, Dict[str, Any]] = {}
    if flip_keys:
        for key in flip_keys:
            plan[key] = {"strategy": "flip"}
    if fixed_deltas:
        for item in fixed_deltas:
            key, delta = item.split(":")
            plan[key.strip()] = {"strategy": "fixed_delta", "delta": float(delta)}
    if random_deltas:
        for item in random_deltas:
            key, low, high = item.split(":")
            plan[key.strip()] = {
                "strategy": "random_delta",
                "random_range": (float(low), float(high)),
            }
    return plan


def _scalarize(v: Any) -> Any:
    if torch.is_tensor(v):
        return v.item() if v.numel() == 1 else v.detach().cpu().tolist()
    return v


def tensor_to_pil(img_tensor: torch.Tensor) -> Image.Image:
    """Convert [C, H, W] or [1, C, H, W] in [-1, 1] -> PIL Image."""
    if img_tensor.dim() == 4:
        img_tensor = img_tensor[0]
    img = (img_tensor.cpu().permute(1, 2, 0).float() + 1) / 2
    img = (img.clamp(0, 1).numpy() * 255).astype("uint8")
    return Image.fromarray(img)


def np_to_pil(np_img: np.ndarray) -> Image.Image:
    """Convert [H, W, 3] numpy in [0, 1] -> PIL Image."""
    return Image.fromarray((np_img * 255).astype("uint8"))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_pipeline(
    checkpoint: str, schema: list, device: str,
    model_id: str = "microsoft/radedit",
):
    """Load model + pipeline for the given adapter type."""
    if checkpoint.endswith(".ckpt"):
        model = LoRARadEditModel.from_lightning_checkpoint(
            ckpt_path=checkpoint, schema=schema, device=device,
            model_id=model_id,
        )
    else:
        model = LoRARadEditModel.from_checkpoint(
            lora_ckpt_dir=checkpoint, schema=schema, device=device,
            model_id=model_id,
        )
    return ConditionedRadEditPipeline.from_model(model, model_id=model_id)


@torch.no_grad()
def run_batch(
    *,
    dataset_name: str,
    checkpoint: str,
    model_id: str = "microsoft/radedit",
    base_model_id: str = "runwayml/stable-diffusion-inpainting",
    intervention_plan: Dict[str, Dict[str, Any]],
    split: str,
    out_dir: str,
    shard_rank: int,
    num_shards: int,
    num_inference_steps: int,
    skip_ratio: float,
    weights: float,
    seed: int | None,
    save_originals: bool,
    skip_existing: bool,
    device: str,
):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Dataset + schema
    config = load_custom_dataset_config(dataset_name)
    schema = get_radedit_schema(config)
    test_ds = build_dataset(config, split)

    idxs = list(shard_indices(len(test_ds), shard_rank, num_shards))
    subset = Subset(test_ds, idxs)
    print(f"[INFO] Shard {shard_rank}/{num_shards}: {len(subset)} samples "
          f"(total {len(test_ds)})")

    # Load model + pipeline
    print(f"[INFO] Loading LoRA model from {checkpoint}...")
    pipe = _load_pipeline(checkpoint, schema, device,
                          model_id=model_id, base_model_id=base_model_id)
    print("[INFO] Pipeline ready.")

    # Output directories
    target_str = intervention_plan_to_str(intervention_plan)
    base_out = os.path.join(out_dir, target_str)

    dirs = {
        "orig": os.path.join(base_out, "originals"),
        "null": os.path.join(base_out, "null_interventions"),
        "cf":   os.path.join(base_out, "counterfactuals"),
        "rev":  os.path.join(base_out, "reverse_cfs"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    csv_path = os.path.join(base_out, f"interventions_shard{shard_rank}.csv")
    csv_header_written = os.path.exists(csv_path) and os.path.getsize(csv_path) > 0

    print(f"[INFO] Output: {base_out}")
    print(f"[INFO] Intervention: {target_str}")
    print(f"[INFO] Steps={num_inference_steps}, skip_ratio={skip_ratio}, "
          f"weights={weights}")

    loader = DataLoader(subset, batch_size=1, shuffle=False, num_workers=0)

    for img_tensor, source_metas, (sample_id,) in tqdm(
        loader, desc=f"Shard {shard_rank}"
    ):
        clean_name = str(sample_id)
        safe_name = clean_name.replace("/", "__")

        # Check skip_existing
        cf_path = os.path.join(dirs["cf"], f"{safe_name}.png")
        if skip_existing and os.path.exists(cf_path):
            continue

        # Move metadata to device
        source_metas = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in source_metas.items()
        }

        # Build counterfactual metadata
        target_metas = build_counterfactual_meta(
            source_meta=source_metas,
            schema=schema,
            intervention_plan=intervention_plan,
        )

        # Convert input to PIL (pipeline expects PIL)
        pil_image = tensor_to_pil(img_tensor).resize((512, 512), Image.LANCZOS)

        # --- Null intervention (reconstruct with source metadata) ---
        result_null = pipe(
            prompt="",
            image=pil_image,
            metadata=source_metas,
            num_inference_steps=num_inference_steps,
            skip_ratio=skip_ratio,
            weights=weights,
            prog_bar=False,
        )
        null_pil = np_to_pil(result_null[0])

        # --- Counterfactual (edit with target metadata) ---
        result_cf = pipe(
            prompt="",
            image=pil_image,
            metadata=target_metas,
            num_inference_steps=num_inference_steps,
            skip_ratio=skip_ratio,
            weights=weights,
            prog_bar=False,
        )
        cf_pil = np_to_pil(result_cf[0])

        # --- Reverse CF (cycle: CF image back with source metadata) ---
        result_rev = pipe(
            prompt="",
            image=cf_pil.copy(),
            metadata=source_metas,
            num_inference_steps=num_inference_steps,
            skip_ratio=skip_ratio,
            weights=weights,
            prog_bar=False,
        )
        rev_pil = np_to_pil(result_rev[0])

        # Save images
        if save_originals:
            p = os.path.join(dirs["orig"], f"{safe_name}.png")
            if not (skip_existing and os.path.exists(p)):
                pil_image.save(p)

        null_pil.save(os.path.join(dirs["null"], f"{safe_name}.png"))
        cf_pil.save(cf_path)
        rev_pil.save(os.path.join(dirs["rev"], f"{safe_name}.png"))

        # CSV row
        meta_keys = sorted(set(source_metas.keys()) | set(target_metas.keys()))
        row: Dict[str, Any] = {"filename": clean_name}
        for key in meta_keys:
            if key in source_metas:
                row[f"{key}_original"] = _scalarize(source_metas[key][0])
            if key in target_metas:
                row[f"{key}_new"] = _scalarize(target_metas[key][0])

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if not csv_header_written:
                writer.writeheader()
                csv_header_written = True
            writer.writerow(row)

    print("[INFO] Done.")


def main():
    p = argparse.ArgumentParser(
        description="Batch inference for LoRA RADEdit counterfactual generation"
    )

    # Model
    p.add_argument("--dataset", type=str, required=True,
                   help="Dataset config name (e.g. chex8, chexpert)")
    p.add_argument("--checkpoint", type=str, required=True,
                   help="Path to .ckpt (Lightning) or directory (PEFT checkpoint)")
    p.add_argument("--model_id", type=str, default="microsoft/radedit",
                   help="HF model ID for the base UNet")
    p.add_argument("--base_model_id", type=str,
                   default="runwayml/stable-diffusion-inpainting",
                   help="HF model ID for the SD pipeline (VAE, tokenizer, etc.)")

    # Intervention
    p.add_argument("--flip_keys", type=str, default="",
                   help="Comma-separated keys to flip, e.g. 'Pleural Effusion'")
    p.add_argument("--fixed_deltas", type=str, default="",
                   help="Comma-separated fixed deltas, e.g. 'Age:0.5'")
    p.add_argument("--random_deltas", type=str, default="",
                   help="Comma-separated random deltas, e.g. 'Age:-0.1:0.1'")

    # Pipeline
    p.add_argument("--num_inference_steps", type=int, default=200)
    p.add_argument("--skip_ratio", type=float, default=0.0,
                   help="0.0 = max editing, 1.0 = no editing")
    p.add_argument("--weights", type=float, default=3.0,
                   help="CFG scale for metadata guidance")
    p.add_argument("--seed", type=int, default=42)

    # Data
    p.add_argument("--split", type=str, default="test",
                   choices=["train", "validation", "test"])
    p.add_argument("--shard_rank", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    # Output
    p.add_argument("--out_dir", type=str, default="results_lora_radedit")
    p.add_argument("--save_originals", action="store_true")
    p.add_argument("--skip_existing", action="store_true")

    args = p.parse_args()

    flip_keys = [k.strip() for k in args.flip_keys.split(",") if k.strip()]
    fixed_deltas = [x.strip() for x in args.fixed_deltas.split(",") if x.strip()]
    random_deltas = [x.strip() for x in args.random_deltas.split(",") if x.strip()]

    intervention_plan = build_cli_intervention_plan(
        flip_keys=flip_keys,
        fixed_deltas=fixed_deltas,
        random_deltas=random_deltas,
    )

    if not intervention_plan:
        print("[WARNING] No intervention specified; only null reconstructions "
              "will be generated.")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_batch(
        dataset_name=args.dataset,
        checkpoint=args.checkpoint,
        model_id=args.model_id,
        base_model_id=args.base_model_id,
        intervention_plan=intervention_plan,
        split=args.split,
        out_dir=args.out_dir,
        shard_rank=args.shard_rank,
        num_shards=args.num_shards,
        num_inference_steps=args.num_inference_steps,
        skip_ratio=args.skip_ratio,
        weights=args.weights,
        seed=args.seed,
        save_originals=args.save_originals,
        skip_existing=args.skip_existing,
        device=device,
    )


if __name__ == "__main__":
    main()
