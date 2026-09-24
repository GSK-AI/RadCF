"""
run_inference.py — Batch counterfactual inference.

Per-image outputs:
  1) originals/            (optional, --save-originals)
  2) null_interventions/   (reconstruction with original metadata)
  3) counterfactuals/      (metadata flipped / shifted per intervention plan)
  4) reverse_cfs/          (CF image encoded back with original metadata)

Output folder structure:
  {out_dir}/{dataset}/{mode}/{exp_name}/{ode_steps}/{ckpt_steps}/{intervention}/
    ├── originals/
    ├── null_interventions/
    ├── counterfactuals/
    └── reverse_cfs/

Also writes interventions_shard{N}.csv per shard with original and new metadata columns.

Usage:
    python radcf/scripts/run_inference.py \\
        --dataset chex8 \\
        --mode lora \\
        --base-model e2ev2_mixview \\
        --flip-keys View \\
        --random-deltas Age:-0.1:0.1 \\
        --ode-steps 150 \\
        --finetune-steps 30000 \\
        --batch-size 16 \\
        --num-workers 8 \\
        --out-dir ./results \\
        --save-originals \\
        --skip-existing
"""
import argparse
import csv
import logging
import math
import os
from pathlib import Path
from typing import Any

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from radcf.configs import get_checkpoint_info
from radcf.inference_pipeline import InferencePipeline
from radcf.naming import intervention_plan_to_str
from radcf.visualization import build_counterfactual_meta
from custom_datasets import (
    load_custom_dataset_config,
    get_schema,
    get_group_keys,
    build_dataset,
    list_configs,
)


logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _to_uint8_chw(t: torch.Tensor) -> torch.Tensor:
    """Convert a tensor in [-1, 1] or [0, 1] to uint8 CHW."""
    t = t.detach()
    if t.is_cuda:
        t = t.cpu()
    if t.ndim == 4:
        t = t[0]
    t = t.float()
    if t.numel() > 0 and t.min().item() < 0.0:
        t = (t + 1.0) / 2.0
    t = t.clamp(0.0, 1.0)
    t = (t * 255.0).round().to(torch.uint8)
    if t.shape[0] not in (1, 3):
        t = t[:1, ...]
    return t


def save_tensor_image(t: torch.Tensor, path: str) -> None:
    T.ToPILImage()(_to_uint8_chw(t)).save(path)


def shard_indices(n: int, shard_rank: int, num_shards: int) -> range:
    if num_shards <= 0:
        raise ValueError("num_shards must be > 0")
    if shard_rank < 0 or shard_rank >= num_shards:
        raise ValueError(f"shard_rank must be in [0, {num_shards - 1}]")
    per   = int(math.ceil(n / num_shards))
    start = shard_rank * per
    end   = min(n, start + per)
    return range(start, end)


def build_cli_intervention_plan(
    flip_keys: list[str] | None = None,
    fixed_deltas: list[str] | None = None,
    random_deltas: list[str] | None = None,
    fixed_classes: list[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    Build a unified intervention plan from CLI string lists.

    fixed_deltas format:   ["Age:0.5", "BMI:-0.2"]
    random_deltas format:  ["Age:-0.1:0.1", "BMI:-0.2:0.2"]
    fixed_classes format:  ["Perturbation:3", "View:0"]
    """
    plan: dict[str, dict[str, Any]] = {}
    if flip_keys:
        for key in flip_keys:
            plan[key] = {"strategy": "flip"}
    if fixed_deltas:
        for item in fixed_deltas:
            parts = item.split(":")
            if len(parts) != 2:
                raise ValueError(f"--fixed-deltas: expected 'Key:delta', got '{item}'")
            plan[parts[0].strip()] = {"strategy": "fixed_delta", "delta": float(parts[1])}
    if random_deltas:
        for item in random_deltas:
            parts = item.split(":")
            if len(parts) != 3:
                raise ValueError(f"--random-deltas: expected 'Key:low:high', got '{item}'")
            plan[parts[0].strip()] = {"strategy": "random_delta", "random_range": (float(parts[1]), float(parts[2]))}
    if fixed_classes:
        for item in fixed_classes:
            parts = item.split(":")
            if len(parts) != 2:
                raise ValueError(f"--fixed-classes: expected 'Key:ClassInt', got '{item}'")
            try:
                target_class = int(parts[1])
            except ValueError:
                raise ValueError(f"--fixed-classes: class must be an integer, got '{parts[1]}' in '{item}'")
            plan[parts[0].strip()] = {"strategy": "fixed_class", "target_class": target_class}
    return plan


def _scalarize(v: Any) -> Any:
    if torch.is_tensor(v):
        return v.item() if v.numel() == 1 else v.detach().cpu().tolist()
    return v


def build_intervention_rows(
    fnames: list[str],
    source_metas: dict[str, torch.Tensor],
    target_metas: dict[str, torch.Tensor],
) -> list[dict[str, Any]]:
    """Build CSV rows: filename, <key>_original, ..., <key>_new, ..."""
    rows: list[dict[str, Any]] = []
    meta_keys = sorted(set(source_metas.keys()) | set(target_metas.keys()))
    for i in range(len(fnames)):
        row: dict[str, Any] = {
            "filename": str(fnames[i])
        }
        for key in meta_keys:
            val = source_metas.get(key)
            row[f"{key}_original"] = _scalarize(val[i] if torch.is_tensor(val) else val) if val is not None else ""
        for key in meta_keys:
            val = target_metas.get(key)
            row[f"{key}_new"] = _scalarize(val[i] if torch.is_tensor(val) else val) if val is not None else ""
        rows.append(row)
    return rows


def save_interventions_csv(csv_path: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# Inference loop
# ---------------------------------------------------------------------------

def main(args):
    # --------------------------------------------------------------------------
    # Config & schema
    # --------------------------------------------------------------------------
    if args.seed is not None:
        torch.manual_seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format='[\033[34m%(asctime)s\033[0m] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    )

    dataset_config = load_custom_dataset_config(args.dataset)
    schema = get_schema(dataset_config, args.mode)
    logger.info(f"Loaded dataset config: {args.dataset}")

    # --------------------------------------------------------------------------
    # Intervention plan
    # --------------------------------------------------------------------------
    flip_keys: list[str] = []
    if args.flip_all_diseases:
        try:
            flip_keys = get_group_keys(dataset_config, "Disease")
        except ValueError:
            logger.warning("No 'Disease' group found in schema")
    elif args.flip_keys:
        flip_keys = [k.strip() for k in args.flip_keys.split(",") if k.strip()]

    fixed_deltas  = [x.strip() for x in args.fixed_deltas.split(",")  if x.strip()]
    random_deltas = [x.strip() for x in args.random_deltas.split(",") if x.strip()]
    fixed_classes = [x.strip() for x in args.fixed_classes.split(",") if x.strip()]

    intervention_plan = build_cli_intervention_plan(
        flip_keys=flip_keys,
        fixed_deltas=fixed_deltas,
        random_deltas=random_deltas,
        fixed_classes=fixed_classes,
    )
    if not intervention_plan:
        logger.warning("No intervention specified; only null reconstructions will be saved.")

    # --------------------------------------------------------------------------
    # Experiment directory
    # --------------------------------------------------------------------------
    if args.mode in ("lora", "full") and args.finetune_steps is None:
        raise ValueError("--finetune-steps is required for lora/full modes.")

    if args.mode == "base":
        finetune_ckpt_dir = None
    else:
        finetune_ckpt_dir = Path(args.save_dir) / args.exp_name
        if not finetune_ckpt_dir.exists():
            raise FileNotFoundError(f"Experiment directory not found: {finetune_ckpt_dir}")

    # --------------------------------------------------------------------------
    # Data
    # --------------------------------------------------------------------------
    dataset = build_dataset(dataset_config, split=args.split)
    if args.ratio < 1.0:
        n = int(len(dataset) * args.ratio)
        dataset = Subset(dataset, range(n))

    if args.single:
        dataset_shard = Subset(dataset, [0])
    else:
        idxs = list(shard_indices(len(dataset), args.shard_rank, args.num_shards))
        dataset_shard = Subset(dataset, idxs)

    loader = DataLoader(
        dataset_shard,
        batch_size=1 if args.single else args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    logger.info(f"Shard samples: {len(dataset_shard)} / total: {len(dataset)}")

    # --------------------------------------------------------------------------
    # Pipeline
    # --------------------------------------------------------------------------
    device      = "cuda" if torch.cuda.is_available() else "cpu"
    base_info   = get_checkpoint_info(args.base_model)
    null_token  = base_info.get("null_token", 0)

    ckpt_dir = finetune_ckpt_dir / "checkpoints" if finetune_ckpt_dir is not None else None
    if args.mode == "scratch":
        logger.info(f"Loading checkpoint from: {finetune_ckpt_dir} (scratch mode, auto-selects latest)")
    elif args.mode == "base":
        logger.info("Using base model (no fine-tune checkpoint)")
    else:
        logger.info(f"Loading checkpoint from: {ckpt_dir} at step {args.finetune_steps}")

    if args.mode == "base":
        pipeline = InferencePipeline.from_checkpoint(
            mode="base",
            base_model=args.base_model,
            schema=schema,
            device=device,
            null_token=null_token,
        )
    elif args.mode == "lora":
        pipeline = InferencePipeline.from_checkpoint(
            mode="lora",
            schema=schema,
            device=device,
            null_token=null_token,
            lora_ckpt_path=str(ckpt_dir / f"{args.finetune_steps:07d}"),
        )
    elif args.mode == "full":
        pipeline = InferencePipeline.from_checkpoint(
            mode="full",
            schema=schema,
            device=device,
            null_token=null_token,
            full_model_ckpt_path=str(ckpt_dir / f"{args.finetune_steps:07d}.pt"),
        )
    elif args.mode == "scratch":
        pipeline = InferencePipeline.from_checkpoint(
            mode="scratch",
            schema=schema,
            device=device,
            null_token=null_token,
            scratch_ckpt_path=str(finetune_ckpt_dir),
        )

    logger.info("Pipeline loaded successfully.")

    # --------------------------------------------------------------------------
    # Output directories
    # --------------------------------------------------------------------------
    ckpt_steps_name = "no_finetune" if args.mode == "base" else f"ckpt_{args.finetune_steps}"
    intervention_str = intervention_plan_to_str(intervention_plan)
    base_out = os.path.join(
        args.out_dir, args.exp_name,
        f"ode_{args.ode_steps}", ckpt_steps_name,
        intervention_str,
    )
    dirs = {
        "orig": os.path.join(base_out, "originals"),
        "null": os.path.join(base_out, "null_interventions"),
        "cf":   os.path.join(base_out, "counterfactuals"),
        "rev":  os.path.join(base_out, "reverse_cfs"),
    }
    for d in dirs.values():
        _ensure_dir(d)

    csv_path           = os.path.join(base_out, f"interventions_shard{args.shard_rank}.csv")
    csv_header_written = False
    logger.info(f"Saving to: {base_out}")

    # --------------------------------------------------------------------------
    # Inference loop
    # --------------------------------------------------------------------------
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Shard {args.shard_rank}"):
            imgs, source_metas, fnames = batch
            imgs         = imgs.to(device)
            source_metas = {k: v.to(device) if torch.is_tensor(v) else v for k, v in source_metas.items()}

            target_metas = build_counterfactual_meta(
                source_meta=source_metas,
                schema=schema,
                intervention_plan=intervention_plan,
            )

            null_batch, cf_batch = pipeline.run_batch_counterfactual(
                images_tensor=imgs,
                source_metas=source_metas,
                target_metas=target_metas,
                num_steps=args.ode_steps,
                cfg_scale=args.cfg_scale,
                guidance_low=args.guidance_low,
                guidance_high=args.guidance_high,
            )
            _, rec_batch = pipeline.run_batch_counterfactual(
                images_tensor=cf_batch.clamp(-1, 1),
                source_metas=target_metas,
                target_metas=source_metas,
                num_steps=args.ode_steps,
                cfg_scale=args.cfg_scale,
                guidance_low=args.guidance_low,
                guidance_high=args.guidance_high,
            )

            for i, fname in enumerate(fnames):
                clean_name = str(fname)

                if args.save_originals:
                    p = os.path.join(dirs["orig"], f"{clean_name}.png")
                    if not (args.skip_existing and os.path.exists(p)):
                        save_tensor_image(imgs[i], p)

                p = os.path.join(dirs["null"], f"{clean_name}.png")
                if not (args.skip_existing and os.path.exists(p)):
                    save_tensor_image(null_batch[i], p)

                p = os.path.join(dirs["cf"], f"{clean_name}.png")
                if not (args.skip_existing and os.path.exists(p)):
                    save_tensor_image(cf_batch[i], p)

                p = os.path.join(dirs["rev"], f"{clean_name}.png")
                if not (args.skip_existing and os.path.exists(p)):
                    save_tensor_image(rec_batch[i], p)

            if args.single:
                break

            rows = build_intervention_rows(
                fnames=fnames,
                source_metas=source_metas,
                target_metas=target_metas,
            )
            if rows:
                with open(csv_path, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    if not csv_header_written:
                        writer.writeheader()
                        csv_header_written = True
                    writer.writerows(rows)

    logger.info("Done.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(
        description="Batch counterfactual inference.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset",    type=str, required=True,
                        help=f"Dataset config name (available: {', '.join(list_configs())})")
    parser.add_argument("--mode",       type=str, required=True,
                        choices=["lora", "full", "scratch", "base"])
    parser.add_argument("--base-model", type=str, required=True)
    parser.add_argument("--save-dir",   type=str, required=True, dest="save_dir",
                        help="Root directory where run_finetune.py saved checkpoints (same --save-dir as training).")
    parser.add_argument("--exp-name",   type=str, required=True, dest="exp_name",
                        help="Experiment name; checkpoint loaded from {save-dir}/{exp-name}/.")
    parser.add_argument("--ratio",      type=float, default=1.0,
                        help="Fraction of dataset to use (1.0 = full dataset).")
    parser.add_argument("--seed",       type=int, default=0,
                        help="Random seed.")

    parser.add_argument("--split",      type=str, default="test",
                        choices=["train", "validation", "test"])
    parser.add_argument("--single",     action="store_true",
                        help="Process only dataset[0]; skip sharding and CSV. Useful for smoke testing.")

    # Intervention specification
    parser.add_argument("--flip-keys",       type=str, default="",
                        help="Comma-separated attribute keys to flip, e.g. 'View,Pneumonia'.")
    parser.add_argument("--flip-all-diseases", action="store_true",
                        help="Flip all keys in the Disease group of the schema.")
    parser.add_argument("--fixed-deltas",    type=str, default="",
                        help="Comma-separated fixed deltas, e.g. 'Age:0.5,BMI:-0.2'.")
    parser.add_argument("--random-deltas",   type=str, default="",
                        help="Comma-separated random delta ranges, e.g. 'Age:-0.1:0.1'.")
    parser.add_argument("--fixed-classes",   type=str, default="",
                        help="Comma-separated fixed class targets for categoricals, e.g. 'Perturbation:3'.")

    # ODE / CFG
    parser.add_argument("--ode-steps",      type=int,   default=100)
    parser.add_argument("--finetune-steps", type=int,   default=None,
                        help="Checkpoint step to load (required for lora/full modes).")
    parser.add_argument("--cfg-scale",      type=float, default=1.0)
    parser.add_argument("--guidance-low",   type=float, default=0.0)
    parser.add_argument("--guidance-high",  type=float, default=1.0)

    # Sharding
    parser.add_argument("--shard-rank",  type=int, default=0)
    parser.add_argument("--num-shards",  type=int, default=1)

    # Performance / output
    parser.add_argument("--batch-size",     type=int, default=32)
    parser.add_argument("--num-workers",    type=int, default=4)
    parser.add_argument("--out-dir",        type=str, default="results_full_cycle",
                        help="Directory where inference outputs (images + CSV) are written.")
    parser.add_argument("--save-originals", action="store_true")
    parser.add_argument("--skip-existing",  action="store_true")

    return parser.parse_args(input_args)


if __name__ == "__main__":
    args = parse_args()
    main(args)
