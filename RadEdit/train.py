"""
Training entry point for RADEdit with LoRA metadata conditioning (M-SPEC).

Usage:
    python train.py --dataset chex8 --config configs/lora.yaml --output_dir runs/chex8_lora

--dataset is any name registered in custom_datasets (chex8, chexpert_frontal, …).
The dataset YAML in custom_datasets/ drives data_dir, transforms, and schema.
--config is a YAML with adapter hyperparams and training settings.

Set RESEARCH_ROOT env var if custom_datasets / radcf are not on PYTHONPATH.
"""

import argparse
import os
import sys

import torch
import yaml
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_RADEDIT_DIR = os.path.dirname(os.path.abspath(__file__))
_RESEARCH_ROOT = os.environ.get(
    "RESEARCH_ROOT",
    os.path.dirname(_RADEDIT_DIR),  # defaults to opensource/
)
for _p in (_RESEARCH_ROOT, _RADEDIT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from custom_datasets import load_custom_dataset_config, build_dataset

from model import LoRARadEditModel
from trainer import LoRARadEditTrainer
from utils import get_radedit_schema, get_time_embed_dim


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Train RADEdit with metadata conditioning")

    parser.add_argument("--dataset", required=True,
                        help="Dataset name registered in custom_datasets (e.g. chex8, chexpert_frontal)")
    parser.add_argument("--config", default=None,
                        help="Path to adapter + training config YAML")
    parser.add_argument("--lora_config", default=None,
                        help="(Deprecated, use --config) Path to LoRA config YAML")
    parser.add_argument("--output_dir", required=True,
                        help="Directory for checkpoints and logs")
    parser.add_argument("--model_id", default="microsoft/radedit",
                        help="HF model ID for the RADEdit UNet (subfolder=unet)")
    parser.add_argument("--base_model_id", default="runwayml/stable-diffusion-inpainting",
                        help="HF model ID for SD 1.x pipeline (VAE, text_encoder, tokenizer, scheduler)")

    # Training overrides
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)

    # Hardware
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--precision", default="32", choices=["32", "16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    pl.seed_everything(args.seed)

    # Load adapter + training config
    config_path = args.config or args.lora_config
    if config_path is None:
        raise ValueError("Must provide --config (or deprecated --lora_config)")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    train_cfg = cfg["training"]

    adapter_params = cfg["lora"]

    # Apply CLI overrides
    if args.batch_size is not None:
        train_cfg["batch_size"] = args.batch_size
    if args.num_workers is not None:
        train_cfg["num_workers"] = args.num_workers
    if args.num_epochs is not None:
        train_cfg["num_epochs"] = args.num_epochs
    if args.learning_rate is not None:
        train_cfg["learning_rate"] = args.learning_rate

    # Dataset + schema from custom_datasets
    config = load_custom_dataset_config(args.dataset)
    time_embed_dim = get_time_embed_dim(args.model_id)
    schema = get_radedit_schema(config, time_embed_dim=time_embed_dim)
    print(f"Using time_embed_dim={time_embed_dim} for model_id={args.model_id}")
    print(f"Schema: {[item['name'] for item in schema]}")

    train_ds = build_dataset(config, "train")
    val_ds   = build_dataset(config, "val")
    print(f"Train: {len(train_ds)} samples | Val: {len(val_ds)} samples")

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=True,
    )

    # Model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LoRARadEditModel.from_pretrained(
        model_id=args.model_id,
        base_model_id=args.base_model_id,
        schema=schema,
        lora_config=adapter_params,
        device=device,
    )

    # Lightning trainer
    os.makedirs(args.output_dir, exist_ok=True)
    lit_trainer = LoRARadEditTrainer(
        model=model,
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg.get("weight_decay", 1e-2),
        schema=schema,
        checkpoint_dir=args.output_dir,
        save_every_n_steps=train_cfg.get("save_every_n_steps", 5000),
        log_visualization_every_n_steps=train_cfg.get("log_visualization_every_n_steps", 5000),
    )

    precision_map = {"32": 32, "16": 16, "bf16": "bf16"}
    trainer = pl.Trainer(
        max_epochs=train_cfg["num_epochs"],
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=args.gpus,
        precision=precision_map[args.precision],
        gradient_clip_val=train_cfg.get("gradient_clip", 1.0),
        log_every_n_steps=10,
        default_root_dir=args.output_dir,
        callbacks=[LearningRateMonitor(logging_interval="step")],
    )

    trainer.fit(lit_trainer, train_loader, val_loader)

    final_dir = os.path.join(args.output_dir, "final")
    model.save_checkpoint(final_dir)
    print(f"\nTraining complete. Final checkpoint → {final_dir}")


if __name__ == "__main__":
    main()
