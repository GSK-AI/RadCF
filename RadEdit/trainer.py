"""
LoRARadEditTrainer: PyTorch Lightning trainer for LoRARadEditModel.

Analogous to radcf's GenericTrainer. Handles training, validation,
optimizer config, periodic checkpointing, and counterfactual visualization.
"""

import os
from typing import Optional

import pytorch_lightning as pl
import torch

from radcf.utils import build_vocabs


class LoRARadEditTrainer(pl.LightningModule):
    """
    Lightning wrapper for LoRARadEditModel.

    Delegates all forward logic to model.training_forward(). This trainer
    coordinates optimization, logging, checkpointing, and visualization.

    Args:
        model:                        LoRARadEditModel instance
        learning_rate:                AdamW learning rate
        weight_decay:                 AdamW weight decay
        schema:                       AttrEmbedder schema list (for visualization)
        checkpoint_dir:               Directory for LoRA + embedder checkpoints
        save_every_n_steps:           Checkpoint save frequency (global steps)
        log_visualization_every_n_steps: Visualization save frequency (0 = off)
        noise_timestep:               Noise level for visualization partial inversion
        num_ddim_steps:               DDIM steps used in visualization denoising
    """

    def __init__(
        self,
        model,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-2,
        schema: Optional[list] = None,
        checkpoint_dir: Optional[str] = None,
        save_every_n_steps: int = 5000,
        log_visualization_every_n_steps: int = 5000,
        noise_timestep: int = 500,
        num_ddim_steps: int = 20,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.schema = schema or []
        self.checkpoint_dir = checkpoint_dir
        self.save_every_n_steps = save_every_n_steps
        self.log_visualization_every_n_steps = log_visualization_every_n_steps
        self.noise_timestep = noise_timestep
        self.num_ddim_steps = num_ddim_steps
        self.vocabs = build_vocabs(self.schema)

        self.save_hyperparameters(ignore=["model", "schema"])

    def training_step(self, batch, batch_idx):
        images, metadata, _ = batch
        loss = self.model.training_forward(images, metadata)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        images, metadata, _ = batch
        loss = self.model.training_forward(images, metadata)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.model.get_trainable_parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

    def on_train_batch_end(self, outputs, batch, batch_idx):
        step = self.global_step
        if step == 0:
            return

        # Checkpoint
        if (
            self.checkpoint_dir is not None
            and self.save_every_n_steps > 0
            and step % self.save_every_n_steps == 0
        ):
            ckpt_dir = os.path.join(self.checkpoint_dir, f"step_{step:07d}")
            self.model.save_checkpoint(ckpt_dir)
            print(f"\n[step {step}] Checkpoint saved → {ckpt_dir}")

        # Visualization
        if (
            self.checkpoint_dir is not None
            and self.log_visualization_every_n_steps > 0
            and step % self.log_visualization_every_n_steps == 0
        ):
            from visualization import log_radedit_visualization
            try:
                images, metadata, _ = batch
                images = images.to(self.device)
                metadata = {
                    k: v.to(self.device) if torch.is_tensor(v) else v
                    for k, v in metadata.items()
                }
                log_radedit_visualization(
                    model=self.model,
                    images=images,
                    metas=metadata,
                    iteration=step,
                    schema=self.schema,
                    exp_dir=self.checkpoint_dir,
                    vocabs=self.vocabs,
                    noise_timestep=self.noise_timestep,
                    num_ddim_steps=self.num_ddim_steps,
                )
            except Exception as e:
                print(f"Warning: visualization failed at step {step}: {e}")
