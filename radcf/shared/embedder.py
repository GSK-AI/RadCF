"""
Clean attribute embedder without monkey-patching.

Uses explicit output_mode parameter instead of runtime modification.
"""

import torch
import torch.nn as nn
from typing import List, Dict, Any


class AttrEmbedder(nn.Module):
    """
    Schema-driven embedding module for metadata conditioning.

    Clean design with explicit output mode - no monkey-patching!

    Supported schema types:
        - "categorical": Uses nn.Embedding
        - "continuous": Uses nn.Linear
        - "group": Uses MLP for multi-key conditioning

    Args:
        schema: List of attribute definitions
        dropout_prob: CFG dropout probability
        output_mode: 'dict' (for adapter) or 'sum' (for lora/full)
    """

    def __init__(
        self,
        schema: List[Dict[str, Any]],
        dropout_prob: float = 0.1,
        output_mode: str = 'dict',
        hidden_size: int = None,
        group_mlp_hidden: int = 128,
    ):
        super().__init__()

        assert output_mode in ['dict', 'sum'], \
            f"output_mode must be 'dict' or 'sum', got '{output_mode}'"

        self.schema = schema
        self.dropout_prob = float(dropout_prob)
        self.output_mode = output_mode

        self.encoders = nn.ModuleDict()
        self.null_embeddings = nn.ParameterDict()

        for item in schema:
            name = item["name"]
            itype = item["type"]
            dim = int(hidden_size) if hidden_size is not None else int(item["dim"])

            # Learnable null embedding for CFG
            self.null_embeddings[name] = nn.Parameter(torch.randn(1, dim) * 0.02)

            # Build encoder based on type
            if "group" in itype:
                keys = item.get("keys", [])
                if not keys:
                    raise ValueError(f"Group item '{name}' must define 'keys' in schema.")

                # MLP: (B, K) -> Hidden -> Dim
                self.encoders[name] = nn.Sequential(
                    nn.Linear(len(keys), group_mlp_hidden),
                    nn.ReLU(inplace=True),
                    nn.Linear(group_mlp_hidden, dim),
                )

            elif "categorical" in itype:
                # Embedding: (B,) -> (B, Dim)
                self.encoders[name] = nn.Embedding(int(item["num_classes"]), dim)

            elif itype == "continuous":
                # Linear: (B, 1) -> (B, Dim)
                self.encoders[name] = nn.Linear(1, dim)

            else:
                raise ValueError(f"Unknown schema type: {itype}")

    def forward(self, metas: Dict[str, Any], training: bool = False):
        """
        Forward pass respects output_mode.

        Args:
            metas: Metadata dict
            training: Whether in training mode (for CFG dropout)

        Returns:
            If output_mode='dict': Dict[str, Tensor] of embeddings
            If output_mode='sum': Tensor of summed embeddings
        """
        if self.output_mode == 'sum':
            return self.forward_sum(metas, training)
        else:
            return self.forward_dict(metas, training)

    def forward_dict(
        self, metas: Dict[str, Any], training: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Return dictionary of individual embeddings.

        Used by adapter mode where each embedding is passed separately.
        """
        embeddings: Dict[str, torch.Tensor] = {}

        # Infer batch size and device
        try:
            first_val = next(iter(metas.values()))
            device = first_val.device
            B = first_val.shape[0]
        except StopIteration:
            return embeddings

        # Global CFG dropout mask
        drop_mask = None
        if training and self.dropout_prob > 0:
            drop_mask = (torch.rand(B, 1, device=device) < self.dropout_prob).float()

        for item in self.schema:
            name = item["name"]
            itype = item["type"]

            # Encode
            if "group" in itype:
                keys = item["keys"]
                val_list = [metas[k].to(device).float().view(B) for k in keys]
                val = torch.stack(val_list, dim=1)
                emb = self.encoders[name](val)

            elif "categorical" in itype:
                if name not in metas:
                    continue
                val = metas[name].to(device).long().view(B)
                emb = self.encoders[name](val)

            elif itype == "continuous":
                if name not in metas:
                    continue
                val = metas[name].to(device).float().view(B, 1)
                emb = self.encoders[name](val)

            else:
                continue

            # Apply CFG dropout
            if drop_mask is not None and emb is not None:
                null = self.null_embeddings[name]
                emb = (1.0 - drop_mask) * emb + drop_mask * null

            if emb is not None:
                embeddings[name] = emb

        return embeddings

    def forward_sum(
        self, metas: Dict[str, Any], training: bool = False
    ) -> torch.Tensor:
        """
        Return summed embedding.

        Used by LoRA/Full modes where embedding is added to time embedding.
        """
        embs = self.forward_dict(metas, training=training)

        if not embs:
            raise ValueError(
                "No embeddings generated. Provide valid 'metas' matching the schema."
            )

        # Check all embeddings have same dimension
        dims = {v.shape[-1] for v in embs.values()}
        if len(dims) != 1:
            raise ValueError(
                f"forward_sum requires all embedding dims to match, got dims={sorted(dims)}. "
                "Use output_mode='dict' if using Adapter with mixed dimensions."
            )

        return sum(embs.values())

    def get_null_embeddings_dict(self, device, batch_size=1) -> Dict[str, torch.Tensor]:
        """
        Get dictionary of learnable null embeddings for CFG.

        Args:
            device: Target device
            batch_size: Batch size

        Returns:
            Dict mapping attribute names to null embeddings [batch_size, dim]
        """
        embeddings = {}
        for item in self.schema:
            name = item["name"]
            null_emb = self.null_embeddings[name].to(device)
            embeddings[name] = null_emb.expand(batch_size, -1)
        return embeddings

    def get_null_embedding_sum(self, device, batch_size=1) -> torch.Tensor:
        """
        Get summed null embedding for CFG unconditional path.

        Args:
            device: Target device
            batch_size: Batch size

        Returns:
            Summed null embedding [batch_size, dim]
        """
        null_dict = self.get_null_embeddings_dict(device, batch_size)
        return sum(null_dict.values())
