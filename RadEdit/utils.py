"""
Utilities for LoRA RADEdit training.

Thin layer on top of custom_datasets — handles overriding schema dims
to match the UNet's time_embed_dim.
"""

from typing import List, Dict, Any

from custom_datasets import get_schema


# Known time_embed_dim values: block_out_channels[0] * 4
RADEDIT_TIME_EMBED_DIM = 512   # RADEdit: 128 * 4
SD15_TIME_EMBED_DIM = 1280     # SD1.5:   320 * 4


def get_time_embed_dim(model_id: str) -> int:
    """Infer time_embed_dim from model_id without loading weights."""
    model_id_lower = model_id.lower()
    if "radedit" in model_id_lower:
        return RADEDIT_TIME_EMBED_DIM
    # SD1.5 and variants
    return SD15_TIME_EMBED_DIM


def get_radedit_schema(config, time_embed_dim: int = RADEDIT_TIME_EMBED_DIM) -> List[Dict[str, Any]]:
    """
    Build an AttrEmbedder-compatible schema list from a custom_datasets Config,
    overriding all embedding dims to match the UNet's time_embed_dim.

    custom_datasets.get_schema(mode="lora") reads lora_dim (1152 for SiT).
    The UNet time embedding output varies by backbone, so we override all dims here.

    Args:
        config:         Config loaded via load_custom_dataset_config(name)
        time_embed_dim: UNet time embedding dim (512 for RADEdit, 1280 for SD1.5)

    Returns:
        Schema list ready for AttrEmbedder(schema, output_mode='sum')
    """
    schema = get_schema(config, mode="lora")
    for item in schema:
        item["dim"] = time_embed_dim
    return schema
