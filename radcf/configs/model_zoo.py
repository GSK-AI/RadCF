"""
Model Checkpoint Registry - maps short names to checkpoint paths.

Centralized location for all base model paths.
When paths change (e.g., moving to new server), update here only.

To use your own checkpoints, update the paths below to point to your
local checkpoint directories.
"""

MODEL_ZOO = {
    # Scratch models (no base model) — path unused in scratch mode
    "scratch": {
        "path": "unused",
        "train_steps": 400000,
        "description": "Placeholder for from-scratch trained models. Path is unused in scratch mode.",
    },

    # Natural image models
    "sit-xl-natural-image": {
        "path": "./checkpoints/base_models/sit-xl-natural-image",
        "train_steps": 400000,
        "description": "SiT-XL with DinoV2 encoder, REPA-E, trained on natural images (end-to-end)"
    },

    # CheXpert models - Mixed views (PA + AP)
    "e2e_mixview": {
        "path": "./checkpoints/base_models/sit-xl-e2e-mixview",
        "train_steps": 400000,
        "description": "SiT-XL trained on full CheXpert (mixed PA/AP views), end-to-end with DinoV2"
    },

    "noe2e_mixview": {
        "path": "./checkpoints/base_models/sit-xl-noe2e-mixview",
        "train_steps": 400000,
        "description": "SiT-XL trained on full CheXpert (mixed views), flow-matching only (no end-to-end)"
    },

    # CheXpert models - Frontal only
    "e2e_frontal": {
        "path": "./checkpoints/base_models/sit-xl-e2e-frontal",
        "train_steps": 400000,
        "description": "SiT-XL trained on full CheXpert frontal only, end-to-end with DinoV2"
    },

    "noe2e_frontal": {
        "path": "./checkpoints/base_models/sit-xl-noe2e-frontal",
        "train_steps": 400000,
        "description": "SiT-XL trained on full CheXpert frontal only, flow-matching only (no end-to-end)"
    },

    "e2ev2_mixview": {
        "path": "./checkpoints/base_models/sit-xl-e2ev2-mixview",
        "train_steps": 400000,
        "description": "SiT-XL trained on full CheXpert mixed view, artifacts removed"
    },

    "e2e_base_nan50_mixview": {
        "path": "./checkpoints/base_models/sit-xl-e2e-nan50-mixview",
        "train_steps": 400000,
        "description": "SiT-XL trained on partial CheXpert frontal only, using split v1, "
                       "artifacts removed, only images with PE labels are used"
    },
}


def get_checkpoint_path(name: str) -> str:
    """
    Get checkpoint path from registry.

    Args:
        name: Short name from MODEL_ZOO keys

    Returns:
        Full path to checkpoint

    Raises:
        ValueError: If model name not found
    """
    if name not in MODEL_ZOO:
        available = list(MODEL_ZOO.keys())
        raise ValueError(
            f"Model '{name}' not in zoo. Available models:\n"
            + "\n".join(f"  - {k}: {MODEL_ZOO[k]['description']}" for k in available)
        )
    return MODEL_ZOO[name]["path"]


def get_checkpoint_info(name: str) -> dict:
    """
    Get full checkpoint info including path, train_steps, description.

    Args:
        name: Short name from MODEL_ZOO keys

    Returns:
        Dict with checkpoint metadata
    """
    if name not in MODEL_ZOO:
        raise ValueError(f"Model '{name}' not in zoo. Use list_models() to see available models.")
    return MODEL_ZOO[name].copy()


def list_models():
    """List all available models with descriptions."""
    return {k: v["description"] for k, v in MODEL_ZOO.items()}
