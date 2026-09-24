"""
Dataset Registry - maps short names to dataset classes.

When you add a new dataset:
1. Add entry here with class path and default kwargs
2. Create config YAML
3. Done!
"""

DATASET_ZOO = {
    # X-rays
    "chex8": {
        "class": "custom_datasets.xrays.chex8.SimpleChexray",
        "description": "ChestX-ray8 dataset (no unknown labels)",
        "default_kwargs": {}
    },

    "chex8_binary": {
        "class": "custom_datasets.xrays.chex8.BinarySimpleChexray",
        "description": "NIH14 / ChestX-ray8 binary dataset",
        "default_kwargs": {}
    },

    "chexpert": {
        "class": "custom_datasets.xrays.chexpert.CheXpertMetaDataset",
        "description": "CheXpert dataset",
        "default_kwargs": {}
    },

    "chexpert_frontal": {
        "class": "custom_datasets.xrays.chexpert.CheXpertMetaDataset",
        "description": "CheXpert dataset frontal only view",
        "default_kwargs": {}
    },

    "brax": {
        "class": "custom_datasets.xrays.brax.BraxDataset",
        "description": "BRAX chest X-ray dataset",
        "default_kwargs": {}
    },

    "brax_frontal": {
        "class": "custom_datasets.xrays.brax.BraxDataset",
        "description": "Frontal only BRAX chest X-ray dataset",
        "default_kwargs": {}
    },

    # Effusion-focused configs (same dataset classes, effusion-specific YAML schemas)
    "chex8_effusion": {
        "class": "custom_datasets.xrays.chex8.SimpleChexray",
        "description": "ChestX-ray8 with Effusion as primary intervention target",
        "default_kwargs": {}
    },

    "chexpert_effusion": {
        "class": "custom_datasets.xrays.chexpert.CheXpertMetaDataset",
        "description": "CheXpert frontal with Pleural Effusion as primary intervention target",
        "default_kwargs": {}
    },

    "brax_effusion": {
        "class": "custom_datasets.xrays.brax.BraxDataset",
        "description": "BRAX frontal with Pleural Effusion as primary intervention target",
        "default_kwargs": {}
    },
}


def get_dataset_class(name: str):
    """
    Get dataset class from registry.

    Args:
        name: Short name from DATASET_ZOO keys

    Returns:
        Dataset class

    Raises:
        ValueError: If dataset name not found
    """
    if name not in DATASET_ZOO:
        available = list(DATASET_ZOO.keys())
        raise ValueError(
            f"Dataset '{name}' not in zoo. Available datasets:\n"
            + "\n".join(f"  - {k}: {DATASET_ZOO[k]['description']}" for k in available)
        )

    entry = DATASET_ZOO[name]
    class_path = entry["class"]

    # Dynamically import
    module_path, class_name = class_path.rsplit(".", 1)
    import importlib
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def get_dataset_defaults(name: str):
    """
    Get default kwargs for a dataset.

    Args:
        name: Short name from DATASET_ZOO keys

    Returns:
        Dict of default kwargs
    """
    if name not in DATASET_ZOO:
        return {}
    return DATASET_ZOO[name].get("default_kwargs", {}).copy()


def list_datasets():
    """List all available datasets with descriptions."""
    return {k: v["description"] for k, v in DATASET_ZOO.items()}
