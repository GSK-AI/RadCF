"""
Configuration System for custom_datasets.

Loads YAML configs and provides schema conversion utilities.
"""

import inspect
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import torchvision.transforms as T
from custom_datasets.dataset_zoo import list_datasets, get_dataset_class, get_dataset_defaults
from custom_datasets.paired_dataset import PairedImageWrapper

__all__ = [
    "Config",
    "load_custom_dataset_config",
    "get_schema",
    "get_group_keys",
    "list_configs",
    "list_datasets",
    "get_dataset_class",
    "build_transforms",
    "build_dataset",
    "build_paired_dataset",
]


class Config(dict):
    """
    Dict wrapper with dot notation access.

    Example:
        config = Config({"model": {"lr": 0.001}})
        print(config.model.lr)  # 0.001
    """
    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict):
                return Config(value)
            return value
        except KeyError:
            raise AttributeError(f"Config has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


def load_custom_dataset_config(dataset_name: str) -> Config:
    """
    Load configuration for a dataset.

    Args:
        dataset_name: Name of dataset (e.g., "chex8", "chexpert")

    Returns:
        Config object with all settings

    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_dir = Path(__file__).parent
    config_path = config_dir / f"{dataset_name}.yaml"

    if not config_path.exists():
        available = [p.stem for p in config_dir.glob("*.yaml")]
        raise FileNotFoundError(
            f"Config '{dataset_name}.yaml' not found.\n"
            f"Available configs: {', '.join(available)}"
        )

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return Config(data)


def get_schema(config: Config, mode: str) -> list:
    """
    Convert config schema to list format for AttrEmbedder.

    Args:
        config: Loaded config object
        mode: "adapter" or "lora" (determines embedding dimensions)

    Returns:
        Schema list in format expected by AttrEmbedder
    """
    schema = config.schema
    schema_list = []

    for attr_name, attr_config in schema.items():
        attr_type = attr_config.get("type")

        if mode == "adapter":
            dim = attr_config.get("adapter_dim", 1152)
        else:  # lora or full
            dim = attr_config.get("lora_dim", 1152)

        entry = {
            "name": attr_name,
            "type": attr_type,
            "dim": dim,
        }

        if attr_type in ["categorical", "group_categorical"]:
            entry["num_classes"] = attr_config.get("num_classes", 2)
            if "vocab" in attr_config:
                entry["vocab"] = attr_config["vocab"]

        if attr_type == "continuous":
            entry["range"] = attr_config.get("range", [0, 1])

        if attr_type == "group_categorical":
            entry["keys"] = attr_config.get("keys", [])

        if "adapter_dim_arg" in attr_config:
            entry["adapter_dim_arg"] = attr_config["adapter_dim_arg"]
        if "adapter_fwd_arg" in attr_config:
            entry["adapter_fwd_arg"] = attr_config["adapter_fwd_arg"]

        schema_list.append(entry)

    return schema_list


def get_group_keys(config: Config, group_name: str) -> list:
    """
    Get keys from a group_categorical attribute.

    Args:
        config: Loaded config object
        group_name: Name of the group (e.g., "Disease")

    Returns:
        List of keys in the group
    """
    if group_name not in config.schema:
        raise ValueError(f"Group '{group_name}' not found in schema")

    attr_config = config.schema[group_name]
    if attr_config.get("type") != "group_categorical":
        raise ValueError(f"Attribute '{group_name}' is not a group_categorical")

    return attr_config.get("keys", [])


def list_configs() -> list:
    """List all available config files."""
    config_dir = Path(__file__).parent
    return sorted([p.stem for p in config_dir.glob("*.yaml")])

def build_transforms(config: Config):
    """
    Build torchvision transforms from config.

    Args:
        config: Config object with dataset.transforms list

    Returns:
        torchvision.transforms.Compose object, or None if not specified
    """
    if "transforms" not in config.dataset:
        return None

    transform_list = []
    for t in config.dataset.transforms:
        t_type = t["type"]

        if t_type == "Resize":
            size = t.get("size")
            if isinstance(size, list):
                size = tuple(size)
            transform_list.append(T.Resize(size))

        elif t_type == "ToTensor":
            transform_list.append(T.ToTensor())

        elif t_type == "Normalize":
            mean = t.get("mean", [0.5, 0.5, 0.5])
            std = t.get("std", [0.5, 0.5, 0.5])
            transform_list.append(T.Normalize(mean=mean, std=std))

        elif t_type == "RandomHorizontalFlip":
            p = t.get("p", 0.5)
            transform_list.append(T.RandomHorizontalFlip(p=p))

        elif t_type == "RandomCrop":
            size = t.get("size")
            if isinstance(size, list):
                size = tuple(size)
            transform_list.append(T.RandomCrop(size))

        elif t_type == "CenterCrop":
            size = t.get("size")
            if isinstance(size, list):
                size = tuple(size)
            transform_list.append(T.CenterCrop(size))

        else:
            raise ValueError(f"Unknown transform type: {t_type}")

    return T.Compose(transform_list)


def build_dataset(config: Config, split: str):
    """
    Build dataset from config.

    Args:
        config: Config object
        split: "train", "validation", or "test"

    Returns:
        Dataset instance
    """
    dataset_class = get_dataset_class(config.dataset.name)
    transforms = build_transforms(config)

    kwargs = {}
    split_name = config.dataset.split.get(split, split)
    kwargs["split"] = split_name

    if "resolution" in config.dataset:
        kwargs["resolution"] = config.dataset.resolution

    if transforms is not None:
        kwargs["transform"] = transforms

    if "data_dir" in config.dataset:
        kwargs["data_dir"] = config.dataset.data_dir

    if "train_csv" in config.dataset and split == "train":
        kwargs["train_csv"] = config.dataset.train_csv
    if "test_csv" in config.dataset and split == "test":
        kwargs["test_csv"] = config.dataset.test_csv
    if "image_root" in config.dataset:
        kwargs["image_root"] = config.dataset.image_root
    if "ratio" in config.dataset:
        kwargs["ratio"] = config.dataset.ratio

    if "use_views" in config.dataset:
        kwargs["use_views"] = config.dataset.use_views

    # Forward schema vocabs to dataset classes that declare <attr>_vocab kwargs.
    if "schema" in config:
        init_params = inspect.signature(dataset_class.__init__).parameters
        schema_lookup = {
            attr.lower().replace(" ", "_"): attr for attr in config.schema.keys()
        }
        for param_name in init_params:
            if not param_name.endswith("_vocab"):
                continue
            schema_name = schema_lookup.get(param_name[: -len("_vocab")])
            if schema_name and "vocab" in config.schema[schema_name]:
                kwargs[param_name] = dict(config.schema[schema_name]["vocab"])

    return dataset_class(**kwargs)


def build_paired_dataset(config: Config, cig_path: str, split: str,):
    base_dataset = build_dataset(config, split)
    return PairedImageWrapper(base_dataset, cig_path)
