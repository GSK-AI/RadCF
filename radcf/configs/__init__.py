import yaml
from pathlib import Path

from .model_zoo import get_checkpoint_path, get_checkpoint_info, list_models

_CONFIG_DIR = Path(__file__).parent

_TRAINING_CONFIG_MAP = {
    "lora": "training_lora.yaml",
    "full": "training_full.yaml",
    "scratch": "training_scratch.yaml",
    "vae": "training_vae.yaml",
}


class TrainingConfig(dict):
    """Dict wrapper with dot notation access for training configs."""
    def __getattr__(self, key):
        try:
            value = self[key]
            if isinstance(value, dict):
                return TrainingConfig(value)
            return value
        except KeyError:
            raise AttributeError(f"TrainingConfig has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def get(self, key, default=None):
        value = super().get(key, default)
        if isinstance(value, dict):
            return TrainingConfig(value)
        return value


def load_training_config(mode: str) -> TrainingConfig:
    """
    Load training configuration for a given mode.

    Args:
        mode: One of "lora", "full", "scratch"

    Returns:
        TrainingConfig object

    Raises:
        ValueError: If mode is not recognized
        FileNotFoundError: If config file doesn't exist
    """
    if mode not in _TRAINING_CONFIG_MAP:
        raise ValueError(
            f"Unknown training mode '{mode}'. "
            f"Available modes: {list(_TRAINING_CONFIG_MAP.keys())}"
        )

    config_path = _CONFIG_DIR / _TRAINING_CONFIG_MAP[mode]
    if not config_path.exists():
        raise FileNotFoundError(f"Training config not found: {config_path}")

    with open(config_path) as f:
        data = yaml.safe_load(f)

    return TrainingConfig(data)


__all__ = [
    "get_checkpoint_path",
    "get_checkpoint_info",
    "list_models",
    "load_training_config",
    "TrainingConfig",
]
