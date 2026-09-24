"""
Centralized naming utilities for radcf experiments.

Canonical experiment name:
    exp_{dataset}_{mode}_{base_model}_{ratio}{suffix}

Examples:
    exp_chex8_lora_e2ev2_mixview_1.0
    exp_chex8_lora_e2ev2_mixview_1.0_lora-rank-32
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


def build_suffix_from_dotlist(dotlist: Iterable[str] | None) -> str:
    """
    Convert OmegaConf-style CLI overrides into a suffix.

    Example:
        ["lora.rank=32", "training.batch_size=16"]
        -> "_lora-rank-32_training-batch_size-16"
    """
    if not dotlist:
        return ""

    parts: List[str] = []
    for item in dotlist:
        if "=" not in item:
            continue

        key, value = item.split("=", 1)
        key = key.replace(".", "-")
        value = str(value).replace("/", "-")
        parts.append(f"{key}-{value}")

    if not parts:
        return ""

    return "_" + "_".join(parts)


def build_experiment_name(
    *,
    dataset_name: str,
    mode: str,
    base_model_name: str | None = None,
    dataset_ratio: float | str | None = 1.0,
    overrides: Iterable[str] | None = None,
) -> str:
    """
    Canonical experiment name:
        exp_{dataset}_{mode}_{base_model}_{ratio}{suffix}

    When *base_model_name* is ``None`` (from-scratch training), the tag
    ``from_scratch`` is used instead.
    """
    tag = base_model_name or "from_scratch"
    ratio = 1.0 if dataset_ratio is None else dataset_ratio
    suffix = build_suffix_from_dotlist(overrides)
    return f"exp_{dataset_name}_{mode}_{tag}_{ratio}{suffix}"



def intervention_plan_to_str(intervention_plan: Dict[str, Dict[str, Any]]) -> str:
    """
    Convert intervention plan to a readable folder name.
    """
    if not intervention_plan:
        return "no_intervention"

    parts: List[str] = []
    for key, spec in intervention_plan.items():
        strategy = spec.get("strategy", "flip")

        if strategy == "flip":
            parts.append(f"{key}_flip")
        elif strategy == "fixed_delta":
            parts.append(f"{key}_fixed_{spec['delta']}")
        elif strategy == "random_delta":
            low, high = spec["random_range"]
            parts.append(f"{key}_rand_{low}_{high}")
        elif strategy == "fixed_class":
            parts.append(f"{key}_fixed_class_{spec['target_class']}")
        else:
            parts.append(f"{key}_{strategy}")

    return "__".join(parts)


