"""Evaluation configuration for counterfactual image generation (CIG)."""

import sys
from pathlib import Path

# Add research directory to path for imports
research_dir = Path(__file__).parent.parent.resolve()
if str(research_dir) not in sys.path:
    sys.path.insert(0, str(research_dir))

import torch

from custom_datasets import (
    build_dataset,
    build_paired_dataset,
    load_custom_dataset_config,
)

class EvalConfig:
    """
    Evaluation configuration for CIG.

    DESIGN
    ------
    - Dataset zoo config (load_custom_dataset_config) is the ONLY source of truth
    - This class only selects datasets + defines evaluation settings
    - No dataset kwargs duplication here
    """

    EVAL_VERSION = "eval_v4"

    CIG_WORKSPACE = Path(".")
    _JUDGE_CKPT_BASE = str(CIG_WORKSPACE / "checkpoints/judge")
    EVAL_OUTPUT_BASE = CIG_WORKSPACE / "eval_results" / EVAL_VERSION

    # Judge type registry
    # Used for loading, evaluation routing, and metric computation.
    JUDGE_BINARY     = ["gender", "view", "disease_pa", "disease_ap", "device"]
    JUDGE_REGRESSION = ["age"]

    # ============================================================
    # Dataset registry (LIGHTWEIGHT)
    # ============================================================
    # Ckpt paths are derived automatically from judge_configs["arch"] below.
    # Pattern: judge_{task}_{arch}_{dataset_key}.pt

    EVAL_DATASETS = {
        "chex8_effusion": {
            "dataset_name": "chex8_effusion",
            "interventions": ["view", "gender", "disease", "age"],
            "metadata_keys": {
                "gender": "Sex",
                "view": "View",
                "disease": "Effusion",  # targets Effusion specifically, not binary any-finding
                "age": "Age",
            },
            # schema_keys: overrides metadata_keys for YAML vocab lookups only.
            # "Effusion" is not a top-level schema key in chex8.yaml (lives under Disease group_categorical).
            "schema_keys": {
                "disease": "Disease",
            },
            "intv_exp_dirs": {  # subdirs under cig_path/ containing CFs for each intervention
                "gender": "Sex_flip",
                "view": "View_flip",
                "disease": "Effusion_flip",
                "age": "Age_rand_-0.5_0.5",
            },
            "judge_configs": {  # resnet: ImageNet-pretrained
                "gender": {"arch": "resnet18", "lr": 1e-4, "epochs": 3},
                "view": {"arch": "resnet18", "lr": 1e-4, "epochs": 3},
                "disease": {
                    "arch": "densenet121-chex",  # CheXpert-pretrained, no NIH14 overlap
                    "lr": 1e-5,
                    "epochs": 20,
                    "label_smoothing": 0.1,
                    "scheduler": {"factor": 0.2, "patience": 3},
                },
                # "disease": {
                #     "arch": "resnet18", "lr": 1e-4, "epochs": 15,
                #     "label_smoothing": 0.1, "scheduler": {"factor": 0.2, "patience": 3},
                # },
                "age": {
                    "arch": "resnet18", "lr": 1e-4, "epochs": 10,
                    "scheduler": {"factor": 0.2, "patience": 2},
                },
            },
        },

        "chex8_binary": {
            "dataset_name": "chex8_binary",
            "interventions": ["view", "gender", "disease", "age"],
            "metadata_keys": {
                "gender": "Sex",
                "view": "View",
                "disease": "Disease",
                "age": "Age",
            },
            "intv_exp_dirs": {  # subdirs under cig_path/ containing CFs for each intervention
                "gender": "Sex_flip",
                "view": "View_flip",
                # TODO: disease evaluation is currently misleading. The judge is trained on
                # any-finding ("Disease") labels, but the CIGs only flip Effusion. Known bug in the intervention label flips
                "disease": "Effusion_flip",
                "age": "Age_rand_-0.5_0.5",
            },
            "judge_configs": {  # resnet: ImageNet-pretrained
                "gender": {"arch": "resnet18", "lr": 1e-4, "epochs": 3},
                "view": {"arch": "resnet18", "lr": 1e-4, "epochs": 3},
                "disease": {
                    "arch": "densenet121-chex",  # CheXpert-pretrained, no NIH14 overlap
                    "lr": 1e-4,
                    "epochs": 20,
                    "label_smoothing": 0.1,
                    "scheduler": {"factor": 0.2, "patience": 3},
                },
                # "disease": {
                #     "arch": "resnet18", "lr": 1e-4, "epochs": 15,
                #     "label_smoothing": 0.1, "scheduler": {"factor": 0.2, "patience": 3},
                # },
                "age": {
                    "arch": "resnet18", "lr": 1e-4, "epochs": 10,
                    "scheduler": {"factor": 0.2, "patience": 2},
                },
            },
        },

        "chexpert_effusion": {
            "dataset_name": "chexpert_effusion",
            "interventions": ["view", "gender", "age", "disease"],
            "metadata_keys": {
                "gender": "Sex",
                "view": "View",
                "disease": "Pleural Effusion",
                "age": "Age",
            },
            "intv_exp_dirs": {  # subdirs under cig_path/
                "gender": "Sex_flip",
                "view": "View_flip",
                "disease": "Effusion_flip",
                "age": "Age_rand_-0.5_0.5",
            },
            "judge_configs": {  # resnet: ImageNet-pretrained
                "gender": {"arch": "resnet18", "lr": 1e-4, "epochs": 3},
                "view": {"arch": "resnet18", "lr": 1e-4, "epochs": 3},
                "disease": {
                    "arch": "resnet18", "lr": 1e-4, "epochs": 15,
                    "label_smoothing": 0.1, "scheduler": {"factor": 0.2, "patience": 3},
                },
                "age": {
                    "arch": "resnet18", "lr": 1e-4, "epochs": 10,
                    "scheduler": {"factor": 0.2, "patience": 2},
                },
            },
        },

        "brax_effusion": {
            "dataset_name": "brax_effusion",
            "interventions": ["view", "gender", "disease", "device", "age"],
            "metadata_keys": {
                "gender": "Sex",
                "view": "View",
                "disease": "Pleural Effusion",
                "device": "Device",
                "age": "Age",
            },
            # schema_keys: "Pleural Effusion" lives under Disease group_categorical in brax_effusion.yaml
            "schema_keys": {
                "disease": "Disease",
            },
            "intv_exp_dirs": {  # subdirs under cig_path/
                "gender": "Sex_flip",
                "view": "View_flip",
                "disease": "Effusion_flip",
                "device": "Device_flip",
                "age": "Age_rand_-0.5_0.5",
            },
            "judge_configs": {  # resnet: ImageNet-pretrained
                "gender": {"arch": "resnet18", "lr": 1e-4, "epochs": 3},
                "view": {"arch": "resnet18", "lr": 1e-4, "epochs": 3},
                "disease": {
                    "arch": "resnet18", "lr": 1e-4, "epochs": 15,
                    "label_smoothing": 0.1, "scheduler": {"factor": 0.2, "patience": 3},
                },
                "device": {"arch": "resnet18", "lr": 1e-4, "epochs": 5},
                "age": {
                    "arch": "resnet18", "lr": 1e-4, "epochs": 10,
                    "scheduler": {"factor": 0.2, "patience": 2},
                },
            },
        },
    }

    # Derive ckpt paths from judge_configs["arch"] — single source of truth.
    # Pattern: judge_{task}_{arch}_{dataset_key}.pt
    _b = f"{_JUDGE_CKPT_BASE}/{EVAL_VERSION}"
    for _key, _ds in EVAL_DATASETS.items():
        _c = _ds["judge_configs"]
        _ds["judge_gender_ckpt"]     = f"{_b}/judge_gender_{_c['gender']['arch']}_{_key}.pt"
        _ds["judge_view_ckpt"]       = f"{_b}/judge_view_{_c['view']['arch']}_{_key}.pt"
        _ds["judge_disease_pa_ckpt"] = f"{_b}/judge_disease_pa_{_c['disease']['arch']}_{_key}.pt"
        _ds["judge_disease_ap_ckpt"] = f"{_b}/judge_disease_ap_{_c['disease']['arch']}_{_key}.pt"
        _ds["judge_age_ckpt"]        = f"{_b}/judge_age_{_c['age']['arch']}_{_key}.pt"
        if "device" in _c:
            _ds["judge_device_ckpt"] = f"{_b}/judge_device_{_c['device']['arch']}_{_key}.pt"
    del _b, _key, _ds, _c

    # ============================================================
    # Core dataset API
    # ============================================================

    @staticmethod
    def get_intervention_labels(dataset_key: str, intervention_name: str) -> tuple:
        """Return (label_for_0, label_for_1) lowercased, from YAML schema vocab.

        Uses schema_keys[intervention_name] for the YAML lookup if present, falling
        back to metadata_keys. Allows group_categorical entries (e.g. Disease) to
        supply the vocab for individual disease targets (e.g. Effusion).
        """
        cfg = EvalConfig.EVAL_DATASETS[dataset_key]
        schema_key = cfg.get("schema_keys", {}).get(intervention_name) or cfg["metadata_keys"][intervention_name]
        dataset_name = cfg["dataset_name"]
        yaml_config = load_custom_dataset_config(dataset_name)
        vocab = yaml_config["schema"][schema_key]["vocab"]  # {str: int}
        _bool_map = {False: "no", True: "yes"}  # YAML parses No/Yes as booleans
        inv_vocab = {v: _bool_map.get(k, str(k)) for k, v in vocab.items()}
        return inv_vocab[0].lower(), inv_vocab[1].lower()

    @staticmethod
    def get_dataset_config(dataset_key: str) -> dict:
        if dataset_key not in EvalConfig.EVAL_DATASETS:
            raise ValueError(f"Unknown dataset: {dataset_key}")
        return EvalConfig.EVAL_DATASETS[dataset_key]

    @staticmethod
    def get_judge_arch(dataset_key: str, task_name: str) -> str:
        return EvalConfig.EVAL_DATASETS[dataset_key]["judge_configs"][task_name]["arch"]

    @staticmethod
    def build_judge_dataset(dataset_key: str):
        """Build dataset using zoo config (train split)."""
        dataset_name = EvalConfig.EVAL_DATASETS[dataset_key]["dataset_name"]
        config = load_custom_dataset_config(dataset_name)
        return build_dataset(config, split='train')

    @staticmethod
    def build_paired_dataset(dataset_key: str, cig_path: str):
        """Build paired dataset using zoo config (test split)."""
        dataset_name = EvalConfig.EVAL_DATASETS[dataset_key]["dataset_name"]
        config = load_custom_dataset_config(dataset_name)
        return build_paired_dataset(config, cig_path=cig_path, split='test')



    # ============================================================
    # Training / runtime / eval
    # ============================================================

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 0

    BATCH_SIZE = 64
    NUM_WORKERS = 4

    JUDGE_BATCH_SIZE = 64
    JUDGE_NUM_WORKERS = 4
    JUDGE_VAL_PROP = 0.2
    JUDGE_THRESHOLD_CRITERION = {"gender": "youden", "view": "youden", "disease": "youden"}
    CLD_LATENT_DIM = 128

    LPIPS_NET = "alex"