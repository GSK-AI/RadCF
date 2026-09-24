"""
CIG evaluation pipeline.

Design principles:
- Dataset zoo is the single source of truth
- No dataset introspection or hidden parsing
- Minimal, explicit data flow
- Clean separation: config → data → evaluation

USAGE:
python eval/scripts/eval_pipeline.py \
    --cig_path /path/to/experiment_root \
    --test_dataset chex8_binary \
    --intervention view
    # --output_dir defaults to <workspace>/eval_results/<eval_version>/<experiment_rel_path>/
"""

from pathlib import Path
import argparse
import json

import glob

import pandas as pd
import torch
from torch.utils.data import DataLoader

from eval.config import EvalConfig
from eval.metric_descriptions import build_metric_descriptions
from eval.models.encoders import FrozenFeatureEncoder
from eval.models.judges import load_binary_classifier, load_regression_judge
from eval.metrics.utils import seed_all

# Evaluators
from eval.eval_composition import evaluate_composition
from eval.eval_effectiveness import (
    evaluate_binary_effectiveness,
    evaluate_continuous_effectiveness,
    evaluate_judge_calibration,
)
from eval.eval_realism import evaluate_realism
from eval.eval_minimality import evaluate_minimality
from eval.eval_reversibility import evaluate_reversibility


# ============================================================
# Utilities
# ============================================================

def save_metric_documentation(output_dir: Path, config):
    """Save a single metric descriptions file covering all configured interventions."""
    doc_path = output_dir / "metric_descriptions.json"
    if doc_path.exists():
        print(f"Metric documentation already exists, skipping → {doc_path}")
        return
    with open(doc_path, "w") as f:
        json.dump(build_metric_descriptions(config), f, indent=2)
    print(f"Saved metric documentation → {doc_path}")


def parse_method_name(cig_path: str) -> str:
    p = Path(cig_path).resolve().parts
    if len(p) >= 5:
        return "-".join(p[-5:])
    return Path(cig_path).name


def load_intervention_targets(cf_dir: Path) -> dict:
    """Load per-sample intervention targets from sharded CSVs in the CF directory.

    Expects files matching interventions_shard*.csv, each with a 'filename' column
    and one or more '{Attr}_new' columns. Returns {filename: {attr: target_value}}.
    """
    shards = sorted(glob.glob(str(cf_dir / "interventions_shard*.csv")))
    if not shards:
        raise FileNotFoundError(f"No interventions_shard*.csv found in {cf_dir}")
    df = pd.concat([pd.read_csv(s) for s in shards], ignore_index=True)
    df = df[df["filename"] != "filename"]  # drop duplicate header rows from sharded CSVs
    new_cols = [c for c in df.columns if c.endswith("_new")]
    if not new_cols:
        raise ValueError(
            f"No '*_new' columns found in CSVs at {cf_dir}. "
            f"Expected columns like 'Age_new'. Found: {list(df.columns)}"
        )
    return {
        str(row["filename"]): {col.replace("_new", ""): float(row[col]) for col in new_cols}
        for _, row in df.iterrows()
    }


def load_judges(dataset_key: str, config: EvalConfig):
    cfg = config.get_dataset_config(dataset_key)
    judges = {}
    thresholds = {}

    for attr in config.JUDGE_BINARY:
        ckpt_key = f"judge_{attr}_ckpt"
        if ckpt_key not in cfg:
            continue
        model, th = load_binary_classifier(checkpoint_path=cfg[ckpt_key], device=config.DEVICE)  # type: ignore[misc]
        judges[attr] = model
        thresholds[attr] = th

    for attr in config.JUDGE_REGRESSION:
        ckpt_key = f"judge_{attr}_ckpt"
        if ckpt_key not in cfg:
            continue
        judges[attr] = load_regression_judge(checkpoint_path=cfg[ckpt_key], device=config.DEVICE)

    judges["thresholds"] = thresholds
    return judges


# ============================================================
# Core evaluation
# ============================================================

def evaluate_method(
    method_name,
    dataset_key,
    config,
    judges,
    loader,
    train_ref_loader,
    feat_enc,
    aspects,
    intervention_name,
    intervention_key,
    val_labels,
    intervention_targets=None,
    is_continuous=False,
    output_path=None,
    partial_saves=True,
):
    print(f"\n{'='*60}")
    print(f"Method: {method_name}")
    print(f"Dataset: {dataset_key}")
    print(f"Intervention: {intervention_name}")
    print(f"{'='*60}\n")

    # -------------------------------
    # Results
    # -------------------------------
    results = {
        "method": method_name,
        "test_dataset": dataset_key,
        "intervention": intervention_name,
    }

    thresholds = judges["thresholds"]

    # -------------------------------
    # Evaluation loop
    # -------------------------------
    for aspect in aspects:
        print(f"\n--- {aspect.upper()} ---")

        if aspect == "composition":
            out = evaluate_composition(
                loader=loader,
                gender_judge=judges["gender"],
                view_judge=judges["view"],
                disease_pa_judge=judges["disease_pa"],
                disease_ap_judge=judges["disease_ap"],
                age_judge=judges["age"],
                feat_enc=feat_enc,
                config=config,
                thresholds=thresholds,
                intervention_name=intervention_name,
                intervention_key=intervention_key,
                is_continuous=is_continuous,
            )

        elif aspect == "effectiveness":
            dataset_cfg = config.get_dataset_config(dataset_key)
            disease_meta_key = dataset_cfg["metadata_keys"]["disease"]
            view_meta_key = dataset_cfg["metadata_keys"]["view"]

            cal_out = evaluate_judge_calibration(
                loader=loader,
                binary_judges={
                    attr: (judges[attr], thresholds[attr])
                    for attr in config.JUDGE_BINARY
                    if attr in judges
                },
                regression_judges={
                    attr: judges[attr]
                    for attr in config.JUDGE_REGRESSION
                    if attr in judges
                },
                meta_keys={
                    "gender":     dataset_cfg["metadata_keys"]["gender"],
                    "view":       view_meta_key,
                    "disease_pa": disease_meta_key,
                    "disease_ap": disease_meta_key,
                    "age":        dataset_cfg["metadata_keys"]["age"],
                },
                intervention_name=intervention_name,
                is_continuous=is_continuous,
                config=config,
                targets=intervention_targets,
                view_filters={"disease_pa": 0, "disease_ap": 1},
                view_meta_key=view_meta_key,
            )
            results.update(cal_out)

            if is_continuous:
                out = evaluate_continuous_effectiveness(
                    loader=loader,
                    judge=judges[intervention_name],
                    config=config,
                    targets=intervention_targets,
                    intervention_name=intervention_name,
                    intervention_key=intervention_key,
                )
            else:
                if intervention_name == "disease":
                    out = {}
                    for sub in ("disease_pa", "disease_ap"):
                        out.update(evaluate_binary_effectiveness(
                            loader=loader,
                            judge=judges[sub],
                            config=config,
                            thresholds=thresholds,
                            intervention_name=sub,
                            intervention_key=intervention_key,
                            val_labels=val_labels,
                        ))
                else:
                    out = evaluate_binary_effectiveness(
                        loader=loader,
                        judge=judges[intervention_name],
                        config=config,
                        thresholds=thresholds,
                        intervention_name=intervention_name,
                        intervention_key=intervention_key,
                        val_labels=val_labels,
                    )

        elif aspect == "realism":
            out = evaluate_realism(
                loader=loader,
                feat_enc=feat_enc,
                config=config,
                intervention_name=intervention_name,
                intervention_key=intervention_key,
                val_labels=val_labels,
                is_continuous=is_continuous,
                train_ref_loader=train_ref_loader,
            )

        elif aspect == "minimality":
            out = evaluate_minimality(
                loader=loader,
                gender_judge=judges["gender"],
                view_judge=judges["view"],
                disease_pa_judge=judges["disease_pa"],
                disease_ap_judge=judges["disease_ap"],
                age_judge=judges["age"],
                config=config,
                thresholds=thresholds,
                intervention_name=intervention_name,
                intervention_key=intervention_key,
                val_labels=val_labels,
                is_continuous=is_continuous,
            )

        elif aspect == "reversibility":
            out = evaluate_reversibility(
                loader=loader,
                gender_judge=judges["gender"],
                view_judge=judges["view"],
                disease_pa_judge=judges["disease_pa"],
                disease_ap_judge=judges["disease_ap"],
                age_judge=judges["age"],
                feat_enc=feat_enc,
                config=config,
                thresholds=thresholds,
                intervention_name=intervention_name,
                intervention_key=intervention_key,
                val_labels=val_labels,
                is_continuous=is_continuous,
            )

        else:
            print(f"Skipping unknown aspect: {aspect}")
            continue

        results.update(out)
        print(f"✓ {aspect} done ({len(out)} metrics)")

        if output_path and partial_saves:
            partial_path = output_path.with_name(output_path.stem + "_partial" + output_path.suffix)
            pd.DataFrame([results]).to_csv(partial_path, index=False)

    print(f"\nTotal metrics: {len(results)}")

    if output_path:
        partial_path = output_path.with_name(output_path.stem + "_partial" + output_path.suffix)
        if partial_saves and partial_path.exists():
            partial_path.rename(output_path)
        else:
            pd.DataFrame([results]).to_csv(output_path, index=False)

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser("CIG Evaluation")

    parser.add_argument("--cig_path", required=True,
                        help="Experiment root directory containing CF subdirs (e.g. View_flip/, Sex_flip/).")
    parser.add_argument(
        "--test_dataset",
        required=True,
        choices=list(EvalConfig.EVAL_DATASETS.keys()),
    )
    parser.add_argument(
        "--aspects",
        nargs="+",
        default=[
            "composition",
            "effectiveness",
            "realism",
            "minimality",
            "reversibility",
        ],
    )
    parser.add_argument("--intervention", default="view",
                        help="Intervention to evaluate, or 'all' to run every configured intervention.")
    parser.add_argument("--cf_subdir", default=None,
                        help="Override the CF subdirectory name (default: {MetadataKey}_flip). "
                             "Required when the CF subdir cannot be resolved from config "
                             "(e.g. continuous interventions: 'Age_rand_-0.5_0.5').")
    parser.add_argument("--sanity_check", action="store_true")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory. Defaults to <workspace>/eval_results/<eval_version>/<experiment_rel_path>/")

    args = parser.parse_args()

    config = EvalConfig()
    seed_all(config.SEED)

    dataset_cfg = config.get_dataset_config(args.test_dataset)
    valid_interventions = dataset_cfg.get("interventions", ["view"])
    if args.intervention == "all":
        interventions_to_run = valid_interventions
    elif args.intervention in valid_interventions:
        interventions_to_run = [args.intervention]
    else:
        raise ValueError(
            f"--intervention '{args.intervention}' not valid for '{args.test_dataset}'. "
            f"Valid: {valid_interventions} (or 'all')"
        )

    method_name = parse_method_name(args.cig_path)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        cig_path_abs = Path(args.cig_path).resolve()
        rel = cig_path_abs.relative_to(config.CIG_WORKSPACE)
        output_dir = config.EVAL_OUTPUT_BASE / rel

    print(f"\nRunning evaluation:")
    print(f"  Method: {method_name}")
    print(f"  Dataset: {args.test_dataset}")
    print(f"  Path: {args.cig_path}")
    print(f"  Output Path: {output_dir}")
    print(f"  Interventions: {interventions_to_run}")

    # -------------------------------
    # Load once — intervention-independent
    # (judges, feature encoder, realism reference)
    # -------------------------------
    train_ref_loader = None
    if "realism" in args.aspects:
        train_dataset = config.build_judge_dataset(args.test_dataset)
        train_ref_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
        print(f"Loaded train reference dataset: {len(train_dataset)} samples")

    judges = load_judges(args.test_dataset, config)

    feat_enc = FrozenFeatureEncoder().to(config.DEVICE)
    feat_enc.eval()

    if not args.sanity_check:
        output_dir.mkdir(parents=True, exist_ok=True)
        save_metric_documentation(output_dir, config)

    for intervention_name in interventions_to_run:
        intervention_key = dataset_cfg["metadata_keys"][intervention_name]

        # Detect continuous intervention from YAML schema.
        # Use schema_keys override if present: some metadata_keys (e.g. "Effusion") are
        # not top-level schema keys (they live inside a group_categorical like "Disease").
        from custom_datasets import load_custom_dataset_config
        yaml_cfg = load_custom_dataset_config(dataset_cfg["dataset_name"])
        _schema_key = dataset_cfg.get("schema_keys", {}).get(intervention_name) or intervention_key
        is_continuous = yaml_cfg["schema"][_schema_key]["type"] == "continuous"

        val_labels = None if is_continuous else EvalConfig.get_intervention_labels(args.test_dataset, intervention_name)

        # Resolve CF path: CLI --cf_subdir > dataset config intv_exp_dirs
        if args.cf_subdir:
            cf_subdir = args.cf_subdir
        elif intervention_name in dataset_cfg.get("intv_exp_dirs", {}):
            cf_subdir = dataset_cfg["intv_exp_dirs"][intervention_name]
        else:
            raise ValueError(
                f"No CF subdir configured for intervention '{intervention_name}'. "
                f"Add it to 'intv_exp_dirs' in EvalConfig.EVAL_DATASETS or pass --cf_subdir."
            )
        cf_path = Path(args.cig_path) / cf_subdir

        intervention_targets = None
        if is_continuous:
            _all_targets = load_intervention_targets(cf_path)
            _sample = next(iter(_all_targets.values()))
            if intervention_key not in _sample:
                raise ValueError(
                    f"intervention_key '{intervention_key}' not found in targets CSV. "
                    f"Available keys: {list(_sample.keys())}"
                )
            intervention_targets = {fn: _row[intervention_key] for fn, _row in _all_targets.items()}
            print(f"Loaded intervention targets ({intervention_name}): {len(intervention_targets)} samples")

        paired_dataset = config.build_paired_dataset(args.test_dataset, str(cf_path))
        loader = DataLoader(
            paired_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
        print(f"Loaded paired dataset ({intervention_name}): {len(paired_dataset)} samples")

        output_path = None if args.sanity_check else (
            output_dir / f"eval_results_{args.test_dataset}_{intervention_name}_{method_name}.csv"
        )

        evaluate_method(
            method_name=method_name,
            dataset_key=args.test_dataset,
            config=config,
            judges=judges,
            loader=loader,
            train_ref_loader=train_ref_loader,
            feat_enc=feat_enc,
            aspects=args.aspects,
            intervention_name=intervention_name,
            intervention_key=intervention_key,
            val_labels=val_labels,
            intervention_targets=intervention_targets,
            is_continuous=is_continuous,
            output_path=output_path,
        )

        if args.sanity_check:
            print("\nSanity check mode — not saving results.")
        else:
            print(f"\nSaved results → {output_path}")


if __name__ == "__main__":
    main()