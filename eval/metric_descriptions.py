"""
Generate a complete metric description dictionary covering all configured interventions.

Produces one description per metric key across all evaluators (composition, effectiveness,
realism, minimality, reversibility). Keys that are irrelevant to a given run are included
for reference -- consumers should ignore keys absent from the CSV.
"""


def build_metric_descriptions(config) -> dict:
    """
    Build a flat {metric_name: description} dict covering all possible output metrics.

    Iterates all datasets and interventions in EvalConfig.EVAL_DATASETS and expands
    parameterised metric names (intervention name, label suffixes). Fixed metrics
    (realism, pixel distances, FID/KID) are included once.
    """
    from eval.config import EvalConfig

    d = {}

    # ------------------------------------------------------------------
    # Identity / metadata
    # ------------------------------------------------------------------
    d.update({
        "method":       "CIG method identifier",
        "test_dataset": "Evaluation dataset key",
        "intervention": "Intervention attribute name",
    })

    # ------------------------------------------------------------------
    # Realism  (orig_cf_*)
    # ------------------------------------------------------------------
    _pixel_descs = {
        "mse":   "MSE between original and CF",
        "psnr":  "PSNR between original and CF (higher=better)",
        "ssim":  "SSIM between original and CF (moderate ideal: not too high/low)",
        "l1":    "L1 distance between original and CF",
        "mae":   "MAE between original and CF",
        "lpips": "LPIPS between original and CF (lower=better perceptual quality)",
    }
    for metric, desc in _pixel_descs.items():
        for stat in ["mean", "median", "std"]:
            d[f"orig_cf_{metric}_{stat}"] = f"{stat.capitalize()} {desc}"

    for stat in ["mean", "median", "std", "p95"]:
        d[f"orig_cf_feat_dist_{stat}"] = (
            f"{stat.capitalize()} ResNet18 feature distance between original and CF"
        )

    d["orig_cf_fid"]      = "FID between original and CF distributions (lower=more realistic)"
    d["orig_cf_fid_n"]    = "Sample size used for orig-CF FID computation"
    d["orig_cf_kid_mean"] = "KID mean between original and CF distributions (lower=more realistic)"
    d["orig_cf_kid_std"]  = "KID std between original and CF distributions"

    # ------------------------------------------------------------------
    # Composition  (orig_null_* + null_*)
    # ------------------------------------------------------------------
    _null_pixel = {
        "mse":   "MSE between original and null (lower=less drift)",
        "psnr":  "PSNR between original and null (higher=less drift)",
        "ssim":  "SSIM between original and null (higher=less drift)",
        "l1":    "L1 distance between original and null",
        "mae":   "MAE between original and null",
        "lpips": "LPIPS between original and null (lower=less drift)",
    }
    for metric, desc in _null_pixel.items():
        for stat in ["mean", "median", "std"]:
            d[f"orig_null_{metric}_{stat}"] = f"{stat.capitalize()} {desc}"

    for stat in ["mean", "median", "std", "p95"]:
        d[f"orig_null_feat_dist_{stat}"] = (
            f"{stat.capitalize()} ResNet18 feature distance between original and null"
        )

    d["orig_null_fid"]      = "FID between original and null distributions (lower=less drift)"
    d["orig_null_fid_n"]    = "Sample size used for orig-null FID computation"
    d["orig_null_kid_mean"] = "KID mean between original and null distributions"
    d["orig_null_kid_std"]  = "KID std between original and null distributions"

    # ------------------------------------------------------------------
    # Minimality  (cf_cld_*) -- disabled: fc_logvar not trained
    # ------------------------------------------------------------------
    # for stat in ["mean", "median", "std"]:
    #     d[f"cf_cld_{stat}"] = (
    #         f"{stat.capitalize()} Contrastive Latent Divergence "
    #         "(balances minimality and sufficiency; lower=better)"
    #     )

    # ------------------------------------------------------------------
    # Reversibility  (orig_rev_cf_* + null_rev_cf_*)
    # ------------------------------------------------------------------
    _rev_pairs = {
        "orig_rev_cf": "original and reverse CF",
        "null_rev_cf": "null and reverse CF (composition control baseline)",
    }
    _rev_pixel = {
        "mse":   "MSE",
        "psnr":  "PSNR (higher=better)",
        "ssim":  "SSIM",
        "l1":    "L1 distance",
        "mae":   "MAE",
        "lpips": "LPIPS",
    }
    for pair, pair_desc in _rev_pairs.items():
        for metric, metric_desc in _rev_pixel.items():
            for stat in ["mean", "median", "std"]:
                d[f"{pair}_{metric}_{stat}"] = (
                    f"{stat.capitalize()} {metric_desc} between {pair_desc}"
                )
        for stat in ["mean", "median", "std", "p95"]:
            d[f"{pair}_feat_dist_{stat}"] = (
                f"{stat.capitalize()} ResNet18 feature distance between {pair_desc}"
            )
        d[f"{pair}_fid"]      = f"FID between {pair_desc} distributions"
        d[f"{pair}_fid_n"]    = f"Sample size used for {pair} FID computation"
        d[f"{pair}_kid_mean"] = f"KID mean between {pair_desc} distributions"
        d[f"{pair}_kid_std"]  = f"KID std between {pair_desc} distributions"

    # ------------------------------------------------------------------
    # Intervention-parameterised metrics
    # ------------------------------------------------------------------
    _all_attrs = {"gender", "view", "disease"}

    from custom_datasets import load_custom_dataset_config

    # ------------------------------------------------------------------
    # Judge calibration and CF response  (per intervention)
    # orig_*: judge performance on original images (baseline)
    # cf_*:   judge performance on CF images (target attr uses flipped/target label;
    #         collateral attrs use original label)
    # delta:  signed CF - orig; |delta| ~= 0 for collateral (preserved),
    #         non-trivial for target (effective intervention)
    # ------------------------------------------------------------------
    _binary_judges = config.JUDGE_BINARY
    _reg_judges    = config.JUDGE_REGRESSION
    for attr in _binary_judges:
        d[f"judge_{attr}_orig_auc"]  = f"AUC of {attr} judge on original test images (baseline)"
        d[f"judge_{attr}_cf_auc"]    = (
            f"AUC of {attr} judge on CF images "
            "(target attr uses flipped label; collateral uses original)"
        )
        d[f"judge_{attr}_auc_delta"] = (
            f"Signed AUC delta for {attr} judge (CF - orig); "
            "near-zero=collateral preserved; non-trivial=effective intervention"
        )
    for attr in _reg_judges:
        d[f"judge_{attr}_orig_mae"]  = f"MAE of {attr} judge on original test images (baseline)"
        d[f"judge_{attr}_cf_mae"]    = (
            f"MAE of {attr} judge on CF images "
            "(target attr uses intervention target value; collateral uses original)"
        )
        d[f"judge_{attr}_mae_delta"] = (
            f"Signed MAE delta for {attr} judge (CF - orig); "
            "near-zero=collateral preserved; non-trivial=effective intervention"
        )

    seen = set()
    for dataset_key, dataset_cfg in config.EVAL_DATASETS.items():
        yaml_cfg = load_custom_dataset_config(dataset_cfg["dataset_name"])
        for intvn in dataset_cfg.get("interventions", ["view"]):
            meta_key = dataset_cfg["metadata_keys"][intvn]
            schema_key = dataset_cfg.get("schema_keys", {}).get(intvn) or meta_key
            intvn_type = yaml_cfg["schema"][schema_key]["type"]  # "categorical" or "continuous"
            is_continuous = intvn_type == "continuous"

            # Dedup on intervention name only -- assumes same name -> same schema
            # (type + labels) across all datasets, which holds for this codebase.
            # Continuous interventions have no labels, so name-only is the only option.
            if intvn in seen:
                continue
            seen.add(intvn)

            if is_continuous:
                # Effectiveness -- directional metrics (no binary flip)
                d[f"cf_{intvn}_direction_success"] = (
                    f"Fraction of CFs where predicted {intvn} moved in the correct direction (higher=better)"
                )
                d[f"cf_{intvn}_direction_success_increase"] = (
                    f"Direction success for interventions requesting an increase in {intvn}"
                )
                d[f"cf_{intvn}_direction_success_decrease"] = (
                    f"Direction success for interventions requesting a decrease in {intvn}"
                )
                for stat in ["mean", "median", "std"]:
                    d[f"cf_{intvn}_delta_{stat}"] = (
                        f"{stat.capitalize()} signed change in predicted {intvn} (pred_cf - pred_orig)"
                    )
                    d[f"cf_{intvn}_target_delta_{stat}"] = (
                        f"{stat.capitalize()} requested change in {intvn} (target - orig); reference"
                    )
                    d[f"cf_{intvn}_error_{stat}"] = (
                        f"{stat.capitalize()} absolute error between predicted and target {intvn} (lower=better)"
                    )

                # Binary collateral for all attrs (none equal the continuous intervention)
                for attr in sorted(_all_attrs):
                    d[f"null_collateral_{attr}_flip_rate"] = (
                        f"Fraction of null interventions changing {attr} prediction (expect ~0.0)"
                    )
                    d[f"cf_collateral_{attr}_flip_rate"] = (
                        f"Fraction of CFs changing {attr} prediction (expect ~0.0)"
                    )
                    d[f"rev_cf_collateral_{attr}_flip_rate"] = (
                        f"Fraction of reverse CFs changing {attr} prediction (expect ~0.0)"
                    )
                    for stat in ["mean", "median", "std"]:
                        d[f"null_js_{attr}_{stat}"] = (
                            f"{stat.capitalize()} JS divergence between original and null "
                            f"{attr} distributions (lower=better)"
                        )
                        d[f"null_{attr}_dp_{stat}"] = (
                            f"{stat.capitalize()} probability delta for {attr} "
                            "between original and null (lower=better)"
                        )
                        d[f"cf_js_{attr}_{stat}"] = (
                            f"{stat.capitalize()} JS divergence between original and CF "
                            f"{attr} distributions (lower=better)"
                        )
                        d[f"cf_{attr}_dp_{stat}"] = (
                            f"{stat.capitalize()} probability delta for {attr} "
                            "between original and CF (lower=better)"
                        )
                        d[f"rev_cf_js_{attr}_{stat}"] = (
                            f"{stat.capitalize()} JS divergence between original and "
                            f"reverse CF {attr} distributions"
                        )
                        d[f"rev_cf_{attr}_dp_{stat}"] = (
                            f"{stat.capitalize()} probability delta for {attr} "
                            "between original and reverse CF"
                        )

                # Age recovery -- round-trip quality for continuous intervention
                d["rev_cf_age_recovery_mae"] = (
                    "Mean absolute error of age judge between reverse CF and original "
                    "(lower=better; continuous analogue of binary recovery rate)"
                )
                d["rev_cf_age_recovery_median"] = (
                    "Median absolute error of age judge between reverse CF and original"
                )
                d["rev_cf_age_recovery_std"] = (
                    "Std of absolute age judge error between reverse CF and original"
                )
                continue

            lbl0, lbl1 = EvalConfig.get_intervention_labels(dataset_key, intvn)
            a01 = f"{lbl0}_to_{lbl1}"
            a10 = f"{lbl1}_to_{lbl0}"

            # Composition -- intervention consistency
            d[f"null_{intvn}_consistency"] = (
                f"Fraction of null interventions preserving original {intvn} (expect ~1.0)"
            )

            # Effectiveness -- disease uses PA and AP judges separately
            eff_names = [f"{intvn}_pa", f"{intvn}_ap"] if intvn == "disease" else [intvn]
            for eff_name in eff_names:
                judge_desc = f" ({eff_name.split('_')[-1].upper()} judge)" if intvn == "disease" else ""
                d[f"cf_{eff_name}_flip_success"] = (
                    f"Fraction of CFs successfully flipping to target {intvn}{judge_desc} (higher=better)"
                )
                d[f"cf_{eff_name}_flip_success_{a01}"] = (
                    f"{intvn.capitalize()} flip success for {lbl0}->{lbl1} transformations{judge_desc}"
                )
                d[f"cf_{eff_name}_flip_success_{a10}"] = (
                    f"{intvn.capitalize()} flip success for {lbl1}->{lbl0} transformations{judge_desc}"
                )
                for stat in ["mean", "median", "std", "p95"]:
                    d[f"cf_{eff_name}_margin_{stat}"] = (
                        f"{stat.capitalize()} P(target)-P(original) for {intvn} flip{judge_desc} (higher=better)"
                    )
                    d[f"cf_{eff_name}_logit_delta_{stat}"] = (
                        f"{stat.capitalize()} logit delta for {intvn} flip{judge_desc} (higher=better)"
                    )
                for sfx, arrow in [(a01, f"{lbl0}->{lbl1}"), (a10, f"{lbl1}->{lbl0}")]:
                    for stat in ["mean", "median", "std"]:
                        d[f"cf_{eff_name}_margin_{stat}_{sfx}"] = (
                            f"{stat.capitalize()} margin for {arrow} transformations{judge_desc}"
                        )
                        d[f"cf_{eff_name}_logit_delta_{stat}_{sfx}"] = (
                            f"{stat.capitalize()} logit delta for {arrow} transformations{judge_desc}"
                        )

            # Reversibility -- intervention recovery
            sfx0 = f"{lbl0}_to_{lbl1}_to_{lbl0}"
            sfx1 = f"{lbl1}_to_{lbl0}_to_{lbl1}"
            d[f"rev_cf_{intvn}_recovery"] = (
                f"Fraction of reverse CFs recovering original {intvn} (expect ~1.0)"
            )
            d[f"rev_cf_{intvn}_recovery_{sfx0}"] = (
                f"{intvn.capitalize()} recovery for {lbl0}->{lbl1}->{lbl0} cycles"
            )
            d[f"rev_cf_{intvn}_recovery_{sfx1}"] = (
                f"{intvn.capitalize()} recovery for {lbl1}->{lbl0}->{lbl1} cycles"
            )

            # Realism -- stratified perceptual metrics (by intervention value)
            for sfx, arrow in [(a01, f"{lbl0}->{lbl1}"), (a10, f"{lbl1}->{lbl0}")]:
                for metric in ["ssim", "lpips"]:
                    for stat in ["mean", "median", "std"]:
                        d[f"orig_cf_{metric}_{stat}_{sfx}"] = (
                            f"{stat.capitalize()} {metric.upper()} for {arrow} transformations"
                        )
                for stat in ["mean", "median", "std"]:
                    d[f"orig_cf_feat_dist_{stat}_{sfx}"] = (
                        f"{stat.capitalize()} feature distance for {arrow} transformations"
                    )

            # Reversibility -- stratified perceptual metrics (by intervention value)
            for sfx, cycle in [(sfx0, f"{lbl0}->{lbl1}->{lbl0}"), (sfx1, f"{lbl1}->{lbl0}->{lbl1}")]:
                for metric in ["ssim", "lpips"]:
                    for stat in ["mean", "median", "std"]:
                        d[f"orig_rev_cf_{metric}_{stat}_{sfx}"] = (
                            f"{stat.capitalize()} {metric.upper()} for {cycle} cycles"
                        )
                for stat in ["mean", "median", "std"]:
                    d[f"orig_rev_cf_feat_dist_{stat}_{sfx}"] = (
                        f"{stat.capitalize()} feature distance for {cycle} cycles"
                    )

            # Age collateral (always present for binary interventions -- age is never the intervention here)
            for stat in ["mean", "median", "std"]:
                d[f"null_collateral_age_delta_{stat}"] = (
                    f"{stat.capitalize()} |age_judge(null) - age_judge(orig)| "
                    "(null should preserve apparent age; lower=better)"
                )
                d[f"cf_collateral_age_delta_{stat}"] = (
                    f"{stat.capitalize()} |age_judge(CF) - age_judge(orig)| "
                    "(CF should not change apparent age; lower=better)"
                )
                d[f"rev_cf_collateral_age_delta_{stat}"] = (
                    f"{stat.capitalize()} |age_judge(rev_CF) - age_judge(orig)| "
                    "(reverse CF should preserve apparent age; lower=better)"
                )

            # Collateral metrics (all attrs except the intervention)
            for attr in sorted(_all_attrs - {intvn}):
                # Composition collateral
                d[f"null_collateral_{attr}_flip_rate"] = (
                    f"Fraction of null interventions changing {attr} prediction (expect ~0.0)"
                )
                for stat in ["mean", "median", "std"]:
                    d[f"null_js_{attr}_{stat}"] = (
                        f"{stat.capitalize()} JS divergence between original and null "
                        f"{attr} distributions (lower=better)"
                    )
                    d[f"null_{attr}_dp_{stat}"] = (
                        f"{stat.capitalize()} probability delta for {attr} "
                        "between original and null (lower=better)"
                    )

                # Minimality collateral (stratified by intervention value)
                d[f"cf_collateral_{attr}_flip_rate"] = (
                    f"Fraction of CFs changing {attr} prediction (expect ~0.0)"
                )
                d[f"cf_collateral_{attr}_flip_rate_{a01}"] = (
                    f"{attr.capitalize()} flip rate for {lbl0}->{lbl1} transformations"
                )
                d[f"cf_collateral_{attr}_flip_rate_{a10}"] = (
                    f"{attr.capitalize()} flip rate for {lbl1}->{lbl0} transformations"
                )
                for stat in ["mean", "median", "std"]:
                    d[f"cf_js_{attr}_{stat}"] = (
                        f"{stat.capitalize()} JS divergence between original and CF "
                        f"{attr} distributions (lower=better)"
                    )
                    d[f"cf_{attr}_dp_{stat}"] = (
                        f"{stat.capitalize()} probability delta for {attr} "
                        "between original and CF (lower=better)"
                    )

                # Reversibility collateral
                d[f"rev_cf_collateral_{attr}_flip_rate"] = (
                    f"Fraction of reverse CFs changing {attr} prediction (expect ~0.0)"
                )
                for stat in ["mean", "median", "std"]:
                    d[f"rev_cf_js_{attr}_{stat}"] = (
                        f"{stat.capitalize()} JS divergence between original and "
                        f"reverse CF {attr} distributions"
                    )
                    d[f"rev_cf_{attr}_dp_{stat}"] = (
                        f"{stat.capitalize()} probability delta for {attr} "
                        "between original and reverse CF"
                    )

    return dict(sorted(d.items()))
