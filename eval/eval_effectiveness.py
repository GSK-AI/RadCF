"""
EFFECTIVENESS (Target Attribute Change) evaluation.

PURPOSE:
    Measures if counterfactuals successfully achieve the target attribute change,
    and reports how all judges respond to the intervention — both as a calibration
    baseline (on originals) and as a measure of effectiveness/collateral impact (on CFs).

KEY METRICS:
    Judge calibration and CF response (per intervention):
    - judge_{attr}_orig_auc / judge_{attr}_orig_mae  — judge performance on original images
    - judge_{attr}_cf_auc   / judge_{attr}_cf_mae    — judge performance on CF images
    - judge_{attr}_auc_delta / judge_{attr}_mae_delta — signed delta (CF − orig)

    Labels used for CF evaluation:
    - Target attribute: intervention target label (flipped for binary; target value for
      continuous). High CF AUC / low |delta| = effective intervention.
    - Collateral attributes: original label (unchanged). Low |delta| = preserved.

    Binary intervention effectiveness:
    - cf_{intervention}_flip_success: fraction of CFs where predicted attribute = target
    - cf_{intervention}_margin_{mean,median,std,p95}: P(target) − P(original)
    - cf_{intervention}_logit_delta_{mean,median,std,p95}: logit-space margin

    Continuous intervention effectiveness:
    - cf_{intervention}_direction_success: fraction moving in the correct direction
    - cf_{intervention}_delta_{mean,median,std}: predicted change distribution
    - cf_{intervention}_error_{mean,median,std}: |pred_cf − target| distribution

INTERPRETATION:
    For all judges: |auc_delta| ≈ 0 is ideal for collateral attributes (preserved);
    |auc_delta| large for target attribute indicates effective intervention.
    orig_auc/mae contextualises all other metrics — low judge quality on originals
    makes CF metrics less meaningful.
"""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from eval.metrics.utils import logit, summarize_np, cat_list


@torch.no_grad()
def evaluate_judge_calibration(
    loader,
    binary_judges,
    regression_judges,
    meta_keys,
    intervention_name,
    is_continuous,
    config,
    targets=None,
    view_filters=None,
    view_meta_key="View",
):
    """Evaluate all judges on original and CF images, reporting AUC/MAE and their delta.

    For each judge, computes performance on both the original (un-intervened) images
    and the CF images. Labels used for CF evaluation:
    - Target attribute (binary): flipped label (1 - original)
    - Target attribute (continuous): target value from the intervention CSV
    - All collateral attributes: original label (unchanged)

    This mirrors the |delta(AUC)| table from the effectiveness literature: lower |delta|
    on collateral attributes = preserved; non-trivial delta on the target = effective.

    Args:
        loader: DataLoader yielding (img_orig, img_cf, img_null, img_rev, metas)
        binary_judges: {attr: (BinaryClassifier, threshold)}
        regression_judges: {attr: RegressionJudge}
        meta_keys: {attr: metadata_key} mapping judge names to metas dict keys
        intervention_name: target attribute name (e.g. "gender", "age")
        is_continuous: whether the target intervention is continuous
        config: EvalConfig
        targets: {filename: target_value} required when is_continuous=True and
            intervention_name is a regression judge
        view_filters: {attr: int} optional per-judge view subset (0=PA, 1=AP)
        view_meta_key: metas key for view label, used when view_filters is set

    Returns:
        Dict with keys:
            judge_{attr}_orig_auc  / judge_{attr}_orig_mae  — on original images
            judge_{attr}_cf_auc    / judge_{attr}_cf_mae    — on CF images
            judge_{attr}_auc_delta / judge_{attr}_mae_delta — signed delta (CF − orig)
    """
    view_filters = view_filters or {}

    for judge, _ in binary_judges.values():
        judge.eval()
    for judge in regression_judges.values():
        judge.eval()

    orig_probs  = {attr: [] for attr in binary_judges}
    cf_probs    = {attr: [] for attr in binary_judges}
    bin_labels  = {attr: [] for attr in binary_judges}

    orig_preds_reg = {attr: [] for attr in regression_judges}
    cf_preds_reg   = {attr: [] for attr in regression_judges}
    orig_labels_reg = {attr: [] for attr in regression_judges}
    cf_labels_reg   = {attr: [] for attr in regression_judges}

    for img_orig, img_cf, _, _, metas in tqdm(loader, desc="eval[judge_calibration]"):
        img_orig = img_orig.to(config.DEVICE)
        img_cf   = img_cf.to(config.DEVICE)

        for attr, (judge, _) in binary_judges.items():
            view_val = view_filters.get(attr)
            if view_val is not None:
                mask = metas[view_meta_key] == view_val
                if not mask.any():
                    continue
                o_imgs = img_orig[mask.to(img_orig.device)]
                orig_lbl = metas[meta_keys[attr]][mask]
                # For view interventions, CFs target the opposite view, so route them
                # to the judge matching their target view (1 - original view).
                cf_mask = ((1 - metas[view_meta_key]) == view_val) if intervention_name == "view" else mask
                c_imgs = img_cf[cf_mask.to(img_cf.device)]
                cf_lbl_base = metas[meta_keys[attr]][cf_mask]
            else:
                o_imgs, c_imgs = img_orig, img_cf
                orig_lbl = metas[meta_keys[attr]]
                cf_lbl_base = orig_lbl

            # CF labels: flipped for target, original for collateral
            cf_lbl = (1 - cf_lbl_base) if (attr == intervention_name and not is_continuous) else cf_lbl_base

            orig_probs[attr].append(judge.predict_probs(o_imgs)[:, 1].cpu().numpy())
            cf_probs[attr].append(judge.predict_probs(c_imgs)[:, 1].cpu().numpy())
            bin_labels[attr].append((orig_lbl.numpy(), cf_lbl.numpy()))

        for attr, judge in regression_judges.items():
            orig_lbl = metas[meta_keys[attr]].float()
            if attr == intervention_name and is_continuous:
                if targets is None:
                    raise ValueError(
                        f"targets=None for continuous intervention '{intervention_name}'; "
                        "cannot compute CF MAE without target values."
                    )
                cf_lbl = torch.tensor(
                    [targets[fn] for fn in metas["filename"]], dtype=torch.float32
                )
            else:
                cf_lbl = orig_lbl

            orig_preds_reg[attr].append(judge(img_orig).cpu().numpy())
            cf_preds_reg[attr].append(judge(img_cf).cpu().numpy())
            orig_labels_reg[attr].append(orig_lbl.numpy())
            cf_labels_reg[attr].append(cf_lbl.numpy())

    results = {}
    for attr in binary_judges:
        if not bin_labels[attr]:
            # No samples passed the view filter for this judge; skip rather than crash.
            print(f"[eval_judge_calibration] WARNING: no samples for judge '{attr}'; skipping AUC.")
            continue
        orig_lbl_all = np.concatenate([p[0] for p in bin_labels[attr]])
        cf_lbl_all   = np.concatenate([p[1] for p in bin_labels[attr]])
        orig_p = np.concatenate(orig_probs[attr])
        cf_p   = np.concatenate(cf_probs[attr])
        try:
            orig_auc = float(roc_auc_score(orig_lbl_all, orig_p))
        except ValueError:
            orig_auc = float("nan")
        try:
            cf_auc = float(roc_auc_score(cf_lbl_all, cf_p))
        except ValueError:
            cf_auc = float("nan")
        results[f"judge_{attr}_orig_auc"]  = orig_auc
        results[f"judge_{attr}_cf_auc"]    = cf_auc
        results[f"judge_{attr}_auc_delta"] = cf_auc - orig_auc

    for attr in regression_judges:
        orig_p = np.concatenate(orig_preds_reg[attr])
        cf_p   = np.concatenate(cf_preds_reg[attr])
        orig_lbl = np.concatenate(orig_labels_reg[attr])
        cf_lbl   = np.concatenate(cf_labels_reg[attr])
        orig_mae = float(np.abs(orig_p - orig_lbl).mean())
        cf_mae   = float(np.abs(cf_p - cf_lbl).mean())
        results[f"judge_{attr}_orig_mae"]  = orig_mae
        results[f"judge_{attr}_cf_mae"]    = cf_mae
        results[f"judge_{attr}_mae_delta"] = cf_mae - orig_mae

    return results


@torch.no_grad()
def evaluate_binary_effectiveness(
    loader, judge, config, thresholds, intervention_name, intervention_key, val_labels
):
    """
    Evaluate effectiveness (target attribute change).

    Args:
        loader: DataLoader with (img_orig, img_cf, img_null, img_rev, metas)
        judge: Classifier for the intervention attribute
        config: Configuration object
        thresholds: Dict with optimal decision thresholds (e.g., {"view": 0.5})
        intervention_name: Short name for the intervention (e.g., "view", "gender")
        intervention_key: Metadata key to read from metas (e.g., "View", "Sex")
        val_labels: Tuple of lowercase labels for values 0 and 1 (e.g., ("pa", "ap"))

    Returns:
        Dictionary of effectiveness metrics (prefixed with 'cf_{intervention_name}_')
    """
    intvn_threshold = thresholds.get(intervention_name, 0.5)
    lbl0, lbl1 = val_labels

    judge.eval()

    n = 0
    cf_flip_ok = 0
    cf_margin = []
    cf_delta = []

    # Stratified by intervention value
    n_0, n_1 = 0, 0
    cf_flip_ok_0, cf_flip_ok_1 = 0, 0
    cf_margin_0, cf_margin_1 = [], []
    cf_delta_0, cf_delta_1 = [], []

    for img_orig, img_cf, img_null, img_rev, metas in tqdm(
        loader, desc="eval[effectiveness]"
    ):
        img_cf = img_cf.to(config.DEVICE)
        intvn_orig = metas[intervention_key].long().to(config.DEVICE)
        intvn_flip = 1 - intvn_orig

        bs = img_cf.size(0)
        n += bs

        mask_0 = (intvn_orig == 0)
        mask_1 = (intvn_orig == 1)
        n_0 += mask_0.sum().item()
        n_1 += mask_1.sum().item()

        p_cf_intvn = judge.predict_probs(img_cf)
        pred_cf_intvn = (p_cf_intvn[:, 1] >= intvn_threshold).long()

        cf_flip_ok += (pred_cf_intvn == intvn_flip).sum().item()

        p_target = p_cf_intvn.gather(1, intvn_flip.view(-1, 1)).squeeze(1)
        p_orig_v = p_cf_intvn.gather(1, intvn_orig.view(-1, 1)).squeeze(1)

        margin = p_target - p_orig_v
        cf_margin.append(margin.cpu())

        delta = logit(p_target) - logit(p_orig_v)
        cf_delta.append(delta.cpu())

        if mask_0.any():
            cf_flip_ok_0 += (pred_cf_intvn[mask_0] == intvn_flip[mask_0]).sum().item()
            cf_margin_0.append(margin[mask_0].cpu())
            cf_delta_0.append(delta[mask_0].cpu())
        if mask_1.any():
            cf_flip_ok_1 += (pred_cf_intvn[mask_1] == intvn_flip[mask_1]).sum().item()
            cf_margin_1.append(margin[mask_1].cpu())
            cf_delta_1.append(delta[mask_1].cpu())

    # Aggregate
    margin_np = cat_list(cf_margin)
    delta_np = cat_list(cf_delta)
    margin_0_np = cat_list(cf_margin_0)
    margin_1_np = cat_list(cf_margin_1)
    delta_0_np = cat_list(cf_delta_0)
    delta_1_np = cat_list(cf_delta_1)
    intvn = intervention_name

    return {
        f"cf_{intvn}_flip_success": float(cf_flip_ok / max(n, 1)),
        f"cf_{intvn}_flip_success_{lbl0}_to_{lbl1}": float(cf_flip_ok_0 / max(n_0, 1)),
        f"cf_{intvn}_flip_success_{lbl1}_to_{lbl0}": float(cf_flip_ok_1 / max(n_1, 1)),
        # Overall margin and logit delta
        f"cf_{intvn}_margin_mean": summarize_np(margin_np)["mean"],
        f"cf_{intvn}_margin_median": summarize_np(margin_np)["median"],
        f"cf_{intvn}_margin_std": summarize_np(margin_np)["std"],
        f"cf_{intvn}_margin_p95": summarize_np(margin_np)["p95"],
        f"cf_{intvn}_logit_delta_mean": summarize_np(delta_np)["mean"],
        f"cf_{intvn}_logit_delta_median": summarize_np(delta_np)["median"],
        f"cf_{intvn}_logit_delta_std": summarize_np(delta_np)["std"],
        f"cf_{intvn}_logit_delta_p95": summarize_np(delta_np)["p95"],
        # Stratified margin
        f"cf_{intvn}_margin_mean_{lbl0}_to_{lbl1}": summarize_np(margin_0_np)["mean"],
        f"cf_{intvn}_margin_median_{lbl0}_to_{lbl1}": summarize_np(margin_0_np)["median"],
        f"cf_{intvn}_margin_std_{lbl0}_to_{lbl1}": summarize_np(margin_0_np)["std"],
        f"cf_{intvn}_margin_mean_{lbl1}_to_{lbl0}": summarize_np(margin_1_np)["mean"],
        f"cf_{intvn}_margin_median_{lbl1}_to_{lbl0}": summarize_np(margin_1_np)["median"],
        f"cf_{intvn}_margin_std_{lbl1}_to_{lbl0}": summarize_np(margin_1_np)["std"],
        # Stratified logit delta
        f"cf_{intvn}_logit_delta_mean_{lbl0}_to_{lbl1}": summarize_np(delta_0_np)["mean"],
        f"cf_{intvn}_logit_delta_median_{lbl0}_to_{lbl1}": summarize_np(delta_0_np)["median"],
        f"cf_{intvn}_logit_delta_std_{lbl0}_to_{lbl1}": summarize_np(delta_0_np)["std"],
        f"cf_{intvn}_logit_delta_mean_{lbl1}_to_{lbl0}": summarize_np(delta_1_np)["mean"],
        f"cf_{intvn}_logit_delta_median_{lbl1}_to_{lbl0}": summarize_np(delta_1_np)["median"],
        f"cf_{intvn}_logit_delta_std_{lbl1}_to_{lbl0}": summarize_np(delta_1_np)["std"],
    }


@torch.no_grad()
def evaluate_continuous_effectiveness(
    loader, judge, config, targets, intervention_name, intervention_key
):
    """
    Evaluate effectiveness for a continuous (regression) intervention.

    Uses a regression judge to measure whether the CF image's predicted value
    moved in the correct direction and towards the intended target value.
    Binary flip success is not applicable here — instead we report directional
    accuracy and error distributions.

    Args:
        loader: DataLoader with (img_orig, img_cf, img_null, img_rev, metas).
            metas must contain 'filename' and intervention_key (e.g., 'Age').
        judge: RegressionJudge for the attribute
        config: EvalConfig
        targets: dict mapping filename → target value in [0, 1], loaded from
            the intervention CSV ({Attr}_new column)
        intervention_name: Short name, e.g., "age"
        intervention_key: Metadata key, e.g., "Age"

    Returns:
        Dictionary of effectiveness metrics prefixed with 'cf_{intervention_name}_'
    """
    # Threshold below which a requested change is too small to assess direction
    _DIR_THRESHOLD = 1e-3

    judge.eval()

    all_delta = []         # pred_cf - pred_orig (signed predicted change)
    all_target_delta = []  # target - orig (requested change)
    all_error = []         # |pred_cf - target|

    for img_orig, img_cf, _, _, metas in tqdm(
        loader, desc=f"eval[{intervention_name}_effectiveness]"
    ):
        img_orig = img_orig.to(config.DEVICE)
        img_cf = img_cf.to(config.DEVICE)

        attr_orig = metas[intervention_key].float().to(config.DEVICE)
        attr_target = torch.tensor(
            [targets[fn] for fn in metas["filename"]], dtype=torch.float32
        ).to(config.DEVICE)

        pred_orig = judge(img_orig)   # (B,)
        pred_cf = judge(img_cf)       # (B,)

        all_delta.append((pred_cf - pred_orig).cpu())
        all_target_delta.append((attr_target - attr_orig).cpu())
        all_error.append((pred_cf - attr_target).abs().cpu())

    delta_np = cat_list(all_delta)
    target_delta_np = cat_list(all_target_delta)
    error_np = cat_list(all_error)

    sig_mask = np.abs(target_delta_np) > _DIR_THRESHOLD
    inc_mask = target_delta_np > _DIR_THRESHOLD
    dec_mask = target_delta_np < -_DIR_THRESHOLD

    direction_ok = np.sign(delta_np[sig_mask]) == np.sign(target_delta_np[sig_mask])
    direction_ok_inc = delta_np[inc_mask] > 0
    direction_ok_dec = delta_np[dec_mask] < 0

    def _dir_success(arr):
        return float(arr.mean()) if arr.size > 0 else float("nan")

    intvn = intervention_name
    return {
        f"cf_{intvn}_direction_success": _dir_success(direction_ok),
        f"cf_{intvn}_direction_success_increase": _dir_success(direction_ok_inc),
        f"cf_{intvn}_direction_success_decrease": _dir_success(direction_ok_dec),
        # Predicted change distribution
        f"cf_{intvn}_delta_mean": summarize_np(delta_np)["mean"],
        f"cf_{intvn}_delta_median": summarize_np(delta_np)["median"],
        f"cf_{intvn}_delta_std": summarize_np(delta_np)["std"],
        # Requested change distribution (reference — what the model was asked to do)
        f"cf_{intvn}_target_delta_mean": summarize_np(target_delta_np)["mean"],
        f"cf_{intvn}_target_delta_median": summarize_np(target_delta_np)["median"],
        f"cf_{intvn}_target_delta_std": summarize_np(target_delta_np)["std"],
        # Absolute error between predicted and intended target age
        f"cf_{intvn}_error_mean": summarize_np(error_np)["mean"],
        f"cf_{intvn}_error_median": summarize_np(error_np)["median"],
        f"cf_{intvn}_error_std": summarize_np(error_np)["std"],
    }
