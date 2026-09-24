"""
MINIMALITY (Non-Target Preservation) evaluation.

PURPOSE:
    Measures if ONLY the target attribute changes in counterfactuals.
    All non-target attributes must stay the same - collateral changes are bad!

KEY METRICS:
    - cf_collateral_{attr}_flip_rate: % of CFs where a non-target binary attr flips (BAD!)
    - cf_collateral_{attr}_flip_rate_{lbl0}_to_{lbl1}: Stratified by intervention value
      (binary interventions only)
    - cf_js_{attr}_*: Jensen-Shannon divergence for each non-target attr distribution
    - cf_{attr}_dp_*: Probability delta for each non-target attr prediction
    - cf_cld_*: Contrastive Latent Divergence — DISABLED (fc_logvar not trained)
    - cf_collateral_age_delta_*: Mean/median/std |age_judge(CF) - age_judge(orig)|
      (skipped when intervention is age)

INTERPRETATION:
    Lower = better for ALL metrics
    Collateral flip rates and age delta should be ~0% (ideally <5%)
    JS divergence should be near 0 (no distribution shift)
    Probability deltas should be small (individual predictions stable)
"""

import torch
from tqdm import tqdm

# from eval.metrics.cld import CLDScorer  # disabled: fc_logvar not trained
from eval.metrics.utils import js_divergence, summarize_np, cat_list


@torch.no_grad()
def evaluate_minimality(
    loader,
    gender_judge,
    view_judge,
    disease_pa_judge,
    disease_ap_judge,
    age_judge,
    config,
    thresholds,
    intervention_name,
    intervention_key,
    val_labels,
    is_continuous,
):
    """
    Evaluate minimality (non-target attribute preservation).

    Args:
        loader: DataLoader with (img_orig, img_cf, img_null, img_rev, metas)
        gender_judge: Gender classifier
        view_judge: View classifier
        disease_pa_judge: Disease classifier for PA images
        disease_ap_judge: Disease classifier for AP images
        config: Configuration object
        thresholds: Dict with optimal decision thresholds (e.g., {"gender": 0.5, "view": 0.5, ...})
        intervention_name: Short name for the intervention (e.g., "view", "gender")
        intervention_key: Metadata key to read from metas (e.g., "View", "Sex")
        val_labels: Tuple of lowercase labels for values 0 and 1; None for continuous interventions
        is_continuous: If True, skip CLD and binary-stratified metrics
        age_judge: Regression judge for age collateral/recovery metrics

    Returns:
        Dictionary of minimality metrics (all prefixed with 'cf_')
    """
    gender_threshold = thresholds.get("gender", 0.5)
    view_threshold = thresholds.get("view", 0.5)
    disease_pa_threshold = thresholds.get("disease_pa", 0.5)
    disease_ap_threshold = thresholds.get("disease_ap", 0.5)

    gender_judge.eval()
    view_judge.eval()
    disease_pa_judge.eval()
    disease_ap_judge.eval()
    age_judge.eval()

    n = 0

    # Non-target preservation metrics
    js_gender, js_view, js_dis = [], [], []
    gender_dp, view_dp, dis_dp = [], [], []
    gender_flip, view_flip, dis_flip = 0, 0, 0

    # Stratified metrics by intervention value (binary only)
    n_0, n_1 = 0, 0
    gender_flip_0, gender_flip_1 = 0, 0
    view_flip_0, view_flip_1 = 0, 0
    dis_flip_0, dis_flip_1 = 0, 0
    lbl0, lbl1 = val_labels if not is_continuous else (None, None)

    # Age collateral
    age_delta = []

    # Pass 1: Compute metrics
    # cld_scorer = CLDScorer(latent_dim=config.CLD_LATENT_DIM, device=config.DEVICE) if not is_continuous else None  # disabled: fc_logvar not trained

    for img_orig, img_cf, img_null, img_rev, metas in tqdm(
        loader, desc="eval[minimality] (pass 1)"
    ):
        img_orig = img_orig.to(config.DEVICE)
        img_cf = img_cf.to(config.DEVICE)
        intvn_orig = metas[intervention_key].to(config.DEVICE)

        bs = img_orig.size(0)
        n += bs

        # Stratification and CLD caching (binary only — skip for continuous)
        if not is_continuous:
            mask_0 = (intvn_orig == 0)
            mask_1 = (intvn_orig == 1)
            n_0 += mask_0.sum().item()
            n_1 += mask_1.sum().item()
            # cld_scorer.update(img_orig, intvn_orig)  # disabled

        # Gender/view predictions
        p_orig_gender = gender_judge.predict_probs(img_orig)
        p_cf_gender = gender_judge.predict_probs(img_cf)
        p_orig_view = view_judge.predict_probs(img_orig)
        p_cf_view = view_judge.predict_probs(img_cf)

        # View predictions for disease routing (using optimal threshold)
        pred_orig_view = (p_orig_view[:, 1] >= view_threshold).long()  # 0=PA, 1=AP
        pred_cf_view = (p_cf_view[:, 1] >= view_threshold).long()

        # Route disease predictions based on view (batched for efficiency)
        p_orig_dis = torch.zeros(bs, 2, device=config.DEVICE)
        p_cf_dis = torch.zeros(bs, 2, device=config.DEVICE)

        # Routing is exhaustive: pred_view is always 0 or 1, so every sample lands in
        # either pa_mask or ap_mask.
        # NOTE: CF routing uses the view judge's prediction on the CF image, not the
        # ground-truth target view. For view interventions, a successfully flipped CF
        # routes to the target judge; an unsuccessful flip stays with the original judge.
        pa_mask_orig = pred_orig_view == 0
        ap_mask_orig = pred_orig_view == 1
        if pa_mask_orig.any():
            p_orig_dis[pa_mask_orig] = disease_pa_judge.predict_probs(
                img_orig[pa_mask_orig]
            )
        if ap_mask_orig.any():
            p_orig_dis[ap_mask_orig] = disease_ap_judge.predict_probs(
                img_orig[ap_mask_orig]
            )

        pa_mask_cf = pred_cf_view == 0
        ap_mask_cf = pred_cf_view == 1
        if pa_mask_cf.any():
            p_cf_dis[pa_mask_cf] = disease_pa_judge.predict_probs(img_cf[pa_mask_cf])
        if ap_mask_cf.any():
            p_cf_dis[ap_mask_cf] = disease_ap_judge.predict_probs(img_cf[ap_mask_cf])

        # Predictions (using optimal thresholds)
        pred_orig_gender = (p_orig_gender[:, 1] >= gender_threshold).long()
        pred_cf_gender = (p_cf_gender[:, 1] >= gender_threshold).long()

        # Disease predictions use view-specific thresholds
        pred_orig_dis = torch.zeros(bs, dtype=torch.long, device=config.DEVICE)
        pred_cf_dis = torch.zeros(bs, dtype=torch.long, device=config.DEVICE)
        pred_orig_dis[pa_mask_orig] = (p_orig_dis[pa_mask_orig, 1] >= disease_pa_threshold).long()
        pred_orig_dis[ap_mask_orig] = (p_orig_dis[ap_mask_orig, 1] >= disease_ap_threshold).long()
        pred_cf_dis[pa_mask_cf] = (p_cf_dis[pa_mask_cf, 1] >= disease_pa_threshold).long()
        pred_cf_dis[ap_mask_cf] = (p_cf_dis[ap_mask_cf, 1] >= disease_ap_threshold).long()

        # Distribution-level preservation (JS divergence)
        js_gender.append(js_divergence(p_orig_gender, p_cf_gender).cpu())
        js_view.append(js_divergence(p_orig_view, p_cf_view).cpu())
        js_dis.append(js_divergence(p_orig_dis, p_cf_dis).cpu())

        # Individual prediction changes (probability delta)
        gender_dp.append(torch.abs(p_orig_gender[:, 1] - p_cf_gender[:, 1]).cpu())
        view_dp.append(torch.abs(p_orig_view[:, 1] - p_cf_view[:, 1]).cpu())
        dis_dp.append(torch.abs(p_orig_dis[:, 1] - p_cf_dis[:, 1]).cpu())

        # Hard prediction flips (CRITICAL: should be ~0%)
        gender_flip += (pred_cf_gender != pred_orig_gender).sum().item()
        view_flip += (pred_cf_view != pred_orig_view).sum().item()
        dis_flip += (pred_cf_dis != pred_orig_dis).sum().item()

        # Stratified flips by intervention value (binary only)
        if not is_continuous:
            if mask_0.any():
                gender_flip_0 += (pred_cf_gender[mask_0] != pred_orig_gender[mask_0]).sum().item()
                view_flip_0 += (pred_cf_view[mask_0] != pred_orig_view[mask_0]).sum().item()
                dis_flip_0 += (pred_cf_dis[mask_0] != pred_orig_dis[mask_0]).sum().item()
            if mask_1.any():
                gender_flip_1 += (pred_cf_gender[mask_1] != pred_orig_gender[mask_1]).sum().item()
                view_flip_1 += (pred_cf_view[mask_1] != pred_orig_view[mask_1]).sum().item()
                dis_flip_1 += (pred_cf_dis[mask_1] != pred_orig_dis[mask_1]).sum().item()

        # Age collateral delta (CF should preserve apparent age unless age is the intervention)
        age_delta.append((age_judge(img_cf) - age_judge(img_orig)).abs().cpu())

    # Pass 2: CLD disabled (fc_logvar not trained)
    # cld_scores = None
    # if not is_continuous:
    #     cld_scores_list = []
    #     for img_orig, img_cf, _, _, metas in tqdm(
    #         loader, desc="eval[minimality] (pass 2: CLD)"
    #     ):
    #         img_orig = img_orig.to(config.DEVICE)
    #         img_cf = img_cf.to(config.DEVICE)
    #         intvn_orig = metas[intervention_key].to(config.DEVICE)
    #         intvn_cf = 1 - intvn_orig
    #
    #         cld_batch = cld_scorer.compute(img_orig, img_cf, intvn_orig, intvn_cf)
    #         cld_scores_list.append(cld_batch.cpu())
    #
    #     cld_scores = torch.cat(cld_scores_list).numpy()

    # Aggregate results
    # Non-target collateral preservation (emit for all attributes except the intervention)
    _collateral = {
        "gender":  (gender_flip, gender_flip_0, gender_flip_1, cat_list(js_gender), cat_list(gender_dp)),
        "view":    (view_flip,   view_flip_0,   view_flip_1,   cat_list(js_view),   cat_list(view_dp)),
        "disease": (dis_flip,    dis_flip_0,    dis_flip_1,    cat_list(js_dis),    cat_list(dis_dp)),
    }

    results = {}
    # CLD disabled (fc_logvar not trained)
    # if not is_continuous:
    #     results["cf_cld_mean"]   = summarize_np(cld_scores)["mean"]
    #     results["cf_cld_median"] = summarize_np(cld_scores)["median"]
    #     results["cf_cld_std"]    = summarize_np(cld_scores)["std"]

    for attr, (flip, flip_0, flip_1, js_np, dp_np) in _collateral.items():
        if attr == intervention_name:
            continue
        results[f"cf_collateral_{attr}_flip_rate"] = float(flip / max(n, 1))
        if not is_continuous:
            results[f"cf_collateral_{attr}_flip_rate_{lbl0}_to_{lbl1}"] = float(flip_0 / max(n_0, 1))
            results[f"cf_collateral_{attr}_flip_rate_{lbl1}_to_{lbl0}"] = float(flip_1 / max(n_1, 1))
        results[f"cf_js_{attr}_mean"]   = summarize_np(js_np)["mean"]
        results[f"cf_js_{attr}_median"] = summarize_np(js_np)["median"]
        results[f"cf_js_{attr}_std"]    = summarize_np(js_np)["std"]
        results[f"cf_{attr}_dp_mean"]   = summarize_np(dp_np)["mean"]
        results[f"cf_{attr}_dp_median"] = summarize_np(dp_np)["median"]
        results[f"cf_{attr}_dp_std"]    = summarize_np(dp_np)["std"]

    if intervention_name != "age":
        stats = summarize_np(cat_list(age_delta))
        results["cf_collateral_age_delta_mean"]   = stats["mean"]
        results["cf_collateral_age_delta_median"] = stats["median"]
        results["cf_collateral_age_delta_std"]    = stats["std"]

    return results
