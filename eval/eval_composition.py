"""
COMPOSITION (Identity Preservation) evaluation.

PURPOSE:
    Measures drift when NO intervention is applied (orig_null pair).
    Null interventions should change nothing - low drift indicates good composition.

KEY METRICS:
    - orig_null_mse/psnr/ssim/l1/mae/lpips: Pixel-level similarity
    - orig_null_feat_dist: Feature-space similarity
    - orig_null_fid, orig_null_kid: Distribution-level similarity
    - null_{intervention}_consistency: % of nulls preserving the intervention attr (~100% expected;
      binary interventions only)
    - null_collateral_{attr}_flip_rate: Undesired flip rate for each non-intervention binary attr
    - null_collateral_age_delta_*: Mean/median/std |age_judge(null) − age_judge(orig)|
      (skipped when intervention is age)

INTERPRETATION:
    Lower MSE/L1/LPIPS/FID/KID = better (less drift)
    Higher PSNR/SSIM = better (more similar)
    null_{intervention}_consistency should be ~100% (binary interventions only)
    Collateral flip rates and age delta should be ~0%

NOTES:
    FID and KID process ALL samples (no subsampling).
"""

import torch
from tqdm import tqdm

from eval.metrics.fid import FIDScorer
from eval.metrics.kid import KIDScorer
from eval.metrics.distances import (
    l1_distance,
    mean_absolute_error,
    lpips_distance,
    mse_per_image,
    psnr_per_image,
    ssim_per_image,
)
from eval.metrics.utils import summarize_np, cat_list, js_divergence


@torch.no_grad()
def evaluate_composition(
    loader,
    gender_judge,
    view_judge,
    disease_pa_judge,
    disease_ap_judge,
    age_judge,
    feat_enc,
    config,
    thresholds,
    intervention_name,
    intervention_key,
    is_continuous,
):
    """
    Evaluate composition (identity preservation under null intervention).

    Args:
        loader: DataLoader with (img_orig, img_cf, img_null, img_rev, metas)
        gender_judge: Gender classifier
        view_judge: View classifier
        disease_pa_judge: Disease classifier for PA images
        disease_ap_judge: Disease classifier for AP images
        feat_enc: Feature encoder
        config: Configuration object
        thresholds: Dict with optimal decision thresholds (e.g., {"gender": 0.5, "view": 0.5, ...})
        intervention_name: Short name for the intervention (e.g., "view", "gender")
        intervention_key: Metadata key to read from metas (e.g., "View", "Sex")
        age_judge: Regression judge for age collateral metrics

    Returns:
        Dictionary of composition metrics (all prefixed with 'null_' or 'orig_null_')
    """
    gender_threshold = thresholds.get("gender", 0.5)
    view_threshold = thresholds.get("view", 0.5)
    disease_pa_threshold = thresholds.get("disease_pa", 0.5)
    disease_ap_threshold = thresholds.get("disease_ap", 0.5)

    gender_judge.eval()
    view_judge.eval()
    disease_pa_judge.eval()
    disease_ap_judge.eval()
    feat_enc.eval()
    age_judge.eval()

    n = 0
    null_intvn_same_ok = 0

    # Pixel metrics
    pix_metrics = {
        "mse": [],
        "psnr": [],
        "ssim": [],
        "l1": [],
        "mae": [],
        "lpips": [],
    }

    # Feature metrics
    feat_dist = []

    # Age collateral
    age_delta = []

    # FID and KID scorers
    fid_scorer = FIDScorer(device=config.DEVICE)
    kid_scorer = KIDScorer(device=config.DEVICE, subset_size=100)

    # Non-target preservation
    js_gender, js_view, js_dis = [], [], []
    gender_dp, view_dp, dis_dp = [], [], []
    gender_flip, view_flip, dis_flip = 0, 0, 0

    for img_orig, img_cf, img_null, img_rev, metas in tqdm(
        loader, desc="eval[composition]"
    ):
        img_orig = img_orig.to(config.DEVICE)
        img_null = img_null.to(config.DEVICE)
        intvn_orig = metas[intervention_key].to(config.DEVICE)

        bs = img_orig.size(0)
        n += bs

        # Gender/view predictions
        p_orig_gender = gender_judge.predict_probs(img_orig)
        p_null_gender = gender_judge.predict_probs(img_null)
        p_orig_view = view_judge.predict_probs(img_orig)
        p_null_view = view_judge.predict_probs(img_null)

        # View predictions for disease routing (using optimal threshold)
        pred_orig_view = (p_orig_view[:, 1] >= view_threshold).long()
        pred_null_view = (p_null_view[:, 1] >= view_threshold).long()

        # Route disease predictions based on view (batched for efficiency)
        p_orig_dis = torch.zeros(bs, 2, device=config.DEVICE)
        p_null_dis = torch.zeros(bs, 2, device=config.DEVICE)

        # Routing is exhaustive: pred_view is always 0 or 1, so every sample lands in
        # either pa_mask or ap_mask.
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

        pa_mask_null = pred_null_view == 0
        ap_mask_null = pred_null_view == 1
        if pa_mask_null.any():
            p_null_dis[pa_mask_null] = disease_pa_judge.predict_probs(
                img_null[pa_mask_null]
            )
        if ap_mask_null.any():
            p_null_dis[ap_mask_null] = disease_ap_judge.predict_probs(
                img_null[ap_mask_null]
            )

        # Predictions (using optimal thresholds)
        pred_orig_gender = (p_orig_gender[:, 1] >= gender_threshold).long()
        pred_null_gender = (p_null_gender[:, 1] >= gender_threshold).long()

        # Disease predictions use view-specific thresholds
        pred_orig_dis = torch.zeros(bs, dtype=torch.long, device=config.DEVICE)
        pred_null_dis = torch.zeros(bs, dtype=torch.long, device=config.DEVICE)
        pred_orig_dis[pa_mask_orig] = (
            p_orig_dis[pa_mask_orig, 1] >= disease_pa_threshold
        ).long()
        pred_orig_dis[ap_mask_orig] = (
            p_orig_dis[ap_mask_orig, 1] >= disease_ap_threshold
        ).long()
        pred_null_dis[pa_mask_null] = (
            p_null_dis[pa_mask_null, 1] >= disease_pa_threshold
        ).long()
        pred_null_dis[ap_mask_null] = (
            p_null_dis[ap_mask_null, 1] >= disease_ap_threshold
        ).long()

        # Null should preserve the intervention attribute (binary only — continuous skipped)
        _null_preds = {"view": pred_null_view, "gender": pred_null_gender, "disease": pred_null_dis}
        if not is_continuous:
            null_intvn_same_ok += (_null_preds[intervention_name] == intvn_orig).sum().item()

        # Non-target preservation
        js_gender.append(js_divergence(p_orig_gender, p_null_gender).cpu())
        js_view.append(js_divergence(p_orig_view, p_null_view).cpu())
        js_dis.append(js_divergence(p_orig_dis, p_null_dis).cpu())
        gender_dp.append(torch.abs(p_orig_gender[:, 1] - p_null_gender[:, 1]).cpu())
        view_dp.append(torch.abs(p_orig_view[:, 1] - p_null_view[:, 1]).cpu())
        dis_dp.append(torch.abs(p_orig_dis[:, 1] - p_null_dis[:, 1]).cpu())
        gender_flip += (pred_null_gender != pred_orig_gender).sum().item()
        view_flip += (pred_null_view != pred_orig_view).sum().item()
        dis_flip += (pred_null_dis != pred_orig_dis).sum().item()

        # Pixel metrics
        pix_metrics["mse"].append(mse_per_image(img_orig, img_null).cpu())
        pix_metrics["psnr"].append(psnr_per_image(img_orig, img_null).cpu())
        pix_metrics["ssim"].append(ssim_per_image(img_orig, img_null).cpu())
        pix_metrics["l1"].append(l1_distance(img_orig, img_null).cpu())
        pix_metrics["mae"].append(mean_absolute_error(img_orig, img_null).cpu())
        pix_metrics["lpips"].append(
            lpips_distance(
                img_orig, img_null, net=config.LPIPS_NET, device=config.DEVICE
            ).cpu()
        )

        # FID and KID
        fid_scorer.update(img_orig, img_null)
        kid_scorer.update(img_orig, img_null)

        # Feature metrics
        f_orig = feat_enc(img_orig)
        f_null = feat_enc(img_null)
        dist = torch.norm(f_null - f_orig, dim=1)
        feat_dist.append(dist.cpu())

        # Age collateral delta (null should preserve apparent age)
        age_delta.append((age_judge(img_null) - age_judge(img_orig)).abs().cpu())

    # Aggregate results
    results = {}
    if not is_continuous:
        results[f"null_{intervention_name}_consistency"] = float(null_intvn_same_ok / max(n, 1))

    # Non-target collateral preservation (emit for all attributes except the intervention)
    _collateral = {
        "gender":  (gender_flip, cat_list(js_gender), cat_list(gender_dp)),
        "view":    (view_flip,   cat_list(js_view),   cat_list(view_dp)),
        "disease": (dis_flip,    cat_list(js_dis),    cat_list(dis_dp)),
    }

    for attr, (flip, js_np, dp_np) in _collateral.items():
        if attr == intervention_name:
            continue
        results[f"null_collateral_{attr}_flip_rate"] = float(flip / max(n, 1))
        results[f"null_js_{attr}_mean"]   = summarize_np(js_np)["mean"]
        results[f"null_js_{attr}_median"] = summarize_np(js_np)["median"]
        results[f"null_js_{attr}_std"]    = summarize_np(js_np)["std"]
        results[f"null_{attr}_dp_mean"]   = summarize_np(dp_np)["mean"]
        results[f"null_{attr}_dp_median"] = summarize_np(dp_np)["median"]
        results[f"null_{attr}_dp_std"]    = summarize_np(dp_np)["std"]

    if intervention_name != "age":
        stats = summarize_np(cat_list(age_delta))
        results["null_collateral_age_delta_mean"]   = stats["mean"]
        results["null_collateral_age_delta_median"] = stats["median"]
        results["null_collateral_age_delta_std"]    = stats["std"]

    # Pixel metrics
    for metric in ["mse", "psnr", "ssim", "l1", "mae", "lpips"]:
        vals = cat_list(pix_metrics[metric])
        stats = summarize_np(vals)
        results[f"orig_null_{metric}_mean"] = stats["mean"]
        results[f"orig_null_{metric}_median"] = stats["median"]
        results[f"orig_null_{metric}_std"] = stats["std"]

    # Feature distance
    dist_vals = cat_list(feat_dist)
    stats = summarize_np(dist_vals)
    results["orig_null_feat_dist_mean"] = stats["mean"]
    results["orig_null_feat_dist_median"] = stats["median"]
    results["orig_null_feat_dist_std"] = stats["std"]
    results["orig_null_feat_dist_p95"] = stats["p95"]

    # FID
    results["orig_null_fid"] = float(fid_scorer.compute())
    results["orig_null_fid_n"] = float(n)

    # KID
    kid_mean, kid_std = kid_scorer.compute()
    results["orig_null_kid_mean"] = float(kid_mean)
    results["orig_null_kid_std"] = float(kid_std)

    return results
