"""
REVERSIBILITY (Reconstruction Quality) evaluation.

PURPOSE:
    Measures if reverse counterfactuals can accurately reconstruct originals.
    Tests bidirectionality: orig → CF → reverse_CF should ≈ orig.

KEY METRICS:
    - orig_rev_cf_mse/psnr/ssim/l1/mae/lpips: Reconstruction quality (orig vs reverse_CF)
    - null_rev_cf_*: Control for composition drift (null vs reverse_CF)
    - orig_rev_cf_fid, orig_rev_cf_kid: Distribution-level reconstruction
    - null_rev_cf_fid, null_rev_cf_kid: Control baseline distribution
    - rev_cf_{intervention}_recovery: % reverse CFs recovering the original intervention value
      (binary interventions only)
    - orig_rev_cf_{metric}_{stat}_{lbl0}_to_{lbl1}_to_{lbl0}: Stratified perceptual metrics
      (binary interventions only)
    - rev_cf_collateral_{attr}_flip_rate: Undesired flip rate for each non-intervention binary attr
    - rev_cf_collateral_age_delta_*: Mean/median/std |age_judge(rev_CF) - age_judge(orig)|
      (skipped when intervention is age)
    - rev_cf_age_recovery_mae/median/std: Age round-trip error (continuous age intervention only)

INTERPRETATION:
    orig_rev_cf metrics: Lower MSE/L1/LPIPS/FID/KID, higher PSNR/SSIM = better reconstruction
    null_rev_cf metrics: Control baseline (net effect after accounting for drift)
    rev_cf_{intervention}_recovery should be ~100% (binary interventions only)
    Collateral flip rates and age delta should be ~0%
    rev_cf_age_recovery_mae lower = better age restoration after round-trip

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
def evaluate_reversibility(
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
    val_labels,
    is_continuous,
):
    """
    Evaluate reversibility (reconstruction quality via reverse counterfactuals).

    Args:
        loader: DataLoader with (img_orig, img_cf, img_null, img_rev_cf, metas)
        gender_judge: Gender classifier
        view_judge: View classifier
        disease_pa_judge: Disease classifier for PA images
        disease_ap_judge: Disease classifier for AP images
        feat_enc: Feature encoder
        config: Configuration object
        thresholds: Dict with optimal decision thresholds (e.g., {"gender": 0.5, "view": 0.5, ...})
        intervention_name: Short name for the intervention (e.g., "view", "gender")
        intervention_key: Metadata key to read from metas (e.g., "View", "Sex")
        val_labels: Tuple of lowercase labels for values 0 and 1; None for continuous interventions
        is_continuous: If True, skip binary recovery and stratified perceptual metrics
        age_judge: Regression judge for age collateral/recovery metrics

    Returns:
        Dictionary of reversibility metrics (prefixed with 'orig_rev_cf_', 'null_rev_cf_', 'rev_cf_')
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
    rev_cf_intvn_recover_ok = 0
    lbl0, lbl1 = val_labels if not is_continuous else (None, None)

    # Age recovery/collateral
    age_delta = []

    # Pixel metrics for BOTH pairs (orig_rev_cf and null_rev_cf)
    pix_metrics = {
        "orig_rev_cf": {
            "mse": [],
            "psnr": [],
            "ssim": [],
            "l1": [],
            "mae": [],
            "lpips": [],
        },
        "null_rev_cf": {
            "mse": [],
            "psnr": [],
            "ssim": [],
            "l1": [],
            "mae": [],
            "lpips": [],
        },
    }

    # Feature metrics
    feat_dist = {"orig_rev_cf": [], "null_rev_cf": []}

    # Stratified metrics by intervention value
    n_0, n_1 = 0, 0
    rev_cf_intvn_recover_ok_0, rev_cf_intvn_recover_ok_1 = 0, 0
    pix_metrics_pa = {"ssim": [], "lpips": []}
    pix_metrics_ap = {"ssim": [], "lpips": []}
    feat_dist_pa = []
    feat_dist_ap = []

    # FID and KID scorers
    fid_scorers = {
        "orig_rev_cf": FIDScorer(device=config.DEVICE),
        "null_rev_cf": FIDScorer(device=config.DEVICE),
    }
    kid_scorers = {
        "orig_rev_cf": KIDScorer(device=config.DEVICE, subset_size=100),
        "null_rev_cf": KIDScorer(device=config.DEVICE, subset_size=100),
    }

    # Non-target preservation
    js_gender, js_view, js_dis = [], [], []
    gender_dp, view_dp, dis_dp = [], [], []
    gender_flip, view_flip, dis_flip = 0, 0, 0

    for img_orig, img_cf, img_null, img_rev_cf, metas in tqdm(
        loader, desc="eval[reversibility]"
    ):
        img_orig = img_orig.to(config.DEVICE)
        img_null = img_null.to(config.DEVICE)
        img_rev_cf = img_rev_cf.to(config.DEVICE)
        intvn_orig = metas[intervention_key].to(config.DEVICE)

        bs = img_orig.size(0)
        n += bs

        # Stratification masks by intervention value
        mask_0 = intvn_orig == 0
        mask_1 = intvn_orig == 1
        n_0 += mask_0.sum().item()
        n_1 += mask_1.sum().item()

        # Gender/view predictions
        p_orig_gender = gender_judge.predict_probs(img_orig)
        p_rev_cf_gender = gender_judge.predict_probs(img_rev_cf)
        p_orig_view = view_judge.predict_probs(img_orig)
        p_rev_cf_view = view_judge.predict_probs(img_rev_cf)

        # View predictions for disease routing (using optimal threshold)
        pred_orig_view = (p_orig_view[:, 1] >= view_threshold).long()
        pred_rev_cf_view = (p_rev_cf_view[:, 1] >= view_threshold).long()

        # Route disease predictions based on view (batched for efficiency)
        p_orig_dis = torch.zeros(bs, 2, device=config.DEVICE)
        p_rev_cf_dis = torch.zeros(bs, 2, device=config.DEVICE)

        # Routing is exhaustive: pred_view is always 0 or 1, so every sample lands in
        # either pa_mask or ap_mask.
        # NOTE: rev_cf routing uses the view judge's prediction on the rev_cf image, not
        # the ground-truth target view. For view interventions, a successfully flipped
        # rev_cf routes to the target judge; an unsuccessful flip stays with the original judge.
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

        pa_mask_rev_cf = pred_rev_cf_view == 0
        ap_mask_rev_cf = pred_rev_cf_view == 1
        if pa_mask_rev_cf.any():
            p_rev_cf_dis[pa_mask_rev_cf] = disease_pa_judge.predict_probs(
                img_rev_cf[pa_mask_rev_cf]
            )
        if ap_mask_rev_cf.any():
            p_rev_cf_dis[ap_mask_rev_cf] = disease_ap_judge.predict_probs(
                img_rev_cf[ap_mask_rev_cf]
            )

        # Predictions (using optimal thresholds)
        pred_orig_gender = (p_orig_gender[:, 1] >= gender_threshold).long()
        pred_rev_cf_gender = (p_rev_cf_gender[:, 1] >= gender_threshold).long()

        # Disease predictions use view-specific thresholds
        pred_orig_dis = torch.zeros(bs, dtype=torch.long, device=config.DEVICE)
        pred_rev_cf_dis = torch.zeros(bs, dtype=torch.long, device=config.DEVICE)
        pred_orig_dis[pa_mask_orig] = (
            p_orig_dis[pa_mask_orig, 1] >= disease_pa_threshold
        ).long()
        pred_orig_dis[ap_mask_orig] = (
            p_orig_dis[ap_mask_orig, 1] >= disease_ap_threshold
        ).long()
        pred_rev_cf_dis[pa_mask_rev_cf] = (
            p_rev_cf_dis[pa_mask_rev_cf, 1] >= disease_pa_threshold
        ).long()
        pred_rev_cf_dis[ap_mask_rev_cf] = (
            p_rev_cf_dis[ap_mask_rev_cf, 1] >= disease_ap_threshold
        ).long()

        # Intervention attribute recovery (binary only — continuous skipped)
        _rev_cf_preds = {"view": pred_rev_cf_view, "gender": pred_rev_cf_gender, "disease": pred_rev_cf_dis}
        if not is_continuous:
            pred_rev_cf_intvn = _rev_cf_preds[intervention_name]
            rev_cf_intvn_recover_ok += (pred_rev_cf_intvn == intvn_orig).sum().item()
            if mask_0.any():
                rev_cf_intvn_recover_ok_0 += (
                    (pred_rev_cf_intvn[mask_0] == intvn_orig[mask_0]).sum().item()
                )
            if mask_1.any():
                rev_cf_intvn_recover_ok_1 += (
                    (pred_rev_cf_intvn[mask_1] == intvn_orig[mask_1]).sum().item()
                )

        # Age delta (rev_cf should restore original apparent age)
        age_delta.append((age_judge(img_rev_cf) - age_judge(img_orig)).abs().cpu())

        # Non-target preservation
        js_gender.append(js_divergence(p_orig_gender, p_rev_cf_gender).cpu())
        js_view.append(js_divergence(p_orig_view, p_rev_cf_view).cpu())
        js_dis.append(js_divergence(p_orig_dis, p_rev_cf_dis).cpu())
        gender_dp.append(torch.abs(p_orig_gender[:, 1] - p_rev_cf_gender[:, 1]).cpu())
        view_dp.append(torch.abs(p_orig_view[:, 1] - p_rev_cf_view[:, 1]).cpu())
        dis_dp.append(torch.abs(p_orig_dis[:, 1] - p_rev_cf_dis[:, 1]).cpu())
        gender_flip += (pred_rev_cf_gender != pred_orig_gender).sum().item()
        view_flip += (pred_rev_cf_view != pred_orig_view).sum().item()
        dis_flip += (pred_rev_cf_dis != pred_orig_dis).sum().item()

        # Pixel metrics for BOTH pairs
        pairs_imgs = {
            "orig_rev_cf": (img_orig, img_rev_cf),
            "null_rev_cf": (img_null, img_rev_cf),
        }

        # Compute ssim/lpips for orig_rev_cf once; reused for overall and stratified metrics
        ssim_orig_rev = ssim_per_image(img_orig, img_rev_cf)
        lpips_orig_rev = lpips_distance(img_orig, img_rev_cf, net=config.LPIPS_NET, device=config.DEVICE)

        for pair_key, (a, b) in pairs_imgs.items():
            pix_metrics[pair_key]["mse"].append(mse_per_image(a, b).cpu())
            pix_metrics[pair_key]["psnr"].append(psnr_per_image(a, b).cpu())
            pix_metrics[pair_key]["ssim"].append(
                ssim_orig_rev.cpu() if pair_key == "orig_rev_cf" else ssim_per_image(a, b).cpu()
            )
            pix_metrics[pair_key]["l1"].append(l1_distance(a, b).cpu())
            pix_metrics[pair_key]["mae"].append(mean_absolute_error(a, b).cpu())
            pix_metrics[pair_key]["lpips"].append(
                lpips_orig_rev.cpu() if pair_key == "orig_rev_cf"
                else lpips_distance(a, b, net=config.LPIPS_NET, device=config.DEVICE).cpu()
            )

            # FID and KID
            fid_scorers[pair_key].update(a, b)
            kid_scorers[pair_key].update(a, b)

        # Feature metrics
        f_orig = feat_enc(img_orig)
        f_null = feat_enc(img_null)
        f_rev_cf = feat_enc(img_rev_cf)

        dist_orig_rev = torch.norm(f_rev_cf - f_orig, dim=1)
        dist_null_rev = torch.norm(f_rev_cf - f_null, dim=1)
        feat_dist["orig_rev_cf"].append(dist_orig_rev.cpu())
        feat_dist["null_rev_cf"].append(dist_null_rev.cpu())

        # Stratified perceptual metrics by intervention value (categorical only)
        if not is_continuous:
            if mask_0.any():
                pix_metrics_pa["ssim"].append(ssim_orig_rev[mask_0].cpu())
                pix_metrics_pa["lpips"].append(lpips_orig_rev[mask_0].cpu())
                feat_dist_pa.append(dist_orig_rev[mask_0].cpu())
            if mask_1.any():
                pix_metrics_ap["ssim"].append(ssim_orig_rev[mask_1].cpu())
                pix_metrics_ap["lpips"].append(lpips_orig_rev[mask_1].cpu())
                feat_dist_ap.append(dist_orig_rev[mask_1].cpu())

    # Aggregate results
    # Non-target collateral preservation (emit for all attributes except the intervention)
    _collateral = {
        "gender":  (gender_flip, cat_list(js_gender), cat_list(gender_dp)),
        "view":    (view_flip,   cat_list(js_view),   cat_list(view_dp)),
        "disease": (dis_flip,    cat_list(js_dis),    cat_list(dis_dp)),
    }

    results = {}
    if not is_continuous:
        results[f"rev_cf_{intervention_name}_recovery"] = float(rev_cf_intvn_recover_ok / max(n, 1))
        results[f"rev_cf_{intervention_name}_recovery_{lbl0}_to_{lbl1}_to_{lbl0}"] = float(
            rev_cf_intvn_recover_ok_0 / max(n_0, 1)
        )
        results[f"rev_cf_{intervention_name}_recovery_{lbl1}_to_{lbl0}_to_{lbl1}"] = float(
            rev_cf_intvn_recover_ok_1 / max(n_1, 1)
        )

    for attr, (flip, js_np, dp_np) in _collateral.items():
        if attr == intervention_name:
            continue
        results[f"rev_cf_collateral_{attr}_flip_rate"] = float(flip / max(n, 1))
        results[f"rev_cf_js_{attr}_mean"]   = summarize_np(js_np)["mean"]
        results[f"rev_cf_js_{attr}_median"] = summarize_np(js_np)["median"]
        results[f"rev_cf_js_{attr}_std"]    = summarize_np(js_np)["std"]
        results[f"rev_cf_{attr}_dp_mean"]   = summarize_np(dp_np)["mean"]
        results[f"rev_cf_{attr}_dp_median"] = summarize_np(dp_np)["median"]
        results[f"rev_cf_{attr}_dp_std"]    = summarize_np(dp_np)["std"]

    # Pixel + feature metrics for BOTH pairs
    for pair_key in ["orig_rev_cf", "null_rev_cf"]:
        # Pixel metrics
        for metric in ["mse", "psnr", "ssim", "l1", "mae", "lpips"]:
            vals = cat_list(pix_metrics[pair_key][metric])
            stats = summarize_np(vals)
            results[f"{pair_key}_{metric}_mean"] = stats["mean"]
            results[f"{pair_key}_{metric}_median"] = stats["median"]
            results[f"{pair_key}_{metric}_std"] = stats["std"]

        # Feature distance
        dist_vals = cat_list(feat_dist[pair_key])
        stats = summarize_np(dist_vals)
        results[f"{pair_key}_feat_dist_mean"] = stats["mean"]
        results[f"{pair_key}_feat_dist_median"] = stats["median"]
        results[f"{pair_key}_feat_dist_std"] = stats["std"]
        results[f"{pair_key}_feat_dist_p95"] = stats["p95"]

        # FID
        results[f"{pair_key}_fid"] = float(fid_scorers[pair_key].compute())
        results[f"{pair_key}_fid_n"] = float(n)

        # KID
        kid_mean, kid_std = kid_scorers[pair_key].compute()
        results[f"{pair_key}_kid_mean"] = float(kid_mean)
        results[f"{pair_key}_kid_std"] = float(kid_std)

    # Stratified perceptual metrics by intervention value (skip if intervention is continuous)
    if not is_continuous:
        ssim_pa_vals = cat_list(pix_metrics_pa["ssim"])
        ssim_ap_vals = cat_list(pix_metrics_ap["ssim"])
        lpips_pa_vals = cat_list(pix_metrics_pa["lpips"])
        lpips_ap_vals = cat_list(pix_metrics_ap["lpips"])
        feat_dist_pa_vals = cat_list(feat_dist_pa)
        feat_dist_ap_vals = cat_list(feat_dist_ap)

        sfx0 = f"{lbl0}_to_{lbl1}_to_{lbl0}"
        sfx1 = f"{lbl1}_to_{lbl0}_to_{lbl1}"

        results[f"orig_rev_cf_ssim_mean_{sfx0}"] = summarize_np(ssim_pa_vals)["mean"]
        results[f"orig_rev_cf_ssim_median_{sfx0}"] = summarize_np(ssim_pa_vals)["median"]
        results[f"orig_rev_cf_ssim_std_{sfx0}"] = summarize_np(ssim_pa_vals)["std"]
        results[f"orig_rev_cf_ssim_mean_{sfx1}"] = summarize_np(ssim_ap_vals)["mean"]
        results[f"orig_rev_cf_ssim_median_{sfx1}"] = summarize_np(ssim_ap_vals)["median"]
        results[f"orig_rev_cf_ssim_std_{sfx1}"] = summarize_np(ssim_ap_vals)["std"]
        results[f"orig_rev_cf_lpips_mean_{sfx0}"] = summarize_np(lpips_pa_vals)["mean"]
        results[f"orig_rev_cf_lpips_median_{sfx0}"] = summarize_np(lpips_pa_vals)["median"]
        results[f"orig_rev_cf_lpips_std_{sfx0}"] = summarize_np(lpips_pa_vals)["std"]
        results[f"orig_rev_cf_lpips_mean_{sfx1}"] = summarize_np(lpips_ap_vals)["mean"]
        results[f"orig_rev_cf_lpips_median_{sfx1}"] = summarize_np(lpips_ap_vals)["median"]
        results[f"orig_rev_cf_lpips_std_{sfx1}"] = summarize_np(lpips_ap_vals)["std"]
        results[f"orig_rev_cf_feat_dist_mean_{sfx0}"] = summarize_np(feat_dist_pa_vals)["mean"]
        results[f"orig_rev_cf_feat_dist_median_{sfx0}"] = summarize_np(feat_dist_pa_vals)["median"]
        results[f"orig_rev_cf_feat_dist_std_{sfx0}"] = summarize_np(feat_dist_pa_vals)["std"]
        results[f"orig_rev_cf_feat_dist_mean_{sfx1}"] = summarize_np(feat_dist_ap_vals)["mean"]
        results[f"orig_rev_cf_feat_dist_median_{sfx1}"] = summarize_np(feat_dist_ap_vals)["median"]
        results[f"orig_rev_cf_feat_dist_std_{sfx1}"] = summarize_np(feat_dist_ap_vals)["std"]

    # Age metrics: recovery when age is the intervention; collateral otherwise
    age_stats = summarize_np(cat_list(age_delta))
    if intervention_name == "age":
        results["rev_cf_age_recovery_mae"]    = age_stats["mean"]
        results["rev_cf_age_recovery_median"] = age_stats["median"]
        results["rev_cf_age_recovery_std"]    = age_stats["std"]
    else:
        results["rev_cf_collateral_age_delta_mean"]   = age_stats["mean"]
        results["rev_cf_collateral_age_delta_median"] = age_stats["median"]
        results["rev_cf_collateral_age_delta_std"]    = age_stats["std"]

    return results
