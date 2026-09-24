"""
REALISM (Distribution Fidelity) evaluation.

METRIC TYPES:
- Distribution metrics (FID, KID): Compare CF distribution vs reference distribution
  * With train_ref_loader: img_cf (test) vs originals (train)
  * Without train_ref_loader: img_cf vs img_orig pairs (both test)
- Paired sample metrics (SSIM, LPIPS, feat_dist, MSE, PSNR, L1): img_cf vs img_orig pairs (both test)

PURPOSE:
    Measures if counterfactuals are realistic and plausible.
    CFs should be indistinguishable from real images at distribution level.

KEY METRICS:
    - orig_cf_fid: Fréchet Inception Distance (distribution-level realism)
    - orig_cf_kid: Kernel Inception Distance (polynomial MMD on Inception features)
    - orig_cf_ssim: Structural similarity (not too different, not too same)
    - orig_cf_lpips: Learned perceptual similarity
    - orig_cf_feat_dist: Feature-space distance (ResNet18)

INTERPRETATION:
    Lower FID/KID = better (CFs closer to real image distribution)
    SSIM should be moderate (too high = not enough change, too low = unrealistic)
    Lower LPIPS/feat_dist = better perceptual quality

NOTES:
    FID and KID process ALL images (no subsampling).
    Previous MMD metric removed - redundant with KID and suffered from biased subsampling.
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
from eval.metrics.utils import summarize_np, cat_list, denorm01


@torch.no_grad()
def evaluate_realism(
    loader,
    feat_enc,
    config,
    intervention_name,
    intervention_key,
    val_labels,
    is_continuous,
    train_ref_loader=None,
):
    """
    Evaluate realism (distribution fidelity of counterfactuals).

    Args:
        loader: Test split DataLoader with (img_orig, img_cf, img_null, img_rev, metas)
        feat_enc: Feature encoder (ResNet18 for paired feature distance)
        config: Configuration object
        intervention_name: Short name for the intervention (e.g., "view", "gender")
        intervention_key: Metadata key to read from metas (e.g., "View", "Sex")
        val_labels: Tuple of lowercase labels for values 0 and 1; None for continuous interventions
        is_continuous: If True, skip stratified perceptual metrics
        train_ref_loader: Optional train split loader for FID/KID reference distribution.
                         If provided, compares CF (test) vs originals (train).
                         If None, compares CF vs originals (both test, paired).

    Returns:
        Dictionary of realism metrics (all prefixed with 'orig_cf_')
        - FID/KID: ALL samples (no subsampling)
        - Pixel metrics: ALL samples
        - Feature distance: ALL samples
    """
    feat_enc.eval()

    n = 0
    lbl0, lbl1 = val_labels if not is_continuous else (None, None)

    # Pixel metrics (always paired on test split)
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

    # Stratified perceptual metrics by intervention value (categorical only)
    pix_metrics_0 = {"ssim": [], "lpips": []}
    pix_metrics_1 = {"ssim": [], "lpips": []}
    feat_dist_0 = []
    feat_dist_1 = []

    # FID and KID scorers
    fid_scorer = FIDScorer(device=config.DEVICE)
    kid_scorer = KIDScorer(device=config.DEVICE, subset_size=100)

    # Pass 1: Accumulate train reference distribution if provided
    if train_ref_loader is not None:
        for images, _, _ in tqdm(train_ref_loader, desc="eval[realism] (train ref)"):
            images = images.to(config.DEVICE)

            # FID and KID - accumulate train originals as "real" distribution only
            fid_scorer.update_real(images)
            kid_scorer.update_real(images)

    # Pass 2: Process test split (paired metrics + CF for distribution)
    for img_orig, img_cf, img_null, img_rev, metas in tqdm(
        loader, desc="eval[realism] (test CF)"
    ):
        img_orig = img_orig.to(config.DEVICE)
        img_cf = img_cf.to(config.DEVICE)

        bs = img_orig.size(0)
        n += bs

        # Compute once; .cpu() returns a new tensor, so originals stay on device
        # for stratified masking below
        ssim_vals = ssim_per_image(img_orig, img_cf)
        lpips_vals = lpips_distance(img_orig, img_cf, net=config.LPIPS_NET, device=config.DEVICE)

        # Pixel metrics (paired - test split only)
        pix_metrics["mse"].append(mse_per_image(img_orig, img_cf).cpu())
        pix_metrics["psnr"].append(psnr_per_image(img_orig, img_cf).cpu())
        pix_metrics["ssim"].append(ssim_vals.cpu())
        pix_metrics["l1"].append(l1_distance(img_orig, img_cf).cpu())
        pix_metrics["mae"].append(mean_absolute_error(img_orig, img_cf).cpu())
        pix_metrics["lpips"].append(lpips_vals.cpu())

        # FID and KID - add CF to distribution
        if train_ref_loader is None:
            # Paired test approach: compare test originals (real) vs test CFs (fake)
            fid_scorer.update(img_orig, img_cf)
            kid_scorer.update(img_orig, img_cf)
        else:
            # Train reference approach: CFs as "fake" vs train originals (already accumulated as "real")
            fid_scorer.update_fake(img_cf)
            kid_scorer.update_fake(img_cf)

        # Feature distance (paired - test split only)
        f_orig = feat_enc(img_orig)
        f_cf = feat_enc(img_cf)
        dist = torch.norm(f_cf - f_orig, dim=1)
        feat_dist.append(dist.cpu())

        # Stratified perceptual metrics by intervention value (categorical only)
        if not is_continuous:
            intvn_orig = metas[intervention_key].to(config.DEVICE)
            mask_0 = intvn_orig == 0
            mask_1 = intvn_orig == 1
            # ssim_vals and lpips_vals already computed above
            if mask_0.any():
                pix_metrics_0["ssim"].append(ssim_vals[mask_0].cpu())
                pix_metrics_0["lpips"].append(lpips_vals[mask_0].cpu())
                feat_dist_0.append(dist[mask_0].cpu())
            if mask_1.any():
                pix_metrics_1["ssim"].append(ssim_vals[mask_1].cpu())
                pix_metrics_1["lpips"].append(lpips_vals[mask_1].cpu())
                feat_dist_1.append(dist[mask_1].cpu())

    # Aggregate results
    results = {}

    # Pixel metrics
    for metric in ["mse", "psnr", "ssim", "l1", "mae", "lpips"]:
        vals = cat_list(pix_metrics[metric])
        stats = summarize_np(vals)
        results[f"orig_cf_{metric}_mean"] = stats["mean"]
        results[f"orig_cf_{metric}_median"] = stats["median"]
        results[f"orig_cf_{metric}_std"] = stats["std"]

    # Feature distance (paired)
    dist_vals = cat_list(feat_dist)
    stats = summarize_np(dist_vals)
    results["orig_cf_feat_dist_mean"] = stats["mean"]
    results["orig_cf_feat_dist_median"] = stats["median"]
    results["orig_cf_feat_dist_std"] = stats["std"]
    results["orig_cf_feat_dist_p95"] = stats["p95"]

    # Stratified perceptual metrics (categorical interventions only)
    if not is_continuous:
        for metric, vals_0, vals_1 in [
            ("ssim", cat_list(pix_metrics_0["ssim"]), cat_list(pix_metrics_1["ssim"])),
            ("lpips", cat_list(pix_metrics_0["lpips"]), cat_list(pix_metrics_1["lpips"])),
        ]:
            for suffix, vals in [(f"{lbl0}_to_{lbl1}", vals_0), (f"{lbl1}_to_{lbl0}", vals_1)]:
                s = summarize_np(vals)
                results[f"orig_cf_{metric}_mean_{suffix}"] = s["mean"]
                results[f"orig_cf_{metric}_median_{suffix}"] = s["median"]
                results[f"orig_cf_{metric}_std_{suffix}"] = s["std"]
        for suffix, vals in [
            (f"{lbl0}_to_{lbl1}", cat_list(feat_dist_0)),
            (f"{lbl1}_to_{lbl0}", cat_list(feat_dist_1)),
        ]:
            s = summarize_np(vals)
            results[f"orig_cf_feat_dist_mean_{suffix}"] = s["mean"]
            results[f"orig_cf_feat_dist_median_{suffix}"] = s["median"]
            results[f"orig_cf_feat_dist_std_{suffix}"] = s["std"]

    # FID (distribution-level realism)
    results["orig_cf_fid"] = float(fid_scorer.compute())
    results["orig_cf_fid_n"] = float(n)

    # KID (kernel-based distribution distance - polynomial MMD on Inception features)
    kid_mean, kid_std = kid_scorer.compute()
    results["orig_cf_kid_mean"] = float(kid_mean)
    results["orig_cf_kid_std"] = float(kid_std)

    return results
