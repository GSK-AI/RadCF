"""Distance and similarity metrics for CIG evaluation.

Contains pixel-level, perceptual, and structural similarity metrics.
"""

import lpips
import torch
import torch.nn.functional as F

from eval.metrics.utils import _validate_range_minus1_to_1, denorm01


# Module-level cache for LPIPS models to prevent memory leak
_LPIPS_CACHE = {}


def mean_absolute_error(original, edited):
    """
    Calculates the mean absolute error between original and edited images across pixels.
    - To evaluate CIG Composition: Compare factual images to counterfactuals after applying several rounds of null-intervention.
    - To evaluate CIG Reversibility: Compare factual images to reversed counterfactuals.

    Args:
        original: Tensor of shape (B, C, H, W) in range [-1, 1]
        edited: Tensor of shape (B, C, H, W) in range [-1, 1]

    Returns:
        Tensor of shape (B,) containing mean absolute error per sample
    """
    assert (
        original.dim() == 4 and edited.dim() == 4
    ), "Inputs must be 4D tensors of shape (B, C, H, W)"
    assert (
        original.shape == edited.shape
    ), "Original and edited images must have the same shape"

    _validate_range_minus1_to_1(original, "Original images")
    _validate_range_minus1_to_1(edited, "Edited images")

    # Mean absolute difference across pixels
    mae = torch.mean(torch.abs(original - edited), dim=[1, 2, 3])

    return mae


def lpips_distance(original, edited, net="alex", device="cpu"):
    """
    Calculate LPIPS (Learned Perceptual Image Patch Similarity) distance between original and edited images.

    LPIPS measures perceptual similarity using deep features from pre-trained networks. Lower values indicate more similar images.

    - To evaluate CIG Composition: Compare factual images to counterfactuals after applying several rounds of null-intervention.
    - To evaluate CIG Reversibility: Compare factual images to reversed counterfactuals.

    Args:
        original: Tensor of shape (B, C, H, W) for original images. Values must be in range [-1, 1]
        edited: Tensor of shape (B, C, H, W) for edited images. Values must be in range [-1, 1]
        net: Network to use for feature extraction. Options: 'alex', 'vgg', 'squeeze'. Note 'alex' (AlexNet) is fastest, 'vgg' is most accurate
        device: Device to run computation on ('cpu', 'cuda')

    Returns:
        Tensor of shape (B,) containing LPIPS distance per paired sample

    Note:
        - The lpips package interface requires inputs in [-1, 1] regardless of backbone
          (alex, vgg, squeeze); it applies ImageNet normalisation internally.
        - The first call will download the pretrained network weights.
        - Model is cached per (net, device) combination to prevent memory leaks.
    """
    # Get or create cached LPIPS model (prevents memory leak from repeated instantiation)
    cache_key = (net, str(device))
    if cache_key not in _LPIPS_CACHE:
        _LPIPS_CACHE[cache_key] = lpips.LPIPS(net=net, verbose=False).to(device)
        _LPIPS_CACHE[cache_key].eval()

    loss_fn = _LPIPS_CACHE[cache_key]

    # Ensure inputs are on the correct device
    original = original.to(device)
    edited = edited.to(device)

    # Validate image input ranges (expected [-1, 1])
    _validate_range_minus1_to_1(original, "Original images")
    _validate_range_minus1_to_1(edited, "Edited images")

    # Compute LPIPS distance
    with torch.no_grad():
        lpips_dists = loss_fn(original, edited)
        lpips_dists = lpips_dists.view(-1)  # Ensure shape is (B,) even for batch size 1

    return lpips_dists


def l1_distance(original, edited):
    """
    Calculates the L1 distance (Manhattan distance) between original and edited images.

    The L1 norm is the sum of absolute differences across all pixels.
    - To evaluate CIG Composition: Compare factual images to counterfactuals after applying several rounds of null-intervention.
    - To evaluate CIG Reversibility: Compare factual images to reversed counterfactuals.

    Args:
        original: Tensor of shape (B, C, H, W) in range [-1, 1]
        edited: Tensor of shape (B, C, H, W) in range [-1, 1]

    Returns:
        Tensor of shape (B,) containing L1 distance per sample (sum of absolute differences)
    """
    assert (
        original.dim() == 4 and edited.dim() == 4
    ), "Inputs must be 4D tensors of shape (B, C, H, W)"
    assert (
        original.shape == edited.shape
    ), "Original and edited images must have the same shape"

    _validate_range_minus1_to_1(original, "Original images")
    _validate_range_minus1_to_1(edited, "Edited images")

    l1_dist = torch.abs(original - edited).sum(dim=[1, 2, 3])
    return l1_dist


def mse_per_image(x, y):
    """Mean Squared Error per image (pixel-level similarity).

    Lower = more similar. Used for composition and reversibility.

    Args:
        x: Tensor of shape (B, C, H, W) in [-1, 1] range
        y: Tensor of shape (B, C, H, W) in [-1, 1] range

    Returns:
        Tensor of shape (B,) with MSE per image
    """
    _validate_range_minus1_to_1(x, "Input x")
    _validate_range_minus1_to_1(y, "Input y")

    # MSE is computed in [0, 1] range for standard interpretation
    x01 = denorm01(x)
    y01 = denorm01(y)

    return torch.mean((x01 - y01) ** 2, dim=(1, 2, 3))


def psnr_per_image(x, y, eps=1e-8):
    """Peak Signal-to-Noise Ratio per image.

    Higher = better quality. Common metric for image reconstruction.
    PSNR = 10 * log10(MAX^2 / MSE) where MAX=1.0 for [0,1] range.

    Args:
        x: Tensor of shape (B, C, H, W) in [-1, 1] range
        y: Tensor of shape (B, C, H, W) in [-1, 1] range
        eps: Small constant to prevent division by zero

    Returns:
        Tensor of shape (B,) with PSNR per image
    """
    mse = mse_per_image(x, y)
    return 10.0 * torch.log10(1.0 / (mse + eps))


def ssim_per_image(x, y, window_size=11, sigma=1.5):
    """Structural Similarity Index per image.

    Higher = better structural similarity. Considers luminance, contrast, structure.
    More aligned with human perception than MSE/PSNR.

    Range: [-1, 1], but typically [0, 1] for similar images.

    Implementation follows Wang et al. 2004 "Image Quality Assessment: From Error
    Visibility to Structural Similarity". C1 and C2 are stability constants that
    prevent division by zero.

    Args:
        x: Tensor of shape (B, C, H, W) in [-1, 1] range
        y: Tensor of shape (B, C, H, W) in [-1, 1] range
        window_size: Size of Gaussian window for local statistics
        sigma: Standard deviation of Gaussian window

    Returns:
        Tensor of shape (B,) with SSIM per image
    """
    _validate_range_minus1_to_1(x, "Input x")
    _validate_range_minus1_to_1(y, "Input y")

    # SSIM is computed in [0, 1] range
    x01 = denorm01(x)
    y01 = denorm01(y)

    device = x01.device
    C1, C2 = 0.01**2, 0.03**2  # Stability constants (prevent division by zero)

    # Gaussian window for local statistics
    coords = torch.arange(window_size, device=device).float() - window_size // 2
    g = torch.exp(-(coords**2) / (2 * sigma**2))
    g = g / g.sum()
    window_2d = g.view(1, 1, -1, 1) * g.view(1, 1, 1, -1)

    def filt(z):
        B, C, H, W = z.shape
        w = window_2d.expand(C, 1, window_size, window_size)
        return F.conv2d(z, w, padding=window_size // 2, groups=C)

    mu_x, mu_y = filt(x01), filt(y01)
    sigma_x2 = filt(x01 * x01) - mu_x * mu_x
    sigma_y2 = filt(y01 * y01) - mu_y * mu_y
    sigma_xy = filt(x01 * y01) - mu_x * mu_y

    # Standard SSIM formula (Wang et al. 2004)
    ssim_map = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
        (mu_x**2 + mu_y**2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )
    return torch.mean(ssim_map, dim=(1, 2, 3))
