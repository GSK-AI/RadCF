"""Utility functions for CIG evaluation.

Contains helper functions for transforms, statistics, and reproducibility.
"""

import numpy as np
import torch


def _validate_range_minus1_to_1(tensor, name="Image"):
    """Validate that tensor is in range [-1, 1].

    Args:
        tensor: Tensor to validate
        name: Name of the tensor for error messages

    Raises:
        AssertionError: If tensor is outside [-1, 1] range
    """
    assert (
        tensor.min() >= -1 and tensor.max() <= 1
    ), f"{name} must be in range [-1, 1], got min={tensor.min():.3f}, max={tensor.max():.3f}"


def _validate_range_0_to_1(tensor, name="Image"):
    """Validate that tensor is in range [0, 1].

    Args:
        tensor: Tensor to validate
        name: Name of the tensor for error messages

    Raises:
        AssertionError: If tensor is outside [0, 1] range
    """
    assert (
        tensor.min() >= 0 and tensor.max() <= 1
    ), f"{name} must be in range [0, 1], got min={tensor.min():.3f}, max={tensor.max():.3f}"


def seed_all(seed=0):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def logit(p, eps=1e-4):
    """Compute logit (inverse sigmoid) of probabilities.

    Used for CF strength metric: logit difference is more discriminative
    than probability difference for extreme values near 0 or 1.

    Note: Uses torch.logit (available since PyTorch 1.7)

    Args:
        p: Probability tensor
        eps: Small constant to prevent log(0) or log(1)

    Returns:
        Logit of probabilities
    """
    return torch.logit(torch.clamp(p, eps, 1 - eps))


def js_divergence(p, q, eps=1e-8):
    """Jensen-Shannon divergence between probability distributions.

    Used for minimality: measures how much gender/disease distributions change
    between original and CF/null/rev. Lower = better preservation.

    JSD is symmetric, bounded [0, log(2)], and more stable than KL divergence.

    Args:
        p: First probability distribution
        q: Second probability distribution
        eps: Small constant for numerical stability

    Returns:
        Jensen-Shannon divergence
    """
    p = torch.clamp(p, eps, 1.0)
    q = torch.clamp(q, eps, 1.0)
    m = 0.5 * (p + q)
    return 0.5 * (
        torch.sum(p * (torch.log(p) - torch.log(m)), dim=-1)
        + torch.sum(q * (torch.log(q) - torch.log(m)), dim=-1)
    )


def denorm01(x):
    """Map [-1,1] -> [0,1] for pixel metrics.

    Dataset outputs images in [-1,1] range (standard normalization).
    FID, MSE, PSNR, SSIM expect [0,1] range.

    Args:
        x: Tensor in [-1, 1] range

    Returns:
        Tensor in [0, 1] range
    """
    return torch.clamp(x * 0.5 + 0.5, 0.0, 1.0)


def summarize_np(x):
    """Compute summary statistics (mean, median, std, 95th percentile).

    Args:
        x: NumPy array

    Returns:
        Dict with 'mean', 'median', 'std', 'p95' keys
    """
    if x.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "std": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "std": float(np.std(x)),
        "p95": float(np.percentile(x, 95)),
    }


def cat_list(lst):
    """Concatenate list of tensors into numpy array.

    Args:
        lst: List of PyTorch tensors

    Returns:
        NumPy array of concatenated tensors
    """
    return torch.cat(lst).numpy() if lst else np.array([])
