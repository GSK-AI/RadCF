import torch
from torchmetrics.image.kid import KernelInceptionDistance as KID

from eval.metrics.utils import denorm01, _validate_range_minus1_to_1


class KIDScorer:
    """
    Wrapper for computing Kernel Inception Distance (KID) between original and edited images.
    KID uses polynomial kernels in Inception feature space (similar to FID but with MMD).
    """

    def __init__(self, device, subset_size=100):
        """
        Initialize KID scorer.

        Args:
            device: Device to run computation on ('cpu', 'cuda')
            subset_size: Number of samples to use per subset (default: 100)
        """
        self.device = device
        # normalize=False: we pass uint8 images directly; normalize=True would
        # apply an additional (img * 255).byte() conversion, overflowing uint8.
        self.kid = KID(
            subset_size=subset_size, normalize=False, reset_real_features=False
        ).to(device)

    def update(self, original, edited):
        """
        Update KID metric with a batch of images (updates both real and fake distributions).

        Args:
            original: Tensor of shape (B, C, H, W) in range [-1, 1]
            edited: Tensor of shape (B, C, H, W) in range [-1, 1]
        """
        original = original.to(self.device)
        edited = edited.to(self.device)

        _validate_range_minus1_to_1(original, "Original images")
        _validate_range_minus1_to_1(edited, "Edited images")

        # Convert from [-1, 1] to [0, 1] then to [0, 255] uint8 as expected by KID
        original_01 = denorm01(original)
        edited_01 = denorm01(edited)
        original_uint8 = (original_01 * 255).to(torch.uint8)
        edited_uint8 = (edited_01 * 255).to(torch.uint8)

        self.kid.update(original_uint8, real=True)
        self.kid.update(edited_uint8, real=False)

    def update_real(self, images):
        """
        Update only the real distribution (for train reference accumulation).

        Args:
            images: Tensor of shape (B, C, H, W) in range [-1, 1]
        """
        images = images.to(self.device)

        _validate_range_minus1_to_1(images, "Images")

        # Convert from [-1, 1] to [0, 1] then to [0, 255] uint8 as expected by KID
        images_01 = denorm01(images)
        images_uint8 = (images_01 * 255).to(torch.uint8)
        self.kid.update(images_uint8, real=True)

    def update_fake(self, images):
        """
        Update only the fake distribution (for CF accumulation).

        Args:
            images: Tensor of shape (B, C, H, W) in range [-1, 1]
        """
        images = images.to(self.device)

        _validate_range_minus1_to_1(images, "Images")

        # Convert from [-1, 1] to [0, 1] then to [0, 255] uint8 as expected by KID
        images_01 = denorm01(images)
        images_uint8 = (images_01 * 255).to(torch.uint8)
        self.kid.update(images_uint8, real=False)

    def compute(self):
        """
        Compute and return the KID score after it has been updated with all batches.

        Returns:
            Tuple of (mean, std) of KID score
        """
        kid_mean, kid_std = self.kid.compute()
        return kid_mean.item(), kid_std.item()

    def reset(self):
        """Reset the KID metric to start fresh."""
        self.kid.reset()
