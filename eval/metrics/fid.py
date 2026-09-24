import torch
from torchmetrics.image.fid import FrechetInceptionDistance as FID

from eval.metrics.utils import denorm01, _validate_range_minus1_to_1


class FIDScorer:
    """
    Wrapper for computing Frechet Inception Distance (FID) between original and edited images.
    """

    def __init__(self, device):
        """
        Initialize FID scorer.

        Args:
            device: Device to run computation on ('cpu', 'cuda')
        """
        self.device = device
        # normalize=False: we pass uint8 images directly; normalize=True would
        # apply an additional (img * 255).byte() conversion, overflowing uint8.
        self.fid = (
            FID(normalize=False, reset_real_features=False)
            .set_dtype(torch.float64)
            .to(device)
        )

    def update(self, original, edited):
        """
        Update FID metric with a batch of images (updates both real and fake distributions).

        Args:
            original: Tensor of shape (B, C, H, W) in range [-1, 1]
            edited: Tensor of shape (B, C, H, W) in range [-1, 1]
        """
        original = original.to(self.device)
        edited = edited.to(self.device)

        _validate_range_minus1_to_1(original, "Original images")
        _validate_range_minus1_to_1(edited, "Edited images")

        # Convert from [-1, 1] to [0, 1] then to [0, 255] uint8 as expected by FID
        original_01 = denorm01(original)
        edited_01 = denorm01(edited)
        original_uint8 = (original_01 * 255).to(torch.uint8)
        edited_uint8 = (edited_01 * 255).to(torch.uint8)

        self.fid.update(original_uint8, real=True)
        self.fid.update(edited_uint8, real=False)

    def update_real(self, images):
        """
        Update only the real distribution (for train reference accumulation).

        Args:
            images: Tensor of shape (B, C, H, W) in range [-1, 1]
        """
        images = images.to(self.device)

        _validate_range_minus1_to_1(images, "Images")

        # Convert from [-1, 1] to [0, 1] then to [0, 255] uint8 as expected by FID
        images_01 = denorm01(images)
        images_uint8 = (images_01 * 255).to(torch.uint8)
        self.fid.update(images_uint8, real=True)

    def update_fake(self, images):
        """
        Update only the fake distribution (for CF accumulation).

        Args:
            images: Tensor of shape (B, C, H, W) in range [-1, 1]
        """
        images = images.to(self.device)

        _validate_range_minus1_to_1(images, "Images")

        # Convert from [-1, 1] to [0, 1] then to [0, 255] uint8 as expected by FID
        images_01 = denorm01(images)
        images_uint8 = (images_01 * 255).to(torch.uint8)
        self.fid.update(images_uint8, real=False)

    def compute(self):
        """
        Compute and return the FID score after it has been updated with all batches.

        Returns:
            Float value of the FID score
        """
        return self.fid.compute().item()

    def reset(self):
        """Reset the FID metric to start fresh."""
        self.fid.reset()
