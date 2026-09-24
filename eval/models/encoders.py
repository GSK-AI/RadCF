"""Encoder models for CIG evaluation.

This module contains two separate encoder classes:
- ImageNetVAEEncoder: For CLD (Counterfactual Latent Divergence) metric
- FrozenFeatureEncoder: For feature distance metrics in composition/realism/reversibility
"""

import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models


class ImageNetVAEEncoder(nn.Module):
    """VAE encoder using ImageNet pretrained ResNet50.

    Returns (mu, logvar) for VAE latent distribution.

    USED BY:
        - CLDScorer (metrics/cld.py) for computing KL divergence in latent space

    OUTPUT:
        mu, logvar: VAE distribution parameters for measuring latent divergence
    """
    def __init__(self, latent_dim=128):
        super().__init__()
        # Load pretrained ResNet50 from ImageNet
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # Remove final FC layer, keep feature extractor
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.features.eval()

        # VAE heads - ResNet50 outputs 2048-dim features
        # NOTE: fc_mu and fc_logvar are randomly initialised and never trained.
        # This makes the logvar-derived sigma uncalibrated, which degrades CLD reliability.
        # TODO: replace with fixed sigma=1 (drop fc_logvar) or a trained VAE encoder.
        self.fc_mu = nn.Linear(2048, latent_dim)
        self.fc_logvar = nn.Linear(2048, latent_dim)
        self.latent_dim = latent_dim

        # ImageNet normalisation: pipeline images are [-1, 1]; backbone expects (x-mean)/std.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("_imagenet_mean", mean)
        self.register_buffer("_imagenet_std", std)

    def forward(self, x):
        assert x.shape[2] >= 224 and x.shape[3] >= 224, "Input images must be at least 224x224"

        # Resize to 224x224 if needed (ImageNet standard size)
        if x.shape[2] > 224 or x.shape[3] > 224:
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        # Convert [-1, 1] → ImageNet-normalised range
        x = (x * 0.5 + 0.5 - self._imagenet_mean) / self._imagenet_std

        # Extract features from pretrained ResNet50 backbone
        with torch.no_grad():
            features = self.features(x)
            features = features.view(features.size(0), -1)  # (batch, 2048)

        # VAE latent distribution parameters
        mu = self.fc_mu(features)  # (batch, latent_dim)
        logvar = self.fc_logvar(features)  # (batch, latent_dim)

        return mu, logvar


class FrozenFeatureEncoder(nn.Module):
    """Frozen ResNet18 for feature-space distance metrics.

    Returns deep feature embeddings for computing semantic similarity.

    USED BY:
        - evaluate_composition (metrics/composition.py) for feature distance
        - evaluate_realism (metrics/realism.py) for feature distance
        - evaluate_reversibility (metrics/reversibility.py) for feature distance

    OUTPUT:
        features: 512-dim embedding for L2 distance computation

    WHY FROZEN:
        Pretrained ImageNet features generalize well to X-rays.
        Frozen weights ensure consistent feature space across evaluations.
    """

    def __init__(self):
        super().__init__()
        m = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        m.fc = nn.Identity()
        self.model = m
        for p in self.parameters():
            p.requires_grad_(False)

        # ImageNet normalisation: pipeline images are [-1, 1]; backbone expects (x-mean)/std.
        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("_imagenet_mean", mean)
        self.register_buffer("_imagenet_std", std)

    def forward(self, x):
        x = (x * 0.5 + 0.5 - self._imagenet_mean) / self._imagenet_std
        return self.model(x)
