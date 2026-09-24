import gc
import json
import logging
import math
import os
import warnings
from pathlib import Path
from dictdot import dictdot

import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
from PIL import Image
from torchvision.transforms import Normalize
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Local/Project Imports
from radcf.models import mocov3_vit
from radcf.models.autoencoder import vae_models
from radcf.models.sit import SiT, SiT_models


# ---------------------------------------------------------------------------
# Vision-encoder preprocessing
# ---------------------------------------------------------------------------

CLIP_DEFAULT_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_DEFAULT_STD = (0.26862954, 0.26130258, 0.27577711)


def preprocess_raw_image(x, enc_type):
    """Prepare images (in [-1, 1]) for a frozen vision encoder.
    
    Modified from: https://github.com/End2End-Diffusion/REPA-E/blob/main/train_repae.py
    See ./LICENSE_originals/LICENSE-REPA-E for the original license.
    """
    resolution = x.shape[-1]
    if 'clip' in enc_type:
        x = (x + 1) / 2.
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
        x = Normalize(CLIP_DEFAULT_MEAN, CLIP_DEFAULT_STD)(x)
    elif 'mocov3' in enc_type or 'mae' in enc_type:
        x = (x + 1) / 2.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'dinov2' in enc_type:
        x = (x + 1) / 2.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    elif 'dinov1' in enc_type:
        x = (x + 1) / 2.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
    elif 'jepa' in enc_type:
        x = (x + 1) / 2.
        x = Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)(x)
        x = torch.nn.functional.interpolate(x, 224 * (resolution // 256), mode='bicubic')
    return x


# ---------------------------------------------------------------------------
# EMA
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """Step the EMA model towards the current model.

    Uses positional matching (zip) instead of name matching so this works
    correctly with PEFT-wrapped models whose named_parameters() may return
    different keys depending on adapter state.
    """
    for ema_p, model_p in zip(ema_model.parameters(), model.parameters()):
        ema_p.mul_(decay).add_(model_p.data, alpha=1 - decay)

    for ema_b, model_b in zip(ema_model.buffers(), model.buffers()):
        if ema_b.dtype in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
            ema_b.mul_(decay).add_(model_b.data, alpha=1 - decay)
        else:
            ema_b.copy_(model_b)

def to_gray(img):
    # img: (C,H,W) or (H,W) → (H,W) luminance
    if img.dim() == 2:
        return img
    C = img.size(0)
    if C == 1:
        return img[0]
    # RGB luminance
    w = torch.tensor([0.299, 0.587, 0.114], dtype=img.dtype, device=img.device).view(3,1,1)
    return (img[:3] * w).sum(dim=0)

def get_rgb_diff(c, f):
    # signed grayscale difference ∈ [-1,1]
    df = to_gray(c) - to_gray(f)
    maxabs = df.abs().max()
    if maxabs > 0:
        d = (df / (maxabs + 1e-8)).clamp(-1, 1)
    else:
        d = df  # all zeros

    pos = d.clamp(min=0.0)  # positive → red
    neg = (-d).clamp(min=0.0)  # negative → blue
    R = 1.0 - neg  # neg increases → less red
    G = 1.0 - (pos + neg).clamp(max=1.0)  # fade green by magnitude
    B = 1.0 - pos  # pos increases → less blue
    diff_rgb = torch.stack([R, G, B], dim=0).clamp(0, 1)  # (3,H,W)
    return diff_rgb


def format_meta_title(meta_dict, vocabs_idx2str, do: str):
    """
    Formats titles for both Categorical (mapped) and Continuous (rounded) data.

    meta_dict: e.g. orig_metas[i]
    vocabs_idx2str: dict[col_name][idx] -> str
    do: either a single key ("Sex") or composite ("Age+Sex")
    """
    # 1. Unified loop for single or composite keys
    keys = do.split("+") if "+" in do else [do]
    pieces = []

    for key in keys:
        val = meta_dict.get(key, None)

        if val is None:
            pieces.append(f"{key}: NA")
            continue

        # 2. Unpack Tensor to Python scalar
        if hasattr(val, 'item'):
            val = val.item()

        # 3. Formatter Logic
        if key in vocabs_idx2str:
            # Case A: It has a vocabulary -> Categorical
            # We cast to int safely to look up the index
            idx = int(val)
            label = vocabs_idx2str[key].get(idx, str(idx))

        elif isinstance(val, float):
            # Case B: No vocab + Float -> Continuous
            # Format to 2 decimal places
            label = f"{val:.2f}"

        else:
            # Case C: Fallback (Integers without vocab, Strings, etc.)
            label = str(val)

        pieces.append(f"{key}: {label}")

    # Join with separator (e.g. "Age: 0.45 | Sex: Female")
    return " | ".join(pieces)


def imshow_unormalize(x):
    # x: [B,3,H,W], assumed [-1,1] or [0,1]
    if x.min() < 0:
        x = (x + 1) / 2
    return x.clamp(0, 1)


def save_batch_heatmaps(orig_imgs, null_intervention_imgs, cf_imgs,
                        orig_metas, cf_metas,
                        save_dir, steps, vocabs,
                        do='stage'):
    """
    orig_imgs, cf_imgs: [B,3,H,W] (tensors in [-1,1])
    orig_metas, cf_metas: list[dict]
    """

    #check the range for visualization
    orig_imgs = imshow_unormalize(orig_imgs)
    null_intervention_imgs = imshow_unormalize(null_intervention_imgs)
    cf_imgs = imshow_unormalize(cf_imgs)

    B = orig_imgs.shape[0]
    os.makedirs(save_dir, exist_ok=True)

    vocabs_idx2str = {
        key: {idx: token for token, idx in d.items()}
        for key, d in vocabs.items()
    }

    for i in range(B):
        diff_norm = get_rgb_diff(cf_imgs[i], orig_imgs[i])

        fig, axes = plt.subplots(1, 4, figsize=(12, 4))

        axes[0].imshow(orig_imgs[i].permute(1, 2, 0).cpu().float().numpy())
        title_str = format_meta_title(orig_metas[i], vocabs_idx2str, do)
        axes[0].set_title(f"Original\n{title_str}")
        axes[0].axis('off')

        axes[1].imshow(null_intervention_imgs[i].permute(1, 2, 0).cpu().float().numpy())
        title_str = format_meta_title(orig_metas[i], vocabs_idx2str, do)
        axes[1].set_title(f"Null Intervention\n{title_str}")
        axes[1].axis('off')

        # Counterfactual
        axes[2].imshow(cf_imgs[i].permute(1, 2, 0).cpu().float().numpy())
        title_str = format_meta_title(cf_metas[i], vocabs_idx2str, do)
        axes[2].set_title(f"CF: {title_str}")
        axes[2].axis('off')

        # Heatmap
        im = axes[3].imshow(diff_norm.permute(1, 2, 0).cpu().float().numpy())
        axes[3].set_title("Diff heatmap")
        axes[3].axis('off')

        plt.colorbar(im, ax=axes[3], fraction=0.046)
        plt.tight_layout()
        out_path = os.path.join(save_dir, f"sample_{steps}_heatmap.png")
        plt.savefig(out_path, dpi=150)
        plt.close()


def fix_mocov3_state_dict(state_dict):
    """
    Copied from: https://github.com/End2End-Diffusion/REPA-E/blob/main/utils.py
    See ./LICENSE_originals/LICENSE-REPA-E for the original license.
    """
    for k in list(state_dict.keys()):
        # retain only base_encoder up to before the embedding layer
        if k.startswith('module.base_encoder'):
            # fix naming bug in checkpoint
            new_k = k[len("module.base_encoder."):]
            if "blocks.13.norm13" in new_k:
                new_k = new_k.replace("norm13", "norm1")
            if "blocks.13.mlp.fc13" in k:
                new_k = new_k.replace("fc13", "fc1")
            if "blocks.14.norm14" in k:
                new_k = new_k.replace("norm14", "norm2")
            if "blocks.14.mlp.fc14" in k:
                new_k = new_k.replace("fc14", "fc2")
            # remove prefix
            if 'head' not in new_k and new_k.split('.')[0] != 'fc':
                state_dict[new_k] = state_dict[k]
        # delete renamed or unused k
        del state_dict[k]
    if 'pos_embed' in state_dict.keys():
        state_dict['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
            state_dict['pos_embed'], [16, 16],
        )
    return state_dict

@torch.no_grad()
def load_encoders(enc_type, device, resolution=256):
    """
    Copied from: https://github.com/End2End-Diffusion/REPA-E/blob/main/utils.py
    See ./LICENSE_originals/LICENSE-REPA-E for the original license.
    """
    assert (resolution == 256) or (resolution == 512)
    
    enc_names = enc_type.split(',')
    encoders, architectures, encoder_types = [], [], []
    for enc_name in enc_names:
        encoder_type, architecture, model_config = enc_name.split('-')
        # Currently, we only support 512x512 experiments with DINOv2 encoders.
        if resolution == 512:
            if encoder_type != 'dinov2':
                raise NotImplementedError(
                    "Currently, we only support 512x512 experiments with DINOv2 encoders."
                    )

        architectures.append(architecture)
        encoder_types.append(encoder_type)
        if encoder_type == 'mocov3':
            if architecture == 'vit':
                if model_config == 's':
                    encoder = mocov3_vit.vit_small()
                elif model_config == 'b':
                    encoder = mocov3_vit.vit_base()
                elif model_config == 'l':
                    encoder = mocov3_vit.vit_large()
                ckpt = torch.load(f'./ckpts/mocov3_vit{model_config}.pth')
                state_dict = fix_mocov3_state_dict(ckpt['state_dict'])
                del encoder.head
                encoder.load_state_dict(state_dict, strict=True)
                encoder.head = torch.nn.Identity()
            elif architecture == 'resnet':
                raise NotImplementedError()
 
            encoder = encoder.to(device)
            encoder.eval()

        elif 'dinov2' in encoder_type:
            import timm
            if 'reg' in encoder_type:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14_reg')
            else:
                encoder = torch.hub.load('facebookresearch/dinov2', f'dinov2_vit{model_config}14')
            del encoder.head
            patch_resolution = 16 * (resolution // 256)
            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [patch_resolution, patch_resolution],
            )
            encoder.head = torch.nn.Identity()
            encoder = encoder.to(device)
            encoder.eval()
        
        elif 'dinov1' == encoder_type:
            import timm
            from models import dinov1
            encoder = dinov1.vit_base()
            ckpt =  torch.load(f'./ckpts/dinov1_vit{model_config}.pth') 
            if 'pos_embed' in ckpt.keys():
                ckpt['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                    ckpt['pos_embed'], [16, 16],
                )
            del encoder.head
            encoder.head = torch.nn.Identity()
            encoder.load_state_dict(ckpt, strict=True)
            encoder = encoder.to(device)
            encoder.forward_features = encoder.forward
            encoder.eval()

        elif encoder_type == 'clip':
            import clip
            from models.clip_vit import UpdatedVisionTransformer
            encoder_ = clip.load(f"ViT-{model_config}/14", device='cpu')[0].visual
            encoder = UpdatedVisionTransformer(encoder_).to(device)
             #.to(device)
            encoder.embed_dim = encoder.model.transformer.width
            encoder.forward_features = encoder.forward
            encoder.eval()
        
        elif encoder_type == 'mae':
            from models.mae_vit import vit_large_patch16
            import timm
            kwargs = dict(img_size=256)
            encoder = vit_large_patch16(**kwargs).to(device)
            with open(f"ckpts/mae_vit{model_config}.pth", "rb") as f:
                state_dict = torch.load(f)
            if 'pos_embed' in state_dict["model"].keys():
                state_dict["model"]['pos_embed'] = timm.layers.pos_embed.resample_abs_pos_embed(
                    state_dict["model"]['pos_embed'], [16, 16],
                )
            encoder.load_state_dict(state_dict["model"])

            encoder.pos_embed.data = timm.layers.pos_embed.resample_abs_pos_embed(
                encoder.pos_embed.data, [16, 16],
            )

        elif encoder_type == 'jepa':
            from models.jepa import vit_huge
            kwargs = dict(img_size=[224, 224], patch_size=14)
            encoder = vit_huge(**kwargs).to(device)
            with open(f"ckpts/ijepa_vit{model_config}.pth", "rb") as f:
                state_dict = torch.load(f, map_location=device)
            new_state_dict = dict()
            for key, value in state_dict['encoder'].items():
                new_state_dict[key[7:]] = value
            encoder.load_state_dict(new_state_dict)
            encoder.forward_features = encoder.forward

        encoders.append(encoder)
    
    return encoders, encoder_types, architectures


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    """
    Copied from: https://github.com/End2End-Diffusion/REPA-E/blob/main/utils.py
    See ./LICENSE_originals/LICENSE-REPA-E for the original license.
    """
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    Copied from: https://github.com/End2End-Diffusion/REPA-E/blob/main/utils.py
    See ./LICENSE_originals/LICENSE-REPA-E for the original license.
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def center_crop_arr(image_arr, image_size):
    """
    Copied from: https://github.com/End2End-Diffusion/REPA-E/blob/main/utils.py
    See ./LICENSE_originals/LICENSE-REPA-E for the original license.

    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    See ./LICENSE_originals/LICENSE-guided-diffusion for the original license.
    """
    pil_image = Image.fromarray(image_arr)
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size]


def count_trainable_params(m):
    """
    Copied from: https://github.com/End2End-Diffusion/REPA-E/blob/main/utils.py
    See ./LICENSE_originals/LICENSE-REPA-E for the original license.
    """
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


def normalize_latents(latents, latents_scale, latents_bias):
    """
    Copied from: https://github.com/End2End-Diffusion/REPA-E/blob/main/utils.py
    See ./LICENSE_originals/LICENSE-REPA-E for the original license.
    """
    return (latents - latents_bias) * latents_scale


def denormalize_latents(latents, latents_scale, latents_bias):
    """
    Copied from: https://github.com/End2End-Diffusion/REPA-E/blob/main/utils.py
    See ./LICENSE_originals/LICENSE-REPA-E for the original license.
    """
    return latents / latents_scale + latents_bias


def get_models(ckpt_path, device, train_steps=0,
               label_type='vector', hf_model_id=None,
               null_token=0):
    """
    Unified loader for base models:
    1. HuggingFace CytoSyn models (if hf_model_id is provided)
    2. Local SiT/REPA-E experiments (if hf_model_id is None)
    
    Modified from: https://github.com/End2End-Diffusion/REPA-E/blob/main/train_repae.py
    See ./LICENSE_originals/LICENSE-REPA-E for the original license.
    """
    device = torch.device(device)
    # NOTE: use no_grad() context, not set_grad_enabled(False), to avoid
    # globally disabling gradients for the caller.
    prev_grad = torch.is_grad_enabled()
    torch.set_grad_enabled(False)

    # =========================================================================
    # SCENARIO A: Hugging Face (CytoSyn)
    # =========================================================================
    if hf_model_id:
        from diffusers import DiffusionPipeline
        from huggingface_hub import snapshot_download

        print(f"Loading base model from HF: {hf_model_id}")

        # 1. Load Pretrained Pipeline (VAE + Stats)
        pipe = DiffusionPipeline.from_pretrained(
            hf_model_id,
            custom_pipeline=hf_model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        # Note: We don't move pipe to device yet to save memory, we extract components first

        vae = pipe.vae.to(device)
        latents_bias = pipe.latents_bias.to(device)
        latents_scale = pipe.latents_scale.to(device)
        # scheduler = pipe.scheduler (not returning this to keep signature consistent)

        # 2. Load SiT Transformer via snapshot
        print("Downloading/Loading SiT Transformer...")
        repo_root = snapshot_download(repo_id=hf_model_id, repo_type="model")
        repo_root = Path(repo_root)
        transformer_dir = repo_root / "transformer"

        transformer = SiT.from_pretrained(
            pretrained_model_name_or_path=str(transformer_dir),
            torch_dtype=torch.float16,
            label_type=label_type
        ).to(device)

        transformer.null_token = null_token

        # 3. Freeze
        for p in vae.parameters(): p.requires_grad_(False)
        for p in transformer.parameters(): p.requires_grad_(False)

        torch.set_grad_enabled(prev_grad)
        return vae, transformer, latents_scale, latents_bias

    # =========================================================================
    # SCENARIO B: Local Experiment (SiT / REPA-E)
    # =========================================================================
    print(f"Loading Local Experiment from: {ckpt_path}")

    # 1. Parse Args
    # We assume ckpt_path points to the experiment root OR a specific .pt file
    # If it points to root, we construct the path using train_steps
    exp_path = Path(ckpt_path)
    if exp_path.is_file():
        # If user passed "exps/my_exp/checkpoints/0400000.pt"
        checkpoint_file = exp_path
        exp_root = exp_path.parent.parent
    else:
        # If user passed "exps/my_exp"
        exp_root = exp_path
        step_str = str(train_steps).zfill(7)
        checkpoint_file = exp_root / "checkpoints" / f"{step_str}.pt"

    with open(exp_root / "args.json", "r") as f:
        config = dictdot(json.load(f))


    # 2. Build Model Architecture
    if config.vae == "f8d4":
        latent_size = config.resolution // 8
        in_channels = 4
    elif config.vae == "f16d32":
        latent_size = config.resolution // 16
        in_channels = 32
    else:
        raise NotImplementedError(f"Unknown VAE: {config.vae}")

    # Load encoders just to get dimensions (cpu only)
    encoders, _, _ = load_encoders(config.enc_type, "cpu", config.resolution)
    z_dims = [e.embed_dim for e in encoders] if config.enc_type != 'None' else [0]
    del encoders
    gc.collect()

    transformer = SiT_models[config.model](
        input_size=latent_size,
        in_channels=in_channels,
        num_classes=config.num_classes,
        class_dropout_prob=config.cfg_prob,
        z_dims=z_dims,
        encoder_depth=config.encoder_depth,
        bn_momentum=config.bn_momentum,
        label_type=label_type,
        fused_attn=config.fused_attn,
        qk_norm=config.qk_norm
    ).to(device)

    transformer.null_token = null_token

    # 3. Load Weights
    print(f"Loading weights from {checkpoint_file}")
    state_dict = torch.load(checkpoint_file, map_location=device, weights_only=False)


    # Load EMA weights
    try:
        result = transformer.load_state_dict(state_dict['ema'], strict=False)
        if result.missing_keys or result.unexpected_keys:
            logging.getLogger(__name__).warning(
                f"Non-strict load: missing={result.missing_keys}, "
                f"unexpected={result.unexpected_keys}"
            )
    except RuntimeError:
        logging.getLogger(__name__).warning(
            "Checkpoint may not have been trained end-to-end. "
            "Filtering projectors.0.4.* keys and retrying."
        )
        sd = state_dict["ema"]
        sd = {k: v for k, v in sd.items() if not k.startswith("projectors.0.4.")}
        transformer.load_state_dict(sd, strict=False)
    transformer.eval()

    # 4. Load VAE & Stats
    vae = vae_models[config.vae]().to(device)

    if "vae" in state_dict:
        # Case B1: VAE inside checkpoint (REPA-E)
        vae.load_state_dict(state_dict['vae'])

        # Stats from BN running stats
        latents_scale = state_dict["ema"]["bn.running_var"].rsqrt().view(1, in_channels, 1, 1).to(device)
        latents_bias = state_dict["ema"]["bn.running_mean"].view(1, in_channels, 1, 1).to(device)
    else:
        # Case B2: External VAE
        vae_state = torch.load(config.vae_ckpt, map_location=device)
        vae.load_state_dict(vae_state)

        stats_path = config.vae_ckpt.replace(".pt", "-latents-stats.pt")
        latents_stats = torch.load(stats_path, map_location=device)
        latents_scale = latents_stats["latents_scale"].to(device)
        latents_bias = latents_stats["latents_bias"].to(device)

    vae.eval()

    # Freeze Local Models
    for p in vae.parameters(): p.requires_grad_(False)
    for p in transformer.parameters(): p.requires_grad_(False)

    torch.set_grad_enabled(prev_grad)
    return vae, transformer, latents_scale, latents_bias


def build_vocabs(schema):
    """
    Build per-attribute vocabularies from a schema definition.

    The schema can define:
      - Standard attributes (categorical / continuous / categorical_with_unknown)
      - Group attributes where a single vocab applies to multiple sub-keys

    For group items, the returned mapping will include an entry for each sub-key
    (e.g., each disease name), all pointing to a copy of the same vocab.

    Parameters
    ----------
    schema : List[Dict]
        List of schema item dictionaries. Each item may contain:
        - name: str
        - type: str in {'categorical','continuous','categorical_with_unknown','group'}
        - vocab: Dict[str, int] (optional)
        - keys: List[str] (only for group)

    Returns
    -------
    Dict[str, Dict[str, int]]
        Mapping from attribute name (or group sub-key) to its vocab dict.
    """
    vocabs = {}
    for item in schema:
        if "vocab" in item:
            if item["type"] in [
                "categorical",
                "continuous",
                "categorical_with_unknown",
            ]:
                vocabs[item["name"]] = item["vocab"].copy()
            elif item["type"] in [
                "group_categorical",
                "group_categorical_with_unknown",
                "group",
            ]:
                for sub_key in item["keys"]:
                    vocabs[sub_key] = item["vocab"].copy()
    return vocabs