"""Judge models and training scripts for evaluation.

EXTERNAL DEPENDENCY:
    Uses judge dataset classes from EvalConfig.EVAL_DATASETS registry.
    Each dataset's judge_dataset_class is used for judge training.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, f1_score
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler, random_split
import torchxrayvision as xrv
from torchvision import models
from tqdm import tqdm

from eval.config import EvalConfig
from eval.metrics.utils import seed_all


# ============================================================
# Models
# ============================================================
def _load_txv_weights_into_densenet121(txv_model_name):
    """Extract TXV domain-pretrained weights into a torchvision DenseNet121.

    TXV DenseNet shares the same layer naming as torchvision (same source heritage),
    so keys remap 1:1 for all body layers. The only adjustment is the first conv:
    TXV trains on 1-channel greyscale so conv0 has shape (64,1,7,7); we expand to
    (64,3,7,7) by repeating ÷3, which is mathematically equivalent for greyscale-as-RGB
    inputs (all 3 channels identical). TXV is used only at weight-load time; runtime
    is pure torchvision with no TXV overhead.

    Any key missing from TXV or with an unexpected shape mismatch falls back to
    random init with a printed warning.
    """
    txv = xrv.models.DenseNet(weights=txv_model_name)
    txv_state = txv.state_dict()

    backbone = models.densenet121(weights=None)
    tv_state = backbone.state_dict()

    new_state = {}
    fallbacks = []
    for key in tv_state:
        if key.startswith("classifier"):
            new_state[key] = tv_state[key]  # replaced by caller
        elif key == "features.conv0.weight":
            if key in txv_state:
                new_state[key] = txv_state[key].repeat(1, 3, 1, 1) / 3.0
            else:
                new_state[key] = tv_state[key]
                fallbacks.append(key)
        elif key in txv_state and txv_state[key].shape == tv_state[key].shape:
            new_state[key] = txv_state[key]
        else:
            new_state[key] = tv_state[key]
            fallbacks.append(key)

    if fallbacks:
        print(f"[TXV remap] {len(fallbacks)} keys fell back to random init: {fallbacks[:5]}{'...' if len(fallbacks) > 5 else ''}")
    else:
        print(f"[TXV remap] All DenseNet body keys loaded from {txv_model_name}")

    backbone.load_state_dict(new_state)
    return backbone


def _create_backbone(arch="resnet18", num_outputs=2):
    """Create backbone with a task head.

    Single source of truth for model architecture. Supports both classification
    (num_outputs=2) and regression (num_outputs=1).
    """
    if arch == "resnet18":
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_outputs)
        # Pipeline delivers [-1, 1] / 256px; ImageNet weights expect (img-mean)/std at 224px.
        # mean/std are the ImageNet-1k channel statistics used by ResNet18_Weights.DEFAULT.
        # Validated from torchvision source: ResNet18_Weights.IMAGENET1K_V1 uses
        # ImageClassification(crop_size=224) whose default mean/std are defined in
        # torchvision/transforms/_presets.py::ImageClassification.__init__.
        backbone.register_buffer("_imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), persistent=False)
        backbone.register_buffer("_imagenet_std",  torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1), persistent=False)
        backbone.register_buffer("_input_size", torch.tensor(224), persistent=False)
    elif arch == "resnet50":
        backbone = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        backbone.fc = nn.Linear(backbone.fc.in_features, num_outputs)
        # Same ImageNet normalisation and 224px resize as resnet18 above, see explanation there.
        backbone.register_buffer("_imagenet_mean", torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1), persistent=False)
        backbone.register_buffer("_imagenet_std",  torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1), persistent=False)
        backbone.register_buffer("_input_size", torch.tensor(224), persistent=False)
    elif arch == "densenet121":
        backbone = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        backbone.classifier = nn.Linear(backbone.classifier.in_features, num_outputs)
    elif arch == "densenet121-chex":
        # torchvision DenseNet121 initialised from TXV CheXpert-pretrained weights.
        # CheXpert pretraining has no NIH14 overlap — safe for **chex8** judges.
        # TXV expects [-1024, 1024] inputs: confirmed via torchxrayvision/utils.py
        # normalize() formula `(2*(img/maxval)-1)*1024` and warn_normalization() which
        # flags inputs outside this range. No internal normalisation in model.forward().
        # Our pipeline delivers [-1, 1]; _input_scale=1024 corrects this in forward().
        # TXV was trained at 224x224 (res224 in model name); pipeline delivers 256x256.
        # _input_size=224 triggers F.interpolate in forward() to match training resolution.
        backbone = _load_txv_weights_into_densenet121("densenet121-res224-chex")
        backbone.classifier = nn.Linear(backbone.classifier.in_features, num_outputs)
        backbone.register_buffer("_input_scale", torch.tensor(1024.0), persistent=False)
        backbone.register_buffer("_input_size", torch.tensor(224), persistent=False)
    elif arch == "densenet121-mimic":
        # torchvision DenseNet121 initialised from TXV MIMIC-CXR-pretrained weights.
        # MIMIC-CXR (BIDMC) has no patient overlap with NIH14 (NIH) or CheXpert (Stanford)
        # — safe for both chex8 and chexpert judges.
        # Same TXV [-1024, 1024] input requirement and 224x224 training resolution as
        # densenet121-chex (see above).
        backbone = _load_txv_weights_into_densenet121("densenet121-res224-mimic_nb")
        backbone.classifier = nn.Linear(backbone.classifier.in_features, num_outputs)
        backbone.register_buffer("_input_scale", torch.tensor(1024.0), persistent=False)
        backbone.register_buffer("_input_size", torch.tensor(224), persistent=False)
    else:
        raise ValueError(f"Unknown architecture: {arch}. Supported: resnet18, resnet50, densenet121, densenet121-chex, densenet121-mimic")
    return backbone


class BinaryClassifier(nn.Module):
    """Binary classifier with integrated optimizer and loss.

    Used for all classification tasks: gender, view, disease.
    """

    def __init__(self, lr=1e-4, device="cpu", arch="resnet18", label_smoothing=0.0):
        super().__init__()
        self.arch = arch
        self.label_smoothing = label_smoothing
        self.backbone = _create_backbone(arch).to(device)
        self.optimizer = optim.Adam(self.backbone.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.device = device

    def forward(self, x):
        # _input_size: resize to match training resolution (TXV 224px, ImageNet resnet 224px).
        size = getattr(self.backbone, "_input_size", None)
        if size is not None and x.shape[-1] != size.item():
            x = F.interpolate(x, size=size.item(), mode="bilinear", align_corners=False)
        # TXV: scale [-1, 1] → [-1024, 1024].
        scale = getattr(self.backbone, "_input_scale", None)
        if scale is not None:
            x = x * scale
        # ImageNet resnet: convert pipeline [-1, 1] → (img - mean) / std.
        mean = getattr(self.backbone, "_imagenet_mean", None)
        if mean is not None:
            x = (x * 0.5 + 0.5 - mean) / self.backbone._imagenet_std
        return self.backbone(x)

    def predict_probs(self, x):
        """Get probability predictions (applies softmax to logits).

        Args:
            x: Input images

        Returns:
            Tensor of shape (B, 2) with probabilities for each class
        """
        return F.softmax(self.forward(x), dim=1)


    def to(self, device):
        self.backbone = self.backbone.to(device)
        self.device = device
        return self


class RegressionJudge(nn.Module):
    """Regression judge for continuous attributes (e.g., age).

    Outputs a scalar in [0, 1] via sigmoid. Trained with HuberLoss.
    """

    def __init__(self, lr=1e-4, device="cpu", arch="resnet18"):
        super().__init__()
        self.arch = arch
        self.backbone = _create_backbone(arch, num_outputs=1).to(device)
        self.optimizer = optim.Adam(self.backbone.parameters(), lr=lr)
        self.criterion = nn.HuberLoss(delta=0.1)
        self.device = device

    def forward(self, x):
        size = getattr(self.backbone, "_input_size", None)
        if size is not None and x.shape[-1] != size.item():
            x = F.interpolate(x, size=size.item(), mode="bilinear", align_corners=False)
        scale = getattr(self.backbone, "_input_scale", None)
        if scale is not None:
            x = x * scale
        mean = getattr(self.backbone, "_imagenet_mean", None)
        if mean is not None:
            x = (x * 0.5 + 0.5 - mean) / self.backbone._imagenet_std
        return torch.sigmoid(self.backbone(x)).squeeze(1)  # (B,)

    def to(self, device):
        self.backbone = self.backbone.to(device)
        self.device = device
        return self


def load_regression_judge(checkpoint_path, device="cpu"):
    """Load a trained RegressionJudge from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = RegressionJudge(device=device, arch=ckpt.get("architecture", "resnet18"))
    model.backbone.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def load_binary_classifier(checkpoint_path, device="cpu", return_threshold=True):
    """Load a trained binary classifier from checkpoint with validation.

    Args:
        checkpoint_path: Path to saved checkpoint
        device: Device to load model on
        return_threshold: If True, return (model, threshold). If False, return only model (for backward compatibility)

    Returns:
        If return_threshold=True: tuple (model, threshold)
        If return_threshold=False: model only
        - model: BinaryClassifier instance in eval mode
        - threshold: Optimal decision threshold (0.5 if not found in checkpoint)

    Raises:
        ValueError: If checkpoint architecture doesn't match expected architecture
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    arch = ckpt.get("architecture", "resnet18")

    if "num_classes" in ckpt:
        expected_classes = 2
        if ckpt["num_classes"] != expected_classes:
            raise ValueError(
                f"Class count mismatch: checkpoint has {ckpt['num_classes']} classes, "
                f"expected {expected_classes}"
            )

    # Load model using the architecture recorded in the checkpoint
    model = BinaryClassifier(device=device, arch=arch)
    model.backbone.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load optimal threshold (default to 0.5 if not found)
    threshold = ckpt.get("optimal_threshold", 0.5)

    if return_threshold:
        return model, threshold
    else:
        return model


# ============================================================
# Training Functions
# ============================================================

# Class names for binary classification tasks
CLASS_NAMES = {
    "gender": ["Male", "Female"],
    "view": ["PA", "AP"],
    "disease": ["No Finding", "Finding"],
}


def _resolve_labels(dataset, target_key):
    """Get labels for all samples in dataset, resolving nested Subset indices correctly.

    For nested Subsets (e.g. filter_by_view → random_split), the outer Subset's
    indices are positional into the inner Subset, not into the base dataset.
    This function composes indices through all levels to get correct base-dataset positions.
    """
    indices = list(dataset.indices) if isinstance(dataset, Subset) else None
    base = dataset
    while isinstance(base, Subset):
        if isinstance(base.dataset, Subset):
            # Compose: map current indices through the next Subset level
            indices = [base.dataset.indices[i] for i in indices]
        base = base.dataset
    all_labels = base.get_labels(target_key)  # type: ignore
    if indices is None:
        return all_labels
    return [all_labels[i] for i in indices]


def calculate_class_weights(dataset, target_key, task_name):
    """Calculate inverse frequency class weights from dataset.

    Args:
        dataset: Dataset to calculate weights from
        target_key: Key in metadata dict containing target labels
        task_name: Task name ("gender", "view", "disease") for class name labeling

    Returns:
        tuple[torch.Tensor, list[int]]: Normalised class weights and per-sample labels
    """
    class_names = CLASS_NAMES[task_name]

    labels = _resolve_labels(dataset, target_key)
    class_counts = torch.bincount(torch.tensor(labels))
    weights = 1.0 / class_counts.float()
    weights = weights / weights.sum()  # Normalize

    print(
        f"  {target_key} - {class_names[0]}: count={class_counts[0]}, weight={weights[0]:.4f}"
    )
    print(
        f"  {target_key} - {class_names[1]}: count={class_counts[1]}, weight={weights[1]:.4f}"
    )

    return weights, labels


def create_balanced_sampler(dataset, labels, class_weights):
    """Create WeightedRandomSampler for class-balanced minibatches.

    Args:
        dataset: Dataset to sample from
        labels: Per-sample integer labels (from calculate_class_weights)
        class_weights: Per-class weights from calculate_class_weights

    Returns:
        WeightedRandomSampler: Sampler that balances classes in each minibatch
    """
    sample_weights = [class_weights[c].item() for c in labels]

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(dataset),
        replacement=True,  # Sample with replacement to balance classes
    )


def filter_by_view(dataset, view_value):
    """Return subset with specified view (0=PA, 1=AP)."""
    # Access df directly to avoid loading images
    # Handle different dataset types
    if hasattr(dataset, "view_col"):
        # CheXpertMetaDataset: uses self.view_col ("AP/PA" or "Frontal/Lateral")
        view_col = dataset.df[dataset.view_col].apply(dataset._map_view)
    else:
        # BinarySimpleChexray: uses "View Position"
        view_col = dataset.df["View Position"].apply(dataset._map_view)
    indices = [i for i, v in enumerate(view_col) if v == view_value]
    return Subset(dataset, indices)


def _compute_threshold(y_true, y_scores, criterion="youden"):
    """Compute optimal decision threshold from pre-collected arrays.

    Args:
        y_true: np.ndarray of ground-truth binary labels
        y_scores: np.ndarray of positive-class probabilities
        criterion: 'youden' or 'f1'

    Returns:
        dict with 'threshold', 'criterion', and 'metrics'
    """
    if criterion == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        optimal_threshold = thresholds[np.argmax(tpr - fpr)]
    elif criterion == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1_scores = np.nan_to_num(2 * precision * recall / (precision + recall))
        idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[idx] if idx < len(thresholds) else 0.5
    else:
        raise ValueError(f"Unknown criterion: {criterion}. Use 'youden' or 'f1'.")

    y_pred = (y_scores >= optimal_threshold).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    return {
        "threshold": float(optimal_threshold),
        "criterion": criterion,
        "metrics": {
            "accuracy": (tp + tn) / len(y_true),
            "sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
            "f1": f1_score(y_true, y_pred),
        },
    }



def train_binary_classifier(
    model, task_name, checkpoint_path, config, dataset_name, view_filter=None
):
    """Train a binary classifier for any task.

    Args:
        model: BinaryClassifier instance with optimizer and criterion
        task_name: "gender", "view", or "disease"
        checkpoint_path: Where to save the model
        config: EvalConfig object with hyperparameters
        dataset_name: Name of evaluation dataset (e.g., "chex8_binary", "chexpert_frontal_effusion")
        view_filter: None (all data), 0 (PA only), or 1 (AP only)
    """
    # Get dataset configuration from registry
    dataset_config = config.get_dataset_config(dataset_name)

    # Get metadata key for this task from dataset config
    target_key = dataset_config["metadata_keys"][task_name]
    view_str = {None: "all", 0: "PA", 1: "AP"}[view_filter]

    print(
        f"{'='*70}\n"
        f"Training {task_name} classifier ({view_str} view) on {dataset_name}\n"
        f"{'='*70}"
    )

    # Prepare dataset via zoo config
    dataset = EvalConfig.build_judge_dataset(dataset_name)
    if view_filter is not None:
        dataset = filter_by_view(dataset, view_filter)

    # Split into train and validation
    train_size = int((1 - config.JUDGE_VAL_PROP) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED),
    )
    print(
        f"{len(train_dataset)} training samples, {len(val_dataset)} validation samples"
    )

    # Calculate class weights for balanced sampler
    print(f"Calculating class weights for {task_name} task...")
    weights, labels = calculate_class_weights(train_dataset, target_key, task_name)

    # Create balanced sampler for class-balanced minibatches
    sampler = create_balanced_sampler(train_dataset, labels, weights)

    loader = DataLoader(
        train_dataset,
        batch_size=config.JUDGE_BATCH_SIZE,
        sampler=sampler,  # Weighted sampling instead of shuffle for balanced batches
        num_workers=config.JUDGE_NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config.JUDGE_BATCH_SIZE,
        shuffle=False,
        num_workers=config.JUDGE_NUM_WORKERS,
        pin_memory=True,
    )

    # Get class names for logging
    class_names = CLASS_NAMES[task_name]

    # Validation evaluation helper
    def evaluate_on_validation():
        """Evaluate model on validation set and return accuracy, loss, per-class accuracy, and AUC."""
        model.eval()
        correct = total = 0
        running_loss = 0.0
        class_correct = [0, 0]  # Per-class correct predictions
        class_total = [0, 0]  # Per-class total samples

        # Collect predictions and targets for AUC computation
        all_probs = []
        all_targets = []

        with torch.no_grad():
            for images, metas, _ in val_loader:
                images = images.to(model.device)
                target = metas[target_key].long().to(model.device)
                outputs = model(images)
                loss = model.criterion(outputs, target)
                running_loss += loss.item()

                predictions = outputs.argmax(1)
                correct += (predictions == target).sum().item()
                total += target.size(0)

                # Track per-class accuracy
                for cls in [0, 1]:
                    mask = target == cls
                    if mask.any():
                        class_correct[cls] += (
                            (predictions[mask] == target[mask]).sum().item()
                        )
                        class_total[cls] += mask.sum().item()

                # Collect probabilities for AUC
                probs = model.predict_probs(images)[
                    :, 1
                ]  # Probability of positive class
                all_probs.extend(probs.cpu().numpy())
                all_targets.extend(target.cpu().numpy())

        val_acc = 100 * correct / total
        val_loss = running_loss / len(val_loader)
        val_class_acc = [
            100 * class_correct[i] / class_total[i] if class_total[i] > 0 else 0
            for i in [0, 1]
        ]

        # Compute AUC-ROC
        y_true = np.array(all_targets)
        y_scores = np.array(all_probs)
        val_auc = roc_auc_score(y_true, y_scores)

        return val_acc, val_loss, val_class_acc, val_auc, y_true, y_scores

    # Setup CSV logging
    log_dir = Path(checkpoint_path).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / f"{Path(checkpoint_path).stem}_training.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                f"train_acc_class0_{class_names[0]}",
                f"train_acc_class1_{class_names[1]}",
                "val_loss",
                "val_acc",
                "val_auc",
                f"val_acc_class0_{class_names[0]}",
                f"val_acc_class1_{class_names[1]}",
                "learning_rate",
                "optimal_threshold",
                "threshold_criterion",
                "threshold_accuracy",
                "threshold_sensitivity",
                "threshold_specificity",
                "threshold_f1",
            ]
        )

    print("Logging training metrics to CSV file.")

    # Training loop
    task_cfg = dataset_config["judge_configs"][task_name]
    n_epochs = task_cfg["epochs"]
    scheduler_cfg = task_cfg.get("scheduler")
    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            model.optimizer,
            mode="max",
            factor=scheduler_cfg["factor"],
            patience=scheduler_cfg["patience"],
        )
        if scheduler_cfg else None
    )
    best_val_auc = -1.0
    best_epoch = 0
    for epoch in range(n_epochs):
        model.train()
        correct = total = 0
        running_loss = 0.0
        train_class_correct = [0, 0]
        train_class_total = [0, 0]

        pbar = tqdm(
            loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=True, ncols=120
        )
        for batch_idx, (images, metas, _) in enumerate(pbar):
            images = images.to(model.device)
            target = metas[target_key].long().to(model.device)

            # Sanity-check first batch of first epoch: sampler should produce ~50/50 classes
            if epoch == 0 and batch_idx == 0:
                counts = torch.bincount(target.cpu(), minlength=2)
                ratio = counts[1].item() / counts.sum().item()
                if not (0.3 <= ratio <= 0.7):
                    print(
                        f"WARNING: Balanced sampler is not producing ~50/50 batches: "
                        f"{counts.tolist()} (class-1 ratio={ratio:.2f}). "
                        "Check _resolve_labels for correct index handling."
                    )
                print(f"  [sampler check] first batch class counts: {counts.tolist()} (ratio={ratio:.2f})")

            outputs = model(images)
            loss = model.criterion(outputs, target)

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            predictions = outputs.argmax(1)
            total += target.size(0)
            correct += (predictions == target).sum().item()
            running_loss += loss.item()

            # Track per-class training accuracy
            for cls in [0, 1]:
                mask = target == cls
                if mask.any():
                    train_class_correct[cls] += (
                        (predictions[mask] == target[mask]).sum().item()
                    )
                    train_class_total[cls] += mask.sum().item()

            pbar.set_postfix(
                {
                    "loss": f"{running_loss/(batch_idx+1):.4f}",
                    "acc": f"{100*correct/total:.1f}%",
                }
            )

        acc = 100 * correct / total
        avg_loss = running_loss / len(loader)
        train_class_acc = [
            (
                100 * train_class_correct[i] / train_class_total[i]
                if train_class_total[i] > 0
                else 0
            )
            for i in [0, 1]
        ]

        # Evaluate on validation
        val_acc, val_loss, val_class_acc, val_auc, y_true, y_scores = evaluate_on_validation()
        print(
            f"Epoch {epoch+1}/{n_epochs}: "
            f"train_loss={avg_loss:.4f}, train_acc={acc:.2f}%, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.2f}%, val_auc={val_auc:.4f}"
        )
        print(
            f"  Train per-class: {class_names[0]}={train_class_acc[0]:.2f}%, "
            f"{class_names[1]}={train_class_acc[1]:.2f}%"
        )
        print(
            f"  Val per-class:   {class_names[0]}={val_class_acc[0]:.2f}%, "
            f"{class_names[1]}={val_class_acc[1]:.2f}%"
        )

        # Find optimal threshold and compute metrics for this epoch (reuses val arrays)
        threshold_criterion = config.JUDGE_THRESHOLD_CRITERION.get(task_name, "f1")
        threshold_result = _compute_threshold(y_true, y_scores, criterion=threshold_criterion)
        print(
            f"  Threshold ({threshold_criterion}): {threshold_result['threshold']:.4f}  "
            f"sens={threshold_result['metrics']['sensitivity']:.4f}  "
            f"spec={threshold_result['metrics']['specificity']:.4f}  "
            f"f1={threshold_result['metrics']['f1']:.4f}"
        )

        # Log all metrics to CSV
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    epoch + 1,
                    avg_loss,
                    acc,
                    train_class_acc[0],
                    train_class_acc[1],
                    val_loss,
                    val_acc,
                    val_auc,
                    val_class_acc[0],
                    val_class_acc[1],
                    model.optimizer.param_groups[0]["lr"],
                    threshold_result["threshold"],
                    threshold_result["criterion"],
                    threshold_result["metrics"]["accuracy"],
                    threshold_result["metrics"]["sensitivity"],
                    threshold_result["metrics"]["specificity"],
                    threshold_result["metrics"]["f1"],
                ]
            )

        # Step LR scheduler on val_auc
        if scheduler is not None:
            scheduler.step(val_auc)

        # Save checkpoint if best val_auc so far
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch + 1
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.backbone.state_dict(),
                    "optimizer_state_dict": model.optimizer.state_dict(),
                    "train_accuracy": acc,
                    "val_accuracy": val_acc,
                    "val_auc": val_auc,
                    "val_loss": val_loss,
                    "architecture": model.arch,
                    "num_classes": 2,
                    "dataset": dataset_name,
                    "split": "train",
                    "optimal_threshold": threshold_result["threshold"],
                    "threshold_criterion": threshold_result["criterion"],
                    "threshold_metrics": threshold_result["metrics"],
                },
                checkpoint_path,
            )
            print(f"  ✓ New best val_auc={val_auc:.4f} at epoch {epoch+1} — checkpoint saved")

    print(f"\nTraining complete. Best checkpoint: epoch {best_epoch}, val_auc={best_val_auc:.4f}")


def train_regression_judge(model, task_name, checkpoint_path, config, dataset_name):
    """Train a regression judge for a continuous attribute (e.g., age).

    Args:
        model: RegressionJudge instance
        task_name: Attribute name used for config lookup (e.g., "age")
        checkpoint_path: Where to save the best checkpoint
        config: EvalConfig object
        dataset_name: Name of evaluation dataset (e.g., "chex8_binary")
    """
    dataset_config = config.get_dataset_config(dataset_name)
    target_key = dataset_config["metadata_keys"][task_name]

    print(
        f"{'='*70}\n"
        f"Training {task_name} regression judge on {dataset_name}\n"
        f"{'='*70}"
    )

    dataset = EvalConfig.build_judge_dataset(dataset_name)
    train_size = int((1 - config.JUDGE_VAL_PROP) * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(config.SEED),
    )
    print(f"{len(train_dataset)} training samples, {len(val_dataset)} validation samples")

    loader = DataLoader(
        train_dataset,
        batch_size=config.JUDGE_BATCH_SIZE,
        shuffle=True,
        num_workers=config.JUDGE_NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.JUDGE_BATCH_SIZE,
        shuffle=False,
        num_workers=config.JUDGE_NUM_WORKERS,
        pin_memory=True,
    )

    # CSV logging
    log_dir = Path(checkpoint_path).parent / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / f"{Path(checkpoint_path).stem}_training.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_mae", "val_loss", "val_mae", "learning_rate"]
        )

    task_cfg = dataset_config["judge_configs"][task_name]
    n_epochs = task_cfg["epochs"]
    scheduler_cfg = task_cfg.get("scheduler")
    scheduler = (
        optim.lr_scheduler.ReduceLROnPlateau(
            model.optimizer,
            mode="min",
            factor=scheduler_cfg["factor"],
            patience=scheduler_cfg["patience"],
        )
        if scheduler_cfg else None
    )

    best_val_mae = float("inf")
    best_epoch = 0

    for epoch in range(n_epochs):
        model.train()
        running_loss = running_mae = total = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{n_epochs}", leave=True, ncols=120)
        for images, metas, _ in pbar:
            images = images.to(model.device)
            target = metas[target_key].float().to(model.device)

            pred = model(images)
            loss = model.criterion(pred, target)

            model.optimizer.zero_grad()
            loss.backward()
            model.optimizer.step()

            running_loss += loss.item()
            running_mae += (pred.detach() - target).abs().mean().item()
            total += 1
            pbar.set_postfix({"loss": f"{running_loss/total:.4f}", "mae": f"{running_mae/total:.4f}"})

        train_loss = running_loss / len(loader)
        train_mae = running_mae / len(loader)

        # Validation
        model.eval()
        val_running_loss = val_running_mae = 0
        with torch.no_grad():
            for images, metas, _ in val_loader:
                images = images.to(model.device)
                target = metas[target_key].float().to(model.device)
                pred = model(images)
                val_running_loss += model.criterion(pred, target).item()
                val_running_mae += (pred - target).abs().mean().item()
        val_loss = val_running_loss / len(val_loader)
        val_mae = val_running_mae / len(val_loader)

        lr = model.optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch+1}/{n_epochs}: "
            f"train_loss={train_loss:.4f}, train_mae={train_mae:.4f}, "
            f"val_loss={val_loss:.4f}, val_mae={val_mae:.4f}, lr={lr:.2e}"
        )

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch + 1, train_loss, train_mae, val_loss, val_mae, lr])

        if scheduler is not None:
            scheduler.step(val_mae)

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_epoch = epoch + 1
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model_state_dict": model.backbone.state_dict(),
                    "optimizer_state_dict": model.optimizer.state_dict(),
                    "val_mae": val_mae,
                    "val_loss": val_loss,
                    "architecture": model.arch,
                    "task": task_name,
                    "dataset": dataset_name,
                },
                checkpoint_path,
            )
            print(f"  ✓ New best val_mae={val_mae:.4f} at epoch {epoch+1} — checkpoint saved")

    print(f"\nTraining complete. Best checkpoint: epoch {best_epoch}, val_mae={best_val_mae:.4f}")


def train_age_judge(dataset_name):
    """Train age regression judge."""
    config = EvalConfig()
    seed_all(config.SEED)
    dataset_config = config.get_dataset_config(dataset_name)
    model = RegressionJudge(lr=dataset_config["judge_configs"]["age"]["lr"], device=config.DEVICE, arch=config.get_judge_arch(dataset_name, "age"))
    print("Age judge training started.")
    train_regression_judge(model, "age", dataset_config["judge_age_ckpt"], config, dataset_name)


def train_gender_judge(dataset_name):
    """Train gender classifier."""
    config = EvalConfig()
    seed_all(config.SEED)
    dataset_config = config.get_dataset_config(dataset_name)
    model = BinaryClassifier(lr=dataset_config["judge_configs"]["gender"]["lr"], device=config.DEVICE, arch=config.get_judge_arch(dataset_name, "gender"))
    print("Gender judge training started.")
    train_binary_classifier(
        model, "gender", dataset_config["judge_gender_ckpt"], config, dataset_name
    )


def train_view_judge(dataset_name):
    """Train view classifier."""
    config = EvalConfig()
    seed_all(config.SEED)
    dataset_config = config.get_dataset_config(dataset_name)
    model = BinaryClassifier(lr=dataset_config["judge_configs"]["view"]["lr"], device=config.DEVICE, arch=config.get_judge_arch(dataset_name, "view"))
    print("View judge training started.")
    train_binary_classifier(
        model, "view", dataset_config["judge_view_ckpt"], config, dataset_name
    )


def train_disease_judge_by_view(dataset_name, view=None):
    """Train view-stratified disease classifiers.

    Args:
        dataset_name: Name of evaluation dataset (e.g., "chex8_binary", "chexpert_frontal_effusion")
        view: "PA", "AP", or None (trains both)
    """
    config = EvalConfig()
    seed_all(config.SEED)
    dataset_config = config.get_dataset_config(dataset_name)

    views_to_train = [
        ("PA", 0, dataset_config["judge_disease_pa_ckpt"]),
        ("AP", 1, dataset_config["judge_disease_ap_ckpt"]),
    ]

    if view:
        views_to_train = [v for v in views_to_train if v[0] == view.upper()]

    disease_cfg = dataset_config["judge_configs"]["disease"]
    for view_name, view_value, ckpt_path in views_to_train:
        model = BinaryClassifier(lr=disease_cfg["lr"], device=config.DEVICE, arch=config.get_judge_arch(dataset_name, "disease"), label_smoothing=disease_cfg.get("label_smoothing", 0.0))
        print(f"Disease {view_name} judge training started.")
        train_binary_classifier(
            model, "disease", ckpt_path, config, dataset_name, view_filter=view_value
        )
