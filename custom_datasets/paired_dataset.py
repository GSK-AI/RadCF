import os
import warnings
from PIL import Image
from torch.utils.data import Dataset


class PairedImageWrapper(Dataset):
    """
    Generic wrapper that returns:
    (orig, cf, null, reverse_cf, metas)

    Assumes paired images are saved as:
        originals/{sample_id}.png
        counterfactuals/{sample_id}.png
        null_interventions/{sample_id}.png
        reverse_cfs/{sample_id}.png
    """

    def __init__(self, base_dataset, cig_path):
        self.base_dataset = base_dataset
        self.cig_path = cig_path
        self.samples = []

        for idx in range(len(base_dataset)):
            sample_id = base_dataset.get_sample_id(idx)
            if sample_id is None:
                continue

            fname = f"{sample_id}.png"
            paths = {
                "orig": os.path.join(cig_path, "originals", fname),
                "cf": os.path.join(cig_path, "counterfactuals", fname),
                "null": os.path.join(cig_path, "null_interventions", fname),
                "rev": os.path.join(cig_path, "reverse_cfs", fname),
            }

            if all(os.path.exists(p) for p in paths.values()):
                self.samples.append((idx, sample_id, paths))

        if len(self.samples) < len(base_dataset):
            warnings.warn(
                f"{len(base_dataset) - len(self.samples)} samples excluded because paired images were missing."
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        base_idx, sample_id, paths = self.samples[i]

        img_orig = Image.open(paths["orig"]).convert("RGB")
        img_cf = Image.open(paths["cf"]).convert("RGB")
        img_null = Image.open(paths["null"]).convert("RGB")
        img_rev = Image.open(paths["rev"]).convert("RGB")

        transform = getattr(self.base_dataset, "im_transform", None)
        if transform is None:
            transform = getattr(self.base_dataset, "transform", None)

        if transform is not None:
            img_orig = transform(img_orig)
            img_cf = transform(img_cf)
            img_null = transform(img_null)
            img_rev = transform(img_rev)

        metas = self.base_dataset.get_metadata(base_idx)
        metas["filename"] = sample_id

        return img_orig, img_cf, img_null, img_rev, metas