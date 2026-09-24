from __future__ import annotations

import os
from random import sample

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from typing import Dict, List

# OPTION A: 1-to-1 Mapping
# Maps the raw text in "Finding Labels" to a specific output column.
MAP_TO_GROUP = {
    "Pneumonia": "Pneumonia",
    "Infiltration": "Infiltration",
    "Consolidation": "Consolidation",
    "Atelectasis": "Atelectasis",
    "Edema": "Edema",
    "Fibrosis": "Fibrosis",
    "Mass": "Mass",
    "Nodule": "Nodule",
    "Effusion": "Effusion",
    "Pneumothorax": "Pneumothorax",
    "Pleural Thickening": "Pleural Thickening",
    "Emphysema": "Emphysema",
    "Hernia": "Hernia",
    "Cardiomegaly": "Cardiomegaly",
    "No Finding": "No Finding",
}

# The list of keys you actually want to output in 'metas'
GROUP_COLS: List[str] = [
    "Pneumonia",
    "Infiltration",
    "Consolidation",
    "Atelectasis",
    "Edema",
    "Fibrosis",
    "Mass",
    "Nodule",
    "Effusion",
    "Pneumothorax",
    "Pleural Thickening",
    "Emphysema",
    "Hernia",
    "Cardiomegaly",
]

def process_disease_labels(
    df: pd.DataFrame, label_col: str, mapping: Dict[str, str], target_groups: List[str]
) -> pd.DataFrame:
    """
    Dynamically adds binary columns (0 or 1) for each target group based on
    substring matching in the label_col.
    """
    # 1. Invert map: Group -> [List of atomic substrings]
    # e.g. "Infection" -> ["Pneumonia", "Infiltration", "Consolidation"]
    group_to_atomics = {}
    for atomic, group in mapping.items():
        if group in target_groups:
            group_to_atomics.setdefault(group, []).append(atomic)

    # 2. Define row processor
    def get_groups_for_row(s: str):
        text = str(s)  # e.g. "Infiltration|Mass"
        res = {}
        for g in target_groups:
            # Check if *any* atomic label for this group exists in the text
            atomics = group_to_atomics.get(g, [])
            present = any(a in text for a in atomics)
            res[g] = 1 if present else 0
        return pd.Series(res)

    # 3. Apply
    print(f"Mapping {label_col} to groups: {target_groups}...")
    # This creates new columns in df corresponding to GROUP_COLS
    group_df = df[label_col].apply(get_groups_for_row)
    return pd.concat([df, group_df], axis=1)


# -----------------------------------------------------------------------------
# 3. DATASET CLASS
# -----------------------------------------------------------------------------

class SimpleChexray(Dataset):
    def __init__(
        self,
        data_dir: str = "./data/xray8",
        split: str = "train",
        transform = None,
        img_size: int = 256,
        ratio: float = 1.0,
        test_size: float | None = None,
        group_cols: List[str] = GROUP_COLS,
        map_to_group: Dict[str, str] = MAP_TO_GROUP,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.group_cols = group_cols  # Store this to loop over later

        # Transforms
        self.im_transform = transform

        # 1. Load CSV
        csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
        df = pd.read_csv(csv_path)

        # 2. Map filenames to absolute paths
        all_images = {}
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(".png"):
                    all_images[f] = os.path.join(root, f)

        df["full_path"] = df["Image Index"].map(all_images)
        df = df.dropna(subset=["full_path"]).reset_index(drop=True)

        # 3. Filter Age (remove >100; keep age up to 100)
        df = df[df["Patient Age"] <= 100].copy()

        # 4. Generate Disease Labels Dynamically
        # This adds columns like "Pneumonia", "Cardiomegaly" (0 or 1) to df
        self.df = process_disease_labels(df, "Finding Labels", map_to_group, group_cols)

        # 5. Split
        if split == "train":
            dataset_size = int(ratio * 100000)
            # self.df = self.df.iloc[:100000].reset_index(drop=True)
            self.df = self.df.iloc[:dataset_size].reset_index(drop=True)
        else:
            if test_size:
                self.df = self.df.iloc[100000 : int(100000 + test_size)].reset_index(
                    drop=True
                )
            else:
                self.df = self.df.iloc[100000:].reset_index(drop=True)

    def _map_gender(self, g_str):
        if str(g_str).upper() == "M":
            return 0
        if str(g_str).upper() == "F":
            return 1
        return 0

    def _map_view(self, v_str):
        v = str(v_str).upper()
        if v == "PA":
            return 0
        if v == "AP":
            return 1
        return 0

    def _map_age(self, a):
        return a / 100

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = self.get_sample_id(idx)

        # --- A. Image ---
        img = Image.open(row["full_path"]).convert("RGB")
        img = self.im_transform(img)

        # --- B. Metadata ---
        # 1. Standard attributes
        metas = {
            "Age": torch.tensor(
                self._map_age(float(row["Patient Age"])), dtype=torch.float32
            ),
            "Sex": torch.tensor(
                self._map_gender(row["Patient Gender"]), dtype=torch.long
            ),
            "View": torch.tensor(
                self._map_view(row["View Position"]), dtype=torch.long
            ),
        }

        # 2. Dynamic Disease Attributes
        # We loop over self.group_cols so we don't have to hardcode names.
        # Logic: 0: no, 1: yes
        for group_name in self.group_cols:
            val = float(row[group_name])
            metas[group_name] = torch.tensor(val, dtype=torch.float32)

        return img, metas, sample_id

    def get_sample_id(self, idx):
        row = self.df.iloc[idx]
        return str(row["Image Index"]).replace(".png", "").replace(".jpg", "")

    def get_labels(self, target_key):
        """Return labels for all samples without loading images."""
        if target_key == "Sex":
            return self.df["Patient Gender"].apply(self._map_gender).tolist()
        if target_key == "View":
            return self.df["View Position"].apply(self._map_view).tolist()
        if target_key in self.group_cols:
            return self.df[target_key].astype(int).tolist()
        raise ValueError(f"Unknown target_key for SimpleChexray: {target_key!r}")

    def get_metadata(self, idx):
        row = self.df.iloc[idx]
        metas = {
            "Age": torch.tensor(
                self._map_age(float(row["Patient Age"])), dtype=torch.float32
            ),
            "Sex": torch.tensor(
                self._map_gender(row["Patient Gender"]), dtype=torch.long
            ),
            "View": torch.tensor(
                self._map_view(row["View Position"]), dtype=torch.long
            ),
        }
        for group_name in self.group_cols:
            metas[group_name] = torch.tensor(float(row[group_name]), dtype=torch.float32)
        return metas

        # 2. Dynamic Disease Attributes
        # We loop over self.group_cols so we don't have to hardcode names.
        # Logic: 0: no, 1: yes
        for group_name in self.group_cols:
            val = float(row[group_name])
            metas[group_name] = torch.tensor(val, dtype=torch.float32)
        return metas

class BinarySimpleChexray(Dataset):
    def __init__(
        self,
        data_dir: str = "./data/xray8",
        split: str = "train",
        transform = None,
        img_size: int = 256,
        ratio: float = 1.0,
        test_size: float | None = None,
        group_cols: List[str] = GROUP_COLS,
        map_to_group: Dict[str, str] = MAP_TO_GROUP,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_size = img_size
        self.group_cols = group_cols  # Store this to loop over later

        # Transforms
        self.im_transform = transform

        # 1. Load CSV
        csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
        df = pd.read_csv(csv_path)

        # 2. Map filenames to absolute paths
        all_images = {}
        for root, _, files in os.walk(data_dir):
            for f in files:
                if f.lower().endswith(".png"):
                    all_images[f] = os.path.join(root, f)

        df["full_path"] = df["Image Index"].map(all_images)
        df = df.dropna(subset=["full_path"]).reset_index(drop=True)

        # 3. Filter Age (remove >100; keep age up to 100)
        df = df[df["Patient Age"] <= 100].copy()

        # 4. Generate Disease Labels Dynamically
        # no finding = 0; finding = 1
        df["Disease"] = self._binarize_disease_labels(df["Finding Labels"])
        self.df = df

        # 5. Split
        if split == "train":
            dataset_size = int(ratio * 100000)
            # self.df = self.df.iloc[:100000].reset_index(drop=True)
            self.df = self.df.iloc[:dataset_size].reset_index(drop=True)
        else:
            if test_size:
                self.df = self.df.iloc[100000 : int(100000 + test_size)].reset_index(
                    drop=True
                )
            else:
                self.df = self.df.iloc[100000:].reset_index(drop=True)
    @staticmethod
    def _binarize_disease_labels(finding_labels_series: pd.Series) -> pd.Series:
        """
        Convert Finding Labels to binary: No Finding = 0, any finding = 1.

        Args:
            finding_labels_series: Series containing finding labels (e.g., "No Finding", "Pneumonia|Effusion")

        Returns:
            Series with binary disease labels (0 or 1)
        """
        return finding_labels_series.apply(lambda x: int(x != "No Finding"))

    @staticmethod
    def _map_gender(g_str):
        """Map gender string to integer: M=0, F=1, unknown=2"""
        if str(g_str).upper() == "M":
            return 0
        if str(g_str).upper() == "F":
            return 1
        return 2

    @staticmethod
    def _map_view(v_str):
        """Map view position to integer: PA=0, AP=1, unknown=2"""
        v = str(v_str).upper()
        if v == "PA":
            return 0
        if v == "AP":
            return 1
        return 2

    @staticmethod
    def _map_age(a):
        """Normalize age to [0, 1] range by dividing by 100"""
        return a / 100

    @staticmethod
    def _map_disease(d_int):
        """Identity mapping for disease (already binary 0/1)"""
        return d_int

    def __len__(self):
        return len(self.df)

    def get_labels(self, target_key):
        """Return integer labels for all samples from df without loading images.

        Args:
            target_key: Metadata key ("Sex", "View", or "Disease")

        Returns:
            list[int]: Integer label for each sample in the dataset
        """
        if target_key == "Sex":
            return self.df["Patient Gender"].apply(self._map_gender).tolist()
        if target_key == "View":
            return self.df["View Position"].apply(self._map_view).tolist()
        if target_key == "Disease":
            return self.df["Disease"].apply(self._map_disease).tolist()
        raise ValueError(f"Unknown target_key for BinarySimpleChexray: {target_key!r}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = self.get_sample_id(idx)

        # --- A. Image ---
        img = Image.open(row["full_path"]).convert("RGB")
        img = self.im_transform(img)

        # --- B. Metadata ---
        # 1. Standard attributes
        metas = {
            "Age": torch.tensor(
                self._map_age(float(row["Patient Age"])), dtype=torch.float32
            ),
            "Sex": torch.tensor(
                self._map_gender(row["Patient Gender"]), dtype=torch.long
            ),
            "View": torch.tensor(
                self._map_view(row["View Position"]), dtype=torch.long
            ),
            "Disease": torch.tensor(
                self._map_disease(row["Disease"]), dtype=torch.long
            ),
        }

        return img, metas, sample_id

    def get_sample_id(self, idx):
        row = self.df.iloc[idx]
        return str(row["Image Index"]).replace(".png", "").replace(".jpg", "")

    def get_metadata(self, idx):
        row = self.df.iloc[idx]
        return {
            "Age": torch.tensor(
                self._map_age(float(row["Patient Age"])), dtype=torch.float32
            ),
            "Sex": torch.tensor(
                self._map_gender(row["Patient Gender"]), dtype=torch.long
            ),
            "View": torch.tensor(
                self._map_view(row["View Position"]), dtype=torch.long
            ),
            "Disease": torch.tensor(
                self._map_disease(row["Disease"]), dtype=torch.long
            ),
        }
