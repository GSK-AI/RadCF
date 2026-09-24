import os
from typing import List, Optional, Dict, Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class BraxDataset(Dataset):
    """
    PyTorch dataset for BRAX-style chest X-ray data with metadata conditioning.

    Returns
    -------
    image : torch.Tensor
        Transformed image tensor.
    meta : dict
        Metadata dictionary with keys:
        - "Sex"
        - "Age"
        - "View"
        - "Disease"
        - "Support Device"
    image_id : str
        Image identifier, here using PngPath.
    """

    DISEASE_COLS: List[str] = [
        "Enlarged Cardiomediastinum",
        "Cardiomegaly",
        "Lung Lesion",
        "Lung Opacity",
        "Edema",
        "Consolidation",
        "Pneumonia",
        "Atelectasis",
        "Pneumothorax",
        "Pleural Effusion",
        "Pleural Other",
        "Fracture",
    ]

    DEVICE_COL = "Support Devices"

    AGE_MAP = {
        "0": 0,
        "5": 5,
        "10": 10,
        "15": 15,
        "20": 20,
        "25": 25,
        "30": 30,
        "35": 35,
        "40": 40,
        "45": 45,
        "50": 50,
        "55": 55,
        "60": 60,
        "65": 65,
        "70": 70,
        "75": 75,
        "80": 80,
        "85 or more": 85,
    }

    VIEW_MAP = {
        "PA": "PA",
        "AP": "AP",
        "L": "LATERAL",
        "RL": "LATERAL",
        "RLO": "LATERAL",
        "LT-DECUB": "LATERAL",
        "AP LLD": "AP",
    }

    VIEW_ENCODING = {
        "PA": 0.0,
        "AP": 1.0,
        "LATERAL": 2.0,
    }

    SEX_ENCODING = {
        "M": 0.0,
        "F": 1.0,
    }

    def __init__(
        self,
        data_dir: str | None = None,
        train_csv: str | None = None,
        test_csv: str | None = None,
        split: str | None = None,
        transform=None,
        use_views: Optional[List[str]] = None,
    ):
        if split == 'train':
            assert train_csv is not None, "train_csv required for split='train'"
            csv_path = train_csv
            print(f'Use {train_csv} for train split')
        else:
            assert test_csv is not None, "test_csv required for split='test'"
            csv_path = test_csv
            print(f'Use {test_csv} for test split')
        self.df = pd.read_csv(csv_path).copy()
        self.data_dir = data_dir

        self.transform = transform

        self._validate_columns()
        self._clean_metadata()

        if use_views is not None:
            allowed = set(use_views)
            valid = set(self.VIEW_ENCODING.keys())
            unknown = allowed - valid
            if unknown:
                raise ValueError(
                    f"use_views contains unrecognised values: {sorted(unknown)}. "
                    f"Valid options are: {sorted(valid)}"
                )
            self.df = self.df[self.df["ViewPosition"].isin(allowed)].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError("Dataset is empty after preprocessing/filtering.")

    def _validate_columns(self) -> None:
        required_cols = [
            "PngPath",
            "PatientSex",
            "PatientAge",
            "ViewPosition",
            self.DEVICE_COL,
        ] + self.DISEASE_COLS

        missing = [c for c in required_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def _clean_metadata(self) -> None:
        # -------- Age --------
        self.df["PatientAgeRaw"] = self.df["PatientAge"].astype(str).str.strip()
        self.df["PatientAge"] = self.df["PatientAgeRaw"].map(self.AGE_MAP)

        unknown_age_mask = self.df["PatientAge"].isna()
        if unknown_age_mask.any():
            unknown_values = sorted(
                self.df.loc[unknown_age_mask, "PatientAgeRaw"].unique().tolist()
            )
            raise ValueError(f"Unknown PatientAge values found: {unknown_values}")

        self.df["PatientAge"] = self.df["PatientAge"].astype(float)

        # -------- Sex --------
        self.df["PatientSex"] = self.df["PatientSex"].astype(str).str.strip().str.upper()
        unknown_sex_mask = ~self.df["PatientSex"].isin(self.SEX_ENCODING)
        if unknown_sex_mask.any():
            unknown_values = sorted(
                self.df.loc[unknown_sex_mask, "PatientSex"].unique().tolist()
            )
            raise ValueError(f"Unknown PatientSex values found: {unknown_values}")

        # -------- View --------
        self.df["ViewPosition"] = self.df["ViewPosition"].replace("", pd.NA)
        self.df = self.df.dropna(subset=["ViewPosition"]).copy()

        self.df["ViewPositionRaw"] = self.df["ViewPosition"].astype(str).str.strip().str.upper()
        self.df["ViewPosition"] = self.df["ViewPositionRaw"].map(self.VIEW_MAP)

        unknown_view_mask = self.df["ViewPosition"].isna()
        if unknown_view_mask.any():
            unknown_values = sorted(
                self.df.loc[unknown_view_mask, "ViewPositionRaw"].unique().tolist()
            )
            raise ValueError(f"Unknown ViewPosition values found: {unknown_values}")

        # -------- Disease labels --------
        self.df[self.DISEASE_COLS] = self.df[self.DISEASE_COLS].fillna(0)
        for col in self.DISEASE_COLS:
            self.df[col] = pd.to_numeric(self.df[col], errors="coerce").fillna(0).astype(float)

        # -------- Support device --------
        # In this dataset, NaN means "no device"
        self.df[self.DEVICE_COL] = pd.to_numeric(
            self.df[self.DEVICE_COL], errors="coerce"
        ).fillna(0).astype(float)

        self.df = self.df.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def encode_metadata(self, row: pd.Series) -> Dict[str, Any]:
        age = float(row["PatientAge"]) / 100.0
        sex = self.SEX_ENCODING[row["PatientSex"]]
        view = self.VIEW_ENCODING[row["ViewPosition"]]
        device = float(row[self.DEVICE_COL])
        meta = {
            "Sex": torch.tensor(sex, dtype=torch.float32),
            "Age": torch.tensor(age, dtype=torch.float32),
            "View": torch.tensor(view, dtype=torch.float32),
            "Device": torch.tensor(device, dtype=torch.float32),
        }

        for col in self.DISEASE_COLS:
            meta[col] = torch.tensor(float(row[col]), dtype=torch.float32)

        return meta

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]

        png_path = row["PngPath"]
        img_path = png_path if os.path.isabs(png_path) else os.path.join(self.data_dir, png_path)

        img = Image.open(img_path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        meta = self.encode_metadata(row)

        sample_id = self.get_sample_id(idx)

        return img, meta, sample_id

    def get_sample_id(self, idx):
        row = self.df.iloc[idx]
        path = row["PngPath"]

        # make it unique + filesystem safe
        return os.path.splitext(path.replace("/", "__"))[0]

    def get_metadata(self, idx):
        row = self.df.iloc[idx]
        return self.encode_metadata(row)
