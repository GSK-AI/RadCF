import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


# Define the standard order for the output label vector
ATTR_ORDER = [
    "Sex",
    "Age",
    "Frontal/Lateral",
    "AP/PA",
    "Support Devices",
    "Enlarged Cardiomediastinum",
    "Cardiomegaly",
    "Lung Opacity",
    "Lung Lesion",
    "Edema",
    "Consolidation",
    "Pneumonia",
    "Atelectasis",
    "Pneumothorax",
    "Pleural Effusion",
    "Pleural Other",
    "Fracture",
    "No Finding",
]

class CheXpertMetaDataset(Dataset):
    """
    Returns:
      img: uint8 tensor [3, H, W]
      meta: dict with keys Sex, Age, View, Pleural Effusion
        - Sex: int64 {0,1}
        - Age: float32 in [0,1]
        - View: int64 {0,1}  (PA=0, AP=1) if AP/PA exists; else uses Frontal/Lateral mapping
        - Pleural Effusion: int64 {0,1}
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        train_csv: str | None = None,
        test_csv: str | None = None,
        require_pe: bool = True,
        allow_uncertain_as_pos: bool = True,
        age_min: float = 0.0,
        age_max: float = 100.0,
        transform = None,
        drop_missing_sex: bool = True,
        drop_missing_view: bool = True,
    ):
        if split == 'train':
            assert train_csv is not None, "train_csv required for split='train'"
            csv_path = train_csv
            print(f'Use {train_csv} for train split')
        else:
            assert test_csv is not None, "test_csv required for split='test'"
            csv_path = test_csv
            print(f'Use {test_csv} for test split')

        self.df = pd.read_csv(csv_path)
        self.data_dir = data_dir

        assert "Path" in self.df.columns, "CSV must contain 'Path'"
        assert (
            "Pleural Effusion" in self.df.columns
        ), "CSV must contain 'Pleural Effusion'"
        assert "Age" in self.df.columns, "CSV must contain 'Age'"
        assert "Sex" in self.df.columns, "CSV must contain 'Sex'"

        # Prefer AP/PA if exists, else fallback
        self.view_col = (
            "AP/PA"
            if "AP/PA" in self.df.columns
            else ("Frontal/Lateral" if "Frontal/Lateral" in self.df.columns else None)
        )
        if drop_missing_view:
            assert (
                self.view_col is not None
            ), "Need 'AP/PA' or 'Frontal/Lateral' in CSV for view."

        self.allow_uncertain_as_pos = allow_uncertain_as_pos
        self.age_min = float(age_min)
        self.age_max = float(age_max)
        assert self.age_max > self.age_min

        # --------------------
        # Row filtering
        # --------------------
        df = self.df

        if require_pe:
            df = df[df["Pleural Effusion"].notna()]

        if drop_missing_sex:
            df = df[df["Sex"].notna()]
            df = df[df["Sex"].astype(str).isin(["Male", "Female"])]

        if drop_missing_view and self.view_col is not None:
            df = df[df[self.view_col].notna()]
            if self.view_col == "AP/PA":
                df = df[df["AP/PA"].astype(str).str.upper().isin(["AP", "PA"])]

        # Age sanity
        df = df[df["Age"].notna()]

        self.df = df.reset_index(drop=True)

        self.transform = transform

    def __len__(self):
        return len(self.df)

    # --------------------
    # Helpers
    # --------------------
    def _map_sex(self, v):
        if isinstance(v, str):
            s = v.strip().lower()
            if s.startswith("m"):
                return 0
            if s.startswith("f"):
                return 1
        raise ValueError(f"Unknown Sex value: {v}")

    def _map_view(self, v):
        # Prefer AP/PA mapping if column is AP/PA
        if self.view_col == "AP/PA":
            if isinstance(v, str):
                s = v.strip().upper()
                if s == "PA":
                    return 0
                if s == "AP":
                    return 1
            raise ValueError(f"Unknown AP/PA value: {v}")

        # Fallback: Frontal/Lateral (not AP/PA)
        if self.view_col == "Frontal/Lateral":
            if isinstance(v, str):
                s = v.strip().lower()
                if s.startswith("front"):
                    return 0
                if s.startswith("lat"):
                    return 1
            raise ValueError(f"Unknown Frontal/Lateral value: {v}")

        raise ValueError("No view column configured")

    def _map_binary_label(self, val):
        """
        CheXpert labels often in {-1, 0, 1, NaN}
          1 -> positive
          0 -> negative
         -1 -> uncertain (treat as positive if allow_uncertain_as_pos else negative)
        """
        if pd.isna(val):
            raise ValueError(
                "Label is NaN but require_pe=False allowed it; you should handle this earlier."
            )
        if val == 1:
            return 1
        if val == 0:
            return 0
        if val == -1:
            return 1 if self.allow_uncertain_as_pos else 0
        # Some csvs use strings
        if isinstance(val, str):
            s = val.strip()
            if s in ("1", "1.0"):
                return 1
            if s in ("0", "0.0"):
                return 0
            if s in ("-1", "-1.0"):
                return 1 if self.allow_uncertain_as_pos else 0
        raise ValueError(f"Unknown label value: {val}")

    def _norm_age(self, age):
        a = float(age)
        a = max(self.age_min, min(self.age_max, a))
        return (a - self.age_min) / (self.age_max - self.age_min)

    def get_labels(self, target_key):
        """Return integer labels for all samples from df without loading images.

        Args:
            target_key: Metadata key ("Sex", "View", or a disease column such as "Pleural Effusion")

        Returns:
            list[int]: Integer label for each sample in the dataset
        """
        if target_key == "Sex":
            return self.df["Sex"].apply(self._map_sex).tolist()
        if target_key == "View":
            return self.df[self.view_col].apply(self._map_view).tolist()
        # Disease columns (e.g. "Pleural Effusion") are stored directly in df
        if target_key in self.df.columns:
            return self.df[target_key].apply(self._map_binary_label).tolist()
        raise ValueError(f"Unknown target_key for CheXpertMetaDataset: {target_key!r}")

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        sample_id = self.get_sample_id(idx)

        rel_path = row["Path"]
        img_path = os.path.join(self.data_dir, rel_path)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)  # float32 [3,H,W] normalized to [-1, 1]

        sex = self._map_sex(row["Sex"])
        age = self._norm_age(row["Age"])
        view = self._map_view(row[self.view_col]) if self.view_col is not None else 0
        pe = self._map_binary_label(row["Pleural Effusion"])

        meta = {
            "Sex": torch.tensor(sex, dtype=torch.long),
            "Age": torch.tensor(age, dtype=torch.float32),
            "View": torch.tensor(view, dtype=torch.long),
            "Pleural Effusion": torch.tensor(pe, dtype=torch.long),
        }
        return img, meta, sample_id

    def get_sample_id(self, idx):
        row = self.df.iloc[idx]
        path_str = row["Path"]
        sample_id = path_str.replace("/", "__")
        return os.path.splitext(sample_id)[0]

    def get_metadata(self, idx):
        row = self.df.iloc[idx]
        return {
            "Age": torch.tensor(self._norm_age(row["Age"]), dtype=torch.float32),
            "Sex": torch.tensor(self._map_sex(row["Sex"]), dtype=torch.long),
            "View": torch.tensor(
                self._map_view(row[self.view_col]), dtype=torch.long
            ),
            "Pleural Effusion": torch.tensor(
                self._map_binary_label(row["Pleural Effusion"]), dtype=torch.long
            ),
        }
