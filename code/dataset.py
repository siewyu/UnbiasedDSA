try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("WARNING: pillow-heif not installed. HEIC files will fail.")

import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2


class HemoglobinDataset(Dataset):
    def __init__(
        self,
        images_dir,
        labels_csv,
        meta_csv=None,
        transform=None,
        use_metadata=True,
        normalize_targets=True,
        norm_min=None,
        norm_max=None,
        metadata_columns=None,  # when provided, aligns one-hots to this schema
    ):
        self.images_dir = images_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.transform = transform
        self.use_metadata = use_metadata and (meta_csv is not None) and os.path.exists(meta_csv)
        self.normalize_targets = normalize_targets

        # Target normalization params
        if self.normalize_targets:
            if norm_min is None or norm_max is None:
                self.hgb_min = float(self.labels_df["hgb"].min())
                self.hgb_max = float(self.labels_df["hgb"].max())
            else:
                self.hgb_min = float(norm_min)
                self.hgb_max = float(norm_max)
        else:
            self.hgb_min = None
            self.hgb_max = None

        # Metadata processing (one-hot)
        self.meta_onehot_cols = []
        if self.use_metadata:
            meta_df = pd.read_csv(meta_csv)
            df = self.labels_df.merge(meta_df, on="image_id", how="left")

            cat_cols = [
                "device_brand", "device_model", "camera_type",
                "iso_bucket", "exposure_bucket", "wb_bucket",
                "ambient_light", "distance_band", "skin_tone_proxy",
                "age_band", "gender",
            ]
            cat_cols = [c for c in cat_cols if c in df.columns]

            if metadata_columns is None:
                dummies = pd.get_dummies(df[cat_cols].fillna("unknown"), prefix=cat_cols)
                self.meta_onehot_cols = list(dummies.columns)
            else:
                self.meta_onehot_cols = list(metadata_columns)
                dummies = pd.get_dummies(df[cat_cols].fillna("unknown"), prefix=cat_cols)
                for col in self.meta_onehot_cols:
                    if col not in dummies.columns:
                        dummies[col] = 0
                dummies = dummies[self.meta_onehot_cols]

            df = pd.concat([df[["image_id", "hgb"]], dummies], axis=1)
            self.labels_df = df

    def __len__(self):
        return len(self.labels_df)

    def _normalize(self, y):
        if not self.normalize_targets:
            return torch.tensor(y, dtype=torch.float32)
        denom = max(1e-6, (self.hgb_max - self.hgb_min))
        y_norm = (y - self.hgb_min) / denom
        return torch.tensor(y_norm, dtype=torch.float32)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        image_id = row["image_id"]

        # resolve actual file extension
        img_path = None
        for ext in [".jpg", ".jpeg", ".png", ".heic", ".HEIC", ".JPG", ".JPEG", ".PNG"]:
            p = os.path.join(self.images_dir, f"{image_id}{ext}")
            if os.path.exists(p):
                img_path = p
                break
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {image_id}")

        image = Image.open(img_path).convert("RGB")
        image = np.array(image)

        if self.transform:
            image = self.transform(image=image)["image"]

        hgb = float(row["hgb"])
        hgb_t = self._normalize(hgb)

        if self.use_metadata and len(self.meta_onehot_cols) > 0:
            meta_vec = torch.tensor(row[self.meta_onehot_cols].values.astype(np.float32))
            return image, meta_vec, hgb_t

        return image, hgb_t


def get_transforms(train=True, size=192):
    # Simplified, color-stable augmentations (good for tiny datasets)
    if train:
        return A.Compose(
            [
                A.Resize(size, size),
                A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.3),
                A.GaussianBlur(blur_limit=(3, 5), p=0.2),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Resize(size, size),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ToTensorV2(),
            ]
        )
