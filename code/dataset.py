import os
import pandas as pd
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class HemoglobinDataset(Dataset):
    def __init__(self, images_dir, labels_csv, meta_csv=None, transform=None, use_metadata=False):
        """
        Args:
            images_dir: Path to folder containing images
            labels_csv: Path to CSV with columns [image_id, hgb]
            meta_csv: Path to CSV with metadata (optional)
            transform: Albumentations transform
            use_metadata: Whether to use metadata features
        """
        self.images_dir = images_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.transform = transform
        self.use_metadata = use_metadata
        
        if use_metadata and meta_csv is not None:
            self.meta_df = pd.read_csv(meta_csv)
            self.labels_df = self.labels_df.merge(self.meta_df, on='image_id')
            self.metadata_cols = [col for col in self.meta_df.columns if col != 'image_id']
        else:
            self.metadata_cols = []
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        
        # Load image
        img_name = f"{row['image_id']}.jpg"
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        # Get label
        hgb = torch.tensor(row['hgb'], dtype=torch.float32)
        
        # Get metadata if using
        if self.use_metadata and len(self.metadata_cols) > 0:
            metadata = torch.tensor(row[self.metadata_cols].values, dtype=torch.float32)
            return image, metadata, hgb
        
        return image, hgb

def get_transforms(train=True):
    """Get image transforms"""
    if train:
        return A.Compose([
            A.Resize(224, 224),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])