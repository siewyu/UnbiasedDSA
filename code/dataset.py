# Enable HEIC support
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
    def __init__(self, images_dir, labels_csv, meta_csv=None, transform=None, use_metadata=False):
        self.images_dir = images_dir
        self.labels_df = pd.read_csv(labels_csv)
        self.transform = transform
        self.use_metadata = use_metadata
        
        if use_metadata and meta_csv is not None and os.path.exists(meta_csv):
            self.meta_df = pd.read_csv(meta_csv)
            self.labels_df = self.labels_df.merge(self.meta_df, on='image_id')
            
            # Only use numeric/encoded metadata columns
            self.metadata_cols = []
            for col in self.meta_df.columns:
                if col != 'image_id' and self.meta_df[col].dtype in ['int64', 'float64']:
                    self.metadata_cols.append(col)
        else:
            self.metadata_cols = []
    
    def __len__(self):
        return len(self.labels_df)
    
    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        
        # Find image with correct extension
        image_id = row['image_id']
        img_path = None
        
        for ext in ['.jpg', '.jpeg', '.png', '.heic', '.HEIC', '.JPG', '.JPEG', '.PNG']:
            potential_path = os.path.join(self.images_dir, f"{image_id}{ext}")
            if os.path.exists(potential_path):
                img_path = potential_path
                break
        
        if img_path is None:
            raise FileNotFoundError(f"Image not found: {image_id}")
        
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        if self.transform:
            transformed = self.transform(image=image)
            image = transformed['image']
        
        hgb = torch.tensor(row['hgb'], dtype=torch.float32)
        
        if self.use_metadata and len(self.metadata_cols) > 0:
            metadata = torch.tensor(row[self.metadata_cols].values, dtype=torch.float32)
            return image, metadata, hgb
        
        return image, hgb

def get_transforms(train=True):
    """
    Refined augmentations for hemoglobin prediction.
    Key: Preserve color information while adding variability.
    """
    if train:
        return A.Compose([
            A.Resize(224, 224),
            
            # REDUCED: Color shifts affect HgB signal
            A.HueSaturationValue(
                hue_shift_limit=5,      # Was 10 - reduced
                sat_shift_limit=10,     # Was 20 - reduced
                val_shift_limit=5,      # Was 10 - reduced
                p=0.3                    # Was 0.5 - reduced frequency
            ),
            
            # REDUCED: Brightness changes affect color perception
            A.RandomBrightnessContrast(
                brightness_limit=0.1,    # Was 0.2 - reduced
                contrast_limit=0.1,      # Was 0.2 - reduced
                p=0.3                    # Was 0.5 - reduced frequency
            ),
            
            # KEEP: Horizontal flip is safe (lips are symmetric)
            A.HorizontalFlip(p=0.5),
            
            # REDUCED: Less noise
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),  # Was (10.0, 30.0), p=0.3
            
            # NEW: Add slight rotation for position variance
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.05,
                rotate_limit=5,
                p=0.3,
                border_mode=0
            ),
            
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])