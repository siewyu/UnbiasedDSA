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

#Use CLAHE preprocessing for better results
def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(224, 224),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),  # ADD THIS LINE
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),  # ADD THIS LINE
            A.Rotate(limit=30, p=0.5),  # ADD THIS LINE  
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(224, 224),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),  # ADD THIS LINE
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])