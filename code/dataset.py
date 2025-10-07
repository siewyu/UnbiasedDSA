# Enable HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    print("WARNING: pillow-heif not installed. HEIC files will fail.")

import cv2
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
            
            # Encode categorical metadata
            self.metadata_cols = []
            
            # Device brand encoding - ADD .astype(int) here
            if 'device_brand' in self.labels_df.columns:
                self.labels_df['device_brand_encoded'] = pd.Categorical(self.labels_df['device_brand']).codes.astype(int)
                self.metadata_cols.append('device_brand_encoded')
            
            # Device model encoding - ADD .astype(int) here
            if 'device_model' in self.labels_df.columns:
                self.labels_df['device_model_encoded'] = pd.Categorical(self.labels_df['device_model']).codes.astype(int)
                self.metadata_cols.append('device_model_encoded')
            
            # Camera type encoding - ADD .astype(int) here
            if 'camera_type' in self.labels_df.columns:
                self.labels_df['camera_type_encoded'] = pd.Categorical(self.labels_df['camera_type']).codes.astype(int)
                self.metadata_cols.append('camera_type_encoded')
            
            # ISO bucket encoding
            if 'iso_bucket' in self.labels_df.columns:
                iso_map = {'low': 0, 'medium': 1, 'high': 2, 'unknown': 3}
                self.labels_df['iso_encoded'] = self.labels_df['iso_bucket'].map(iso_map).fillna(3).astype(int)
                self.metadata_cols.append('iso_encoded')
            
            # Exposure bucket encoding
            if 'exposure_bucket' in self.labels_df.columns:
                exp_map = {'fast': 0, 'medium': 1, 'slow': 2, 'unknown': 3}
                self.labels_df['exposure_encoded'] = self.labels_df['exposure_bucket'].map(exp_map).fillna(3).astype(int)
                self.metadata_cols.append('exposure_encoded')
            
            # White balance encoding
            if 'wb_bucket' in self.labels_df.columns:
                wb_map = {'auto': 0, 'manual': 1, 'daylight': 2, 'unknown': 3}
                self.labels_df['wb_encoded'] = self.labels_df['wb_bucket'].map(wb_map).fillna(3).astype(int)
                self.metadata_cols.append('wb_encoded')
            
            print(f"Using {len(self.metadata_cols)} metadata features: {self.metadata_cols}")
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
            # Explicitly convert to float32
            metadata_values = []
            for col in self.metadata_cols:
                val = row[col]
                try:
                    metadata_values.append(float(val))
                except (ValueError, TypeError):
                    metadata_values.append(0.0)
            
            metadata = torch.tensor(metadata_values, dtype=torch.float32)
            return image, metadata, hgb
        
        return image, hgb

def get_transforms(train=True):
    if train:
        return A.Compose([
            A.Resize(299, 299),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.5),
            A.GaussNoise(var_limit=(10.0, 30.0), p=0.3),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(299, 299),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])