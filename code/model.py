#Switch to InceptionResNetV2

import torch
import torch.nn as nn
try:
    import timm
except ImportError:
    print("Installing timm for InceptionResNetV2...")
    import subprocess
    subprocess.check_call(['pip', 'install', 'timm'])
    import timm

class HemoglobinEstimator(nn.Module):
    def __init__(self, use_metadata=False, metadata_dim=0):
        super().__init__()
        
        self.backbone = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=0)
        feature_dim = 1536
        
        self.use_metadata = use_metadata
        if use_metadata and metadata_dim > 0:
            self.meta_encoder = nn.Sequential(
                nn.Linear(metadata_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            combined_dim = feature_dim + 32
        else:
            combined_dim = feature_dim
        
        # NO BatchNorm - causes issues with small batches
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, image, metadata=None):
        img_features = self.backbone(image)
        
        if self.use_metadata and metadata is not None:
            meta_features = self.meta_encoder(metadata)
            combined = torch.cat([img_features, meta_features], dim=1)
        else:
            combined = img_features
        
        hgb_pred = self.regressor(combined)
        return hgb_pred.squeeze()