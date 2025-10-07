import torch
import torch.nn as nn
import torchvision.models as models

class HemoglobinEstimator(nn.Module):
    def __init__(self, use_metadata=False, metadata_dim=0):
        super().__init__()
        
        # MobileNetV3-Small: lightweight, good for edge deployment
        self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
        feature_dim = 576
        
        self.backbone.classifier = nn.Identity()
        
        self.use_metadata = use_metadata
        if use_metadata and metadata_dim > 0:
            self.meta_encoder = nn.Sequential(
                nn.Linear(metadata_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.3)
            )
            combined_dim = feature_dim + 32
        else:
            combined_dim = feature_dim
        
        # IMPROVED: More capacity + batch norm for better convergence
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 512),      # Was 256 - increased
            nn.BatchNorm1d(512),                # NEW: Batch normalization
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),                # NEW: Additional layer
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),                # Was 64 - increased
            nn.BatchNorm1d(128),
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