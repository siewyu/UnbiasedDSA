import torch
import torch.nn as nn
import torchvision.models as models

class HemoglobinEstimator(nn.Module):
    def __init__(self, use_metadata=False, metadata_dim=0):
        super().__init__()
        
        # Use MobileNetV3-Small as backbone (lightweight, good for edge deployment)
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        feature_dim = 576  # MobileNetV3-Small feature dimension
        
        # Remove original classifier
        self.backbone.classifier = nn.Identity()
        
        # Metadata processing
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
        
        # Regression head for hemoglobin prediction
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Single output: HgB value
        )
    
    def forward(self, image, metadata=None):
        # Extract image features
        img_features = self.backbone(image)
        
        # Combine with metadata if available
        if self.use_metadata and metadata is not None:
            meta_features = self.meta_encoder(metadata)
            combined = torch.cat([img_features, meta_features], dim=1)
        else:
            combined = img_features
        
        # Predict hemoglobin
        hgb_pred = self.regressor(combined)
        return hgb_pred.squeeze()