import torch
import torch.nn as nn
import torchvision.models as models

class HemoglobinEstimator(nn.Module):
    def __init__(self, use_metadata=False, metadata_dim=0, backbone='mobilenet'):
        """
        Hemoglobin estimation model with multiple backbone options
        
        Args:
            use_metadata: Whether to use metadata features
            metadata_dim: Dimension of metadata features
            backbone: Choice of backbone architecture
                     - 'mobilenet': MobileNetV3-Small (lightweight, default)
                     - 'efficientnet': EfficientNet-B0 (better accuracy)
                     - 'resnet18': ResNet-18 (good balance)
        """
        super().__init__()
        
        self.backbone_name = backbone
        
        # Select backbone architecture
        if backbone == 'efficientnet':
            self.backbone = models.efficientnet_b0(weights='DEFAULT')
            feature_dim = 1280
            self.backbone.classifier = nn.Identity()
            
        elif backbone == 'resnet18':
            self.backbone = models.resnet18(weights='DEFAULT')
            feature_dim = 512
            self.backbone.fc = nn.Identity()
            
        else:  # Default: mobilenet
            self.backbone = models.mobilenet_v3_small(weights='DEFAULT')
            feature_dim = 576
            self.backbone.classifier = nn.Identity()
        
        print(f"Using backbone: {backbone} (feature_dim={feature_dim})")
        
        # Metadata encoder (optional)
        self.use_metadata = use_metadata
        if use_metadata and metadata_dim > 0:
            self.meta_encoder = nn.Sequential(
                nn.Linear(metadata_dim, 32),
                nn.ReLU(),
                nn.Dropout(0.2)  # Reduced from 0.3
            )
            combined_dim = feature_dim + 32
        else:
            combined_dim = feature_dim
        
        # Regression head with reduced dropout for small datasets
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),  # Reduced from 0.4
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.1),  # Reduced from 0.3
            nn.Linear(64, 1)
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
        
        # Predict HgB value
        hgb_pred = self.regressor(combined)
        return hgb_pred.squeeze()
    
    def count_parameters(self):
        """Count total and trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable


# For backward compatibility and easy backbone switching
def create_model(backbone='mobilenet', use_metadata=False, metadata_dim=0):
    """
    Factory function to create model with specified backbone
    
    Usage:
        model = create_model('efficientnet')
        model = create_model('resnet18')
        model = create_model('mobilenet')  # default
    """
    return HemoglobinEstimator(
        use_metadata=use_metadata,
        metadata_dim=metadata_dim,
        backbone=backbone
    )


if __name__ == '__main__':
    # Test different backbones
    print("Testing model architectures:\n")
    
    for backbone in ['mobilenet', 'efficientnet', 'resnet18']:
        print(f"\n{'='*50}")
        model = create_model(backbone=backbone)
        total, trainable = model.count_parameters()
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")
        
        # Test forward pass
        dummy_img = torch.randn(2, 3, 224, 224)
        output = model(dummy_img, None)
        print(f"Output shape: {output.shape}")
        print(f"✓ {backbone} working correctly")