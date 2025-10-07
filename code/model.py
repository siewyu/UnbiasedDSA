# model.py
# MobileNetV3-Small (default) or EfficientNet-B0 with optional metadata branch.

import torch
import torch.nn as nn
import torchvision.models as models


def build_backbone(name: str = "mobilenet_v3_small"):
    name = name.lower()
    if name in ["mobilenet", "mobilenet_v3_small", "mbv3"]:
        m = models.mobilenet_v3_small(weights="DEFAULT")
        feature_dim = 576
        m.classifier = nn.Identity()
        return m, feature_dim
    elif name in ["efficientnet_b0", "effb0", "efficientnet"]:
        m = models.efficientnet_b0(weights="DEFAULT")
        feature_dim = m.classifier[1].in_features  # 1280
        m.classifier = nn.Identity()
        return m, feature_dim
    else:
        raise ValueError(f"Unknown backbone: {name}")


class HemoglobinEstimator(nn.Module):
    def __init__(self, backbone_name="mobilenet_v3_small", use_metadata=True, metadata_dim=0):
        super().__init__()
        self.backbone, feature_dim = build_backbone(backbone_name)

        # full fine-tuning
        for p in self.backbone.parameters():
            p.requires_grad = True

        self.use_metadata = use_metadata and metadata_dim > 0
        if self.use_metadata:
            self.meta_encoder = nn.Sequential(
                nn.Linear(metadata_dim, 64),
                nn.ReLU(),
                nn.Dropout(0.35),
            )
            fuse_dim = feature_dim + 64
        else:
            fuse_dim = feature_dim

        self.regressor = nn.Sequential(
            nn.Linear(fuse_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1),
        )

    def forward(self, image, metadata=None):
        img_feat = self.backbone(image)
        if self.use_metadata and metadata is not None:
            meta_feat = self.meta_encoder(metadata)
            feat = torch.cat([img_feat, meta_feat], dim=1)
        else:
            feat = img_feat
        out = self.regressor(feat)
        return out.squeeze(1)
