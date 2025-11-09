import torch
import torch.nn as nn
import torchvision.models as models


class ResNet152Classifier(nn.Module):
    """
    Custom classifier built on top of pretrained ResNet-152.

    - Backbone: Pretrained ResNet-152 (ImageNet)
    - Feature extractor: frozen by default
    - Custom head: AdaptiveAvgPool2d → Flatten → Linear
    - Input: (B, 3, 360, 360)
    - Output: (B, num_classes)
    """

    def __init__(self, num_classes: int = 1, freeze_backbone: bool = True):
        super().__init__()

        # 1️⃣ Load pretrained ResNet152 backbone
        self.backbone = models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1)

        # Remove global avgpool & fc from backbone (we'll handle it manually)
        in_features = self.backbone.fc.in_features
        self.backbone.avgpool = nn.Identity()
        self.backbone.fc = nn.Identity()

        # 2️⃣ Freeze backbone parameters (optional)
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 3️⃣ Define custom classifier head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # global average pooling
            nn.Flatten(start_dim=1),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation through the ResNet backbone and classifier.

        Args:
            x (torch.Tensor): input tensor of shape (B, 3, H, W)

        Returns:
            torch.Tensor: logits tensor of shape (B, num_classes)
        """
        # Extract feature maps (up to layer4)
        features = self.backbone.conv1(x)
        features = self.backbone.bn1(features)
        features = self.backbone.relu(features)
        features = self.backbone.maxpool(features)

        features = self.backbone.layer1(features)
        features = self.backbone.layer2(features)
        features = self.backbone.layer3(features)
        features = self.backbone.layer4(features)

        # Classification head
        logits = self.classifier(features)
        return logits
