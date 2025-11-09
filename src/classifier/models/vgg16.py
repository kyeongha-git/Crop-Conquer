import torch
import torch.nn as nn
import torchvision.models as models


class VGGClassifier(nn.Module):
    """
    Custom classifier based on pretrained VGG16.

    - Backbone: VGG16 (ImageNet pretrained)
    - Feature Extractor: frozen
    - Input: (3, 360, 360)
    - Flattened feature dim: 512 * 11 * 11
    - Classifier: Linear → Dropout → Linear
    """

    def __init__(self, num_classes: int = 1, dropout_p: float = 0.5):
        super().__init__()

        # 1️⃣ Load pretrained VGG16 backbone
        self.backbone = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1)

        # Freeze convolutional feature extractor
        for param in self.backbone.features.parameters():
            param.requires_grad = False

        # 2️⃣ Define classifier head (structure fixed)
        input_dim_flatten = 512 * 11 * 11
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim_flatten, 256),
            nn.Dropout(p=dropout_p),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward propagation through VGG16 feature extractor and custom classifier.

        Args:
            x (torch.Tensor): input tensor of shape (B, 3, 360, 360)
        Returns:
            torch.Tensor: model output logits (B, num_classes)
        """
        # Extract features using frozen VGG16 feature blocks
        features = self.backbone.features(x)

        # Apply custom classifier head
        logits = self.classifier(features)
        return logits
