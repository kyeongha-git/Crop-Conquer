import torch
import torch.nn as nn
import torchvision.models as models


class MobileNetClassifier(nn.Module):
    """
    Unified classifier for MobileNet V1, V2, and V3 (Large).

    Args:
        num_classes (int): number of output classes
        dropout_p (float): dropout probability for classifier
        model_type (str): 'mobilenet_v1', 'mobilenet_v2', or 'mobilenet_v3'
        freeze_backbone (bool): whether to freeze pretrained weights

    Example:
        model = MobileNetClassifier(num_classes=2, model_type='mobilenet_v3')
    """

    def __init__(
        self,
        num_classes: int = 1,
        dropout_p: float = 0.5,
        model_type: str = "mobilenet_v3",
        freeze_backbone: bool = True,
    ):
        super().__init__()

        self.model_type = model_type.lower()

        # 1️⃣ Load pretrained backbone
        if self.model_type == "mobilenet_v2":
            self.backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
            in_features = self.backbone.last_channel

        elif self.model_type == "mobilenet_v3":
            self.backbone = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V1
            )
            in_features = self.backbone.classifier[0].in_features

        else:
            raise ValueError(f"❌ Unsupported model type: {model_type}")

        # 2️⃣ Freeze backbone if required
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # 3️⃣ Replace classifier head
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the MobileNet backbone and custom classifier.

        Args:
            x (torch.Tensor): input tensor of shape (B, 3, H, W)
        Returns:
            torch.Tensor: output logits of shape (B, num_classes)
        """
        return self.backbone(x)
