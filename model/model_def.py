import torch
import torch.nn as nn
from torchvision import models

def build_model(num_classes: int = 3, pretrained: bool = True) -> nn.Module:
    """ResNet-50 fine-tuned for skin lesion classification."""
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # Freeze backbone
    for param in model.parameters():
        param.requires_grad = False

    # Replace classifier head
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    return model

def load_model(path: str, num_classes: int = 3, device: str = "cpu") -> nn.Module:
    model = build_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model
