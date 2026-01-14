# models.py
import timm
import torchvision.models as tvmodels
import torch.nn as nn

def build_model(name, num_classes, pretrained=False, device="cpu"):
    """
    name: one of "vit_base_patch32_224", "resnet18", "resnet50"
    """
    name = name.lower()
    if "vit" in name or name.startswith("vit"):
        # use timm ViT variants
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    elif name.startswith("resnet50"):
        model = tvmodels.resnet50(weights=tvmodels.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif name.startswith("resnet18"):
        model = tvmodels.resnet18(weights=tvmodels.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    else:
        # fallback to timm general creation
        model = timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    model.to(device)
    return model
