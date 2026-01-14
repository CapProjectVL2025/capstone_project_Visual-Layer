import torchvision.models as tvm
import timm
import torch.nn as nn


def build_model(model_name: str, num_classes: int):
    if model_name == "resnet18":
        model = tvm.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "resnet50":
        model = tvm.resnet50(weights=None)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "vit_base_patch32_224":
        model = timm.create_model(
            "vit_base_patch32_224",
            pretrained=False,
            num_classes=num_classes,
        )

    else:
        raise ValueError(
            f"Unknown model: {model_name}. "
            "Supported models: resnet18, resnet50, vit_base_patch32_224"
        )

    return model
