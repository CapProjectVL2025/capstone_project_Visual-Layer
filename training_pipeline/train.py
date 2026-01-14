import yaml
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torch import nn, optim

from dataset import ImageDatasetFromCSV
from models import build_model
from trainer import Trainer
from utils.logging import init_wandb
from utils.seed import set_seed


def main(cfg_path):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    # W&B only if enabled
    run = init_wandb(cfg) if cfg.get("logging", {}).get("use", False) else None

    # Transforms
    transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

    # Datasets / loaders
    train_ds = ImageDatasetFromCSV(cfg["dataset"]["train_csv"], transforms)
    val_ds = ImageDatasetFromCSV(cfg["dataset"]["val_csv"], transforms)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Model / optimizer / loss
    num_classes = cfg["dataset"]["num_classes"]
    model = build_model(cfg["model"]["name"], num_classes).to(device)
    optimizer = optim.SGD(
        model.parameters(),
        lr=cfg["training"]["lr"],
        momentum=0.9,
        weight_decay=1e-4,
    )
    criterion = nn.CrossEntropyLoss()

    # Trainer
    trainer = Trainer(model, optimizer, criterion, train_loader, val_loader, device, cfg)
    trainer.train(run)

    if run:
        run.finish()


if __name__ == "__main__":
    import sys
    main(sys.argv[1])
