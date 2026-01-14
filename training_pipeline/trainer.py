import torch
from tqdm import tqdm
import wandb
from utils.logging import init_wandb  # make sure this exists


class Trainer:
    def __init__(self, model, optimizer, criterion, train_loader, val_loader, device, cfg):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.cfg = cfg

    def train(self, wandb_run=None):
        epochs = self.cfg["training"].get("epochs", 1)

        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch(epoch)
            val_loss, val_acc = self._eval_epoch(epoch)

            # W&B logging only if run exists
            if wandb_run:
                wandb_run.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "train/accuracy": train_acc,
                    "val/loss": val_loss,
                    "val/accuracy": val_acc,
                })

    def _train_epoch(self, epoch):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(self.train_loader, desc=f"Train {epoch}"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total

    @torch.no_grad()
    def _eval_epoch(self, epoch):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        for images, labels in tqdm(self.val_loader, desc=f"Val {epoch}"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        return total_loss / total, correct / total
