import os
from tqdm import tqdm
import torch
import torch.nn as nn
import wandb
import shutil

# ==============================================================
# 🧩 Training Loop (단일 epoch)
# ==============================================================

def train_one_epoch(model: nn.Module, dataloader, criterion, optimizer, device: torch.device):
    """
    Perform one training epoch.
    Returns: (avg_loss, avg_acc)
    """
    model.train()
    total_loss, total_correct = 0.0, 0

    for images, labels in tqdm(dataloader, desc="Train", leave=False):
        images, labels = images.to(device), labels.to(device)
        if labels.ndim == 1:
            labels = labels.unsqueeze(1)
        labels = labels.float()

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = (torch.sigmoid(outputs) > 0.5).float()
        total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = total_correct / len(dataloader.dataset)
    return avg_loss, avg_acc


# ==============================================================
# 🧩 Validation Loop
# ==============================================================

def validate(model: nn.Module, dataloader, criterion, device: torch.device):
    """
    Perform model validation.
    Returns: (avg_loss, avg_acc)
    """
    model.eval()
    total_loss, total_correct = 0.0, 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Valid", leave=False):
            images, labels = images.to(device), labels.to(device)
            if labels.ndim == 1:
                labels = labels.unsqueeze(1)
            labels = labels.float()

            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            preds = (torch.sigmoid(outputs) > 0.5).float()
            total_correct += (preds == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    avg_acc = total_correct / len(dataloader.dataset)
    return avg_loss, avg_acc


# ==============================================================
# 🧩 Train + Validate Full Pipeline
# ==============================================================
def train_model(
    model: nn.Module,
    train_loader,
    valid_loader,
    criterion,
    optimizer,
    device: torch.device,
    epochs: int,
    save_path: str,
    check_path: str,
    wandb_run=None,
):
    """
    Full training pipeline for classification model.

    Args:
        model: model instance
        train_loader: DataLoader for training
        valid_loader: DataLoader for validation
        criterion: loss function
        optimizer: optimizer
        device: torch.device
        epochs: total training epochs
        save_path: final best model save path (e.g. ./saved_model/mobilenet_v2_best.pt)
        check_path: checkpoint base path (e.g. ./checkpoints/mobilenet_v2_last.pt)
        wandb_run: optional wandb run object

    Behavior:
        - <check_dir>/mobilenet_v2_last.pt → 매 epoch마다 덮어쓰기
        - <check_dir>/mobilenet_v2_best.pt → 최고 성능 시 갱신
        - <save_dir>/mobilenet_v2_best.pt → 학습 종료 후 1회 복사
    """
    best_acc = 0.0

    # ✅ check_path 기반으로 last/best 경로 자동 지정
    check_dir = os.path.dirname(check_path)
    os.makedirs(check_dir, exist_ok=True)

    last_ckpt = check_path
    best_ckpt = check_path.replace("_last.pt", "_best.pt")

    for epoch in range(epochs):
        print(f"\n📘 Epoch {epoch + 1}/{epochs}")

        # ---- Training ----
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, valid_loader, criterion, device)

        print(f"📊 Train Loss: {train_loss:.4f} | Acc: {train_acc:.4f}")
        print(f"📈 Valid Loss: {val_loss:.4f} | Acc: {val_acc:.4f}")

        # ---- wandb logging ----
        if wandb_run is not None:
            wandb_run.log({
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "best_val_acc": best_acc,
            })

        # ---- Save last checkpoint ----
        torch.save(model.state_dict(), last_ckpt)

        # ---- Save best checkpoint ----
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), best_ckpt)
            print(f"💾 Best checkpoint updated: {best_ckpt}")

    # ✅ 학습 종료 후 best.pt → save_path 복사
    if os.path.exists(best_ckpt):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        shutil.copy2(best_ckpt, save_path)
        print(f"📦 Copied final best model to: {save_path}")

    print(f"\n🎯 Training Complete! Best Validation Accuracy: {best_acc:.4f}")
    return best_acc