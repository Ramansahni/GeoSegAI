import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from src.dataset import get_dataloader
from src.losses import BCEDiceLoss
from src.metrics import calculate_metrics
from src.model import UNet

def main():
    # Configuration
    batch_size = 4
    epochs = 25
    learning_rate = 1e-4
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Paths
    train_img_dir = "data/processed/patches/train/images"
    train_mask_dir = "data/processed/patches/train/masks"
    val_img_dir = "data/processed/patches/test/images"
    val_mask_dir = "data/processed/patches/test/masks"

    # Dataloaders
    train_loader = get_dataloader(train_img_dir, train_mask_dir, batch_size=batch_size, shuffle=True, split="all")
    val_loader = get_dataloader(val_img_dir, val_mask_dir, batch_size=batch_size, shuffle=False, split="val")

    # Model, Loss, Optimizer
    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # History lists to store metrics
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_iou": [], "val_iou": [],
        "train_dice": [], "val_dice": []
    }

    best_val_loss = float("inf")
    model_save_path = "eco_rgb_model.pth"

    # Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        # Accumulate values for epoch-level metrics calculation
        all_train_preds = []
        all_train_targets = []

        print(f"\nEpoch {epoch}/{epochs}")
        train_bar = tqdm(train_loader, desc="Training")
        for images, masks in train_bar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)

            # Store predictions and targets for epoch metrics
            all_train_preds.append(outputs.detach().cpu())
            all_train_targets.append(masks.cpu())

        train_loss /= len(train_loader.dataset)
        
        # Calculate epoch-level training metrics
        all_train_preds = torch.cat(all_train_preds, dim=0)
        all_train_targets = torch.cat(all_train_targets, dim=0)
        train_metrics = calculate_metrics(all_train_preds, all_train_targets)

        # Validation
        model.eval()
        val_loss = 0.0
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item() * images.size(0)
                all_val_preds.append(outputs.cpu())
                all_val_targets.append(masks.cpu())

        val_loss /= len(val_loader.dataset)
        
        # Calculate epoch-level validation metrics
        all_val_preds = torch.cat(all_val_preds, dim=0)
        all_val_targets = torch.cat(all_val_targets, dim=0)
        val_metrics = calculate_metrics(all_val_preds, all_val_targets)

        # Log epoch values
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_metrics["accuracy"])
        history["val_acc"].append(val_metrics["accuracy"])
        history["train_iou"].append(train_metrics["iou"])
        history["val_iou"].append(val_metrics["iou"])
        history["train_dice"].append(train_metrics["dice"])
        history["val_dice"].append(val_metrics["dice"])

        print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        print(f"Train Acc: {train_metrics['accuracy']:.4f} | Val Acc: {val_metrics['accuracy']:.4f}")
        print(f"Train IoU: {train_metrics['iou']:.4f} | Val IoU: {val_metrics['iou']:.4f}")
        print(f"Train Dice: {train_metrics['dice']:.4f} | Val Dice: {val_metrics['dice']:.4f}")

        # Save Best Model Checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            print(f"Saved new best model with Val Loss: {best_val_loss:.4f} to {model_save_path}")

    # Generate and Save History Graphs
    epochs_range = range(1, epochs + 1)
    os.makedirs("results", exist_ok=True)

    # 1. Train vs Val Loss
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_loss"], label="Train Loss", color='blue', marker='o')
    plt.plot(epochs_range, history["val_loss"], label="Val Loss", color='orange', marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/loss_curve.png")
    plt.close()

    # 2. Train vs Val Accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_acc"], label="Train Acc", color='blue', marker='o')
    plt.plot(epochs_range, history["val_acc"], label="Val Acc", color='orange', marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/accuracy_curve.png")
    plt.close()

    # 3. IoU Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_iou"], label="Train IoU", color='blue', marker='o')
    plt.plot(epochs_range, history["val_iou"], label="Val IoU", color='orange', marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Intersection over Union (IoU)")
    plt.title("Training vs Validation IoU Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/iou_curve.png")
    plt.close()

    # 4. Dice Curve
    plt.figure(figsize=(8, 5))
    plt.plot(epochs_range, history["train_dice"], label="Train Dice", color='blue', marker='o')
    plt.plot(epochs_range, history["val_dice"], label="Val Dice", color='orange', marker='x')
    plt.xlabel("Epoch")
    plt.ylabel("Dice Score")
    plt.title("Training vs Validation Dice Score Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/dice_curve.png")
    plt.close()

    # Save training history array
    np.save("results/history.npy", history)
    print("\nTraining complete! Metrics and curves saved in 'results/' directory.")

if __name__ == "__main__":
    main()