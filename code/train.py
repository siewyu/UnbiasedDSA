import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from model import HemoglobinEstimator
from dataset import HemoglobinDataset, get_transforms


# ----------------------------
# Train and Validation Epoch
# ----------------------------
def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, use_metadata):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        imgs = batch["image"].to(device)
        labels = batch["label"].to(device)
        metadata = batch["metadata"].to(device) if use_metadata else None

        optimizer.zero_grad()
        preds = model(imgs, metadata)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    scheduler.step()
    return total_loss / len(dataloader)


def val_epoch(model, dataloader, criterion, device, use_metadata):
    model.eval()
    total_loss = 0.0
    mae = 0.0
    with torch.no_grad():
        for batch in dataloader:
            imgs = batch["image"].to(device)
            labels = batch["label"].to(device)
            metadata = batch["metadata"].to(device) if use_metadata else None

            preds = model(imgs, metadata)
            loss = criterion(preds, labels)

            total_loss += loss.item()
            mae += torch.abs(preds - labels).mean().item()

    return total_loss / len(dataloader), mae / len(dataloader)


# ----------------------------
# Main Function
# ----------------------------
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    # Load CSVs
    labels_df = pd.read_csv(args.labels_csv)
    meta_df = pd.read_csv(args.meta_csv) if args.use_metadata else None

    labels = labels_df["hgb"].values
    indices = np.arange(len(labels))

    # ✅ Simple random split (instead of stratified)
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, random_state=42
    )

    print(f"Using device: {args.device}")
    print(f"Total samples: {len(indices)} | Train: {len(train_idx)} | Val: {len(val_idx)}")

    # Normalization
    mean, std = labels.mean(), labels.std()
    norm = {"mean": mean, "std": std}
    with open(os.path.join(args.output_dir, "normalization.json"), "w") as f:
        json.dump(norm, f, indent=4)

    labels_df["hgb_norm"] = (labels_df["hgb"] - mean) / std

    # Datasets
    train_dataset = HemoglobinDataset(
    args.images_dir,
    labels_df.iloc[train_idx],
    meta_df.iloc[train_idx] if meta_df is not None else None,
    transform=get_transforms(train=True, size=args.img_size),
    use_metadata=args.use_metadata,
    normalize=True,
    mean=mean,
    std=std
)

    val_dataset = HemoglobinDataset(
        args.images_dir,
        labels_df.iloc[val_idx],
        meta_df.iloc[val_idx] if meta_df is not None else None,
        transform=get_transforms(train=False, size=args.img_size),
        use_metadata=args.use_metadata,
        normalize=True,
        mean=mean,
        std=std
    )


    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Model setup
    metadata_dim = train_dataset.metadata_dim if args.use_metadata else 0
    model = HemoglobinEstimator(backbone_name=args.backbone, use_metadata=args.use_metadata, metadata_dim=metadata_dim)
    model = model.to(args.device)

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_loss = float("inf")
    best_val_mae = float("inf")
    patience_counter = 0

    # ----------------------------
    # Training Loop
    # ----------------------------
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, args.device, args.use_metadata)
        val_loss, val_mae = val_epoch(model, val_loader, criterion, args.device, args.use_metadata)

        print(f"Epoch {epoch+1}/{args.epochs} | Train loss (norm MAE): {train_loss:.4f} | Val loss (norm MAE): {val_loss:.4f}")
        print(f" → Train MAE (g/dL): {(train_loss * std):.2f} | Val MAE (g/dL): {(val_mae * std):.2f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pt"))
            print(f"✓ Saved new best model (val norm-MAE: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping at epoch {epoch+1}. Best val norm-MAE: {best_val_loss:.4f}")
                break

    print("\nTraining complete.")
    print(f"Best validation norm-MAE: {best_val_loss:.4f}")
    print(f"Best validation MAE (g/dL): {(best_val_mae * std):.2f}")


# ----------------------------
# Entry Point
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--labels_csv", type=str, required=True)
    parser.add_argument("--meta_csv", type=str, default=None)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--img_size", type=int, default=192)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--use_metadata", action="store_true")
    parser.add_argument("--backbone", type=str, default="mobilenet_v3_small")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--patience", type=int, default=5)
    args = parser.parse_args()

    main(args)
