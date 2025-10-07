import os
import json
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from model import HemoglobinEstimator
from dataset import HemoglobinDataset, get_transforms


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, use_metadata):
    model.train()
    running = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        if use_metadata:
            images, meta, labels = batch
            images, meta, labels = images.to(device), meta.to(device), labels.to(device)
            preds = model(images, meta)
        else:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            preds = model(images)

        loss = criterion(preds, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        running += loss.item()
    return running / max(1, len(dataloader))


def evaluate(model, dataloader, criterion, device, use_metadata):
    model.eval()
    running = 0.0
    with torch.no_grad():
        for batch in dataloader:
            if use_metadata:
                images, meta, labels = batch
                images, meta, labels = images.to(device), meta.to(device), labels.to(device)
                preds = model(images, meta)
            else:
                images, labels = batch
                images, labels = images.to(device), labels.to(device)
                preds = model(images)
            loss = criterion(preds, labels)
            running += loss.item()
    return running / max(1, len(dataloader))


def main(args):
    device = torch.device("cpu")
    print(f"Using device: {device}")

    base_dataset = HemoglobinDataset(
        images_dir=args.images_dir,
        labels_csv=args.labels_csv,
        meta_csv=args.meta_csv,
        transform=get_transforms(train=True, size=args.img_size),
        use_metadata=args.use_metadata,
        normalize_targets=True,
    )

    print(f"Total samples: {len(base_dataset)}")

    n_total = len(base_dataset)
    n_train = max(1, int(0.8 * n_total))
    n_val = n_total - n_train
    train_ds, val_ds = random_split(base_dataset, [n_train, n_val])
    val_ds.dataset.transform = get_transforms(train=False, size=args.img_size)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    metadata_dim = len(base_dataset.meta_onehot_cols) if args.use_metadata else 0

    model = HemoglobinEstimator(
        backbone_name=args.backbone,
        use_metadata=args.use_metadata,
        metadata_dim=metadata_dim,
    ).to(device)

    criterion = nn.SmoothL1Loss(beta=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.max_lr,
        epochs=args.epochs,
        steps_per_epoch=max(1, len(train_loader)),
        pct_start=0.3,
        div_factor=10.0,
        final_div_factor=10.0,
    )

    best_val = float("inf")
    patience = args.patience
    bad_epochs = 0

    os.makedirs(args.output_dir, exist_ok=True)

    # Save normalization + metadata info
    norm_path = os.path.join(args.output_dir, "normalization.json")
    meta_cols_path = os.path.join(args.output_dir, "metadata_columns.json")
    with open(norm_path, "w") as f:
        json.dump({"hgb_min": base_dataset.hgb_min, "hgb_max": base_dataset.hgb_max}, f)
    if args.use_metadata:
        with open(meta_cols_path, "w") as f:
            json.dump(base_dataset.meta_onehot_cols, f)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, args.use_metadata)
        val_loss = evaluate(model, val_loader, criterion, device, args.use_metadata)

        print(f"Epoch {epoch}/{args.epochs} | Train loss (norm MAE): {train_loss:.4f} | Val loss (norm MAE): {val_loss:.4f}")

        # Convert normalized MAE to actual scale
        val_loss_gdl = val_loss * (base_dataset.hgb_max - base_dataset.hgb_min)
        train_loss_gdl = train_loss * (base_dataset.hgb_max - base_dataset.hgb_min)
        print(f" → Train MAE (g/dL): {train_loss_gdl:.2f} | Val MAE (g/dL): {val_loss_gdl:.2f}")

        # Check for best model
        if val_loss < best_val - 1e-5:
            best_val = val_loss
            bad_epochs = 0
            torch.save(model.state_dict(), os.path.join("../weights", "best_model.pt"))
            print(f"  ✓ Saved new best model (val norm-MAE: {best_val:.4f})")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"Early stopping at epoch {epoch}. Best val norm-MAE: {best_val:.4f}")
                break

    # 🧾 Final summary outside the loop
    best_val_gdl = best_val * (base_dataset.hgb_max - base_dataset.hgb_min)
    print("\nTraining complete.")
    print(f"Best validation norm-MAE: {best_val:.4f}")
    print(f"Best validation MAE (g/dL): {best_val_gdl:.2f}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images_dir", type=str, default="../data/images")
    p.add_argument("--labels_csv", type=str, default="../data/labels.csv")
    p.add_argument("--meta_csv", type=str, default="../data/meta.csv")
    p.add_argument("--output_dir", type=str, default="../data")

    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--img_size", type=int, default=192)

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_lr", type=float, default=2e-4)

    p.add_argument("--use_metadata", action="store_true", default=True)
    p.add_argument("--backbone", type=str, default="mobilenet_v3_small",
                   choices=["mobilenet_v3_small", "efficientnet_b0"])
    p.add_argument("--patience", type=int, default=6)

    args = p.parse_args()
    main(args)
