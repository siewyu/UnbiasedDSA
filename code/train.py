import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse

from model import HemoglobinEstimator
from dataset import HemoglobinDataset, get_transforms

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, use_metadata):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        if use_metadata:
            images, metadata, labels = batch
            images = images.to(device)
            metadata = metadata.to(device)
            labels = labels.to(device)
            predictions = model(images, metadata)
        else:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            predictions = model(images)
        
        loss = criterion(predictions, labels)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()  # Step per batch for OneCycleLR
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, device, use_metadata):
    """Validate and return MAE"""
    model.eval()
    total_mae = 0
    
    with torch.no_grad():
        for batch in dataloader:
            if use_metadata:
                images, metadata, labels = batch
                images = images.to(device)
                metadata = metadata.to(device)
                labels = labels.to(device)
                predictions = model(images, metadata)
            else:
                images, labels = batch
                images = images.to(device)
                labels = labels.to(device)
                predictions = model(images)
            
            # Calculate MAE
            mae = torch.abs(predictions - labels).mean()
            total_mae += mae.item()
    
    return total_mae / len(dataloader)

def main(args):
    print("="*70)
    print("HEMOGLOBIN TRAINING PIPELINE")
    print("="*70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[1/6] Device: {device}")
    
    # Check paths exist
    print(f"\n[2/6] Checking paths...")
    if not os.path.exists(args.images_dir):
        print(f"ERROR: Images directory not found: {args.images_dir}")
        return
    print(f"  ✓ Images directory exists: {args.images_dir}")
    
    if not os.path.exists(args.labels_csv):
        print(f"ERROR: Labels CSV not found: {args.labels_csv}")
        return
    print(f"  ✓ Labels CSV exists: {args.labels_csv}")
    
    # Count images
    image_count = 0
    for ext in ['.jpg', '.jpeg', '.png', '.heic', '.HEIC', '.JPG', '.JPEG', '.PNG']:
        image_count += len([f for f in os.listdir(args.images_dir) if f.endswith(ext)])
    print(f"  ✓ Found {image_count} images in directory")
    
    # Load dataset
    print(f"\n[3/6] Loading dataset...")
    try:
        full_dataset = HemoglobinDataset(
            images_dir=args.images_dir,
            labels_csv=args.labels_csv,
            meta_csv=None,
            transform=get_transforms(train=True),
            use_metadata=False
        )
        print(f"  ✓ Dataset loaded successfully")
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print(f"  Total samples in dataset: {len(full_dataset)}")
    
    if len(full_dataset) == 0:
        print("  ERROR: No samples loaded! Check if image filenames match labels.csv")
        return
    
    # Split train/val
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    print(f"  Training samples: {train_size}")
    print(f"  Validation samples: {val_size}")
    
    # Create dataloaders
    print(f"\n[4/6] Creating dataloaders (batch_size={args.batch_size})...")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    print(f"  ✓ Train batches: {len(train_loader)}")
    print(f"  ✓ Val batches: {len(val_loader)}")
    
    # Create model
    print(f"\n[5/6] Creating model...")
    model = HemoglobinEstimator(use_metadata=False, metadata_dim=0)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  ✓ Model created")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    # Use MSE loss for training (smoother gradients)
    criterion = nn.MSELoss()
    
    # Higher learning rate with AdamW
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=3e-4,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )
    
    # OneCycleLR scheduler with warmup
    from torch.optim.lr_scheduler import OneCycleLR
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=3e-4,
        epochs=args.epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos',
        div_factor=25.0,
        final_div_factor=1e4
    )
    
    print(f"\n[6/6] Starting training...")
    print(f"  Epochs: {args.epochs}")
    print(f"  Learning rate: 3e-4 (OneCycleLR)")
    print(f"  Loss function: MSE (training), MAE (validation)")
    print(f"  Output directory: {args.output_dir}")
    print("="*70)
    
    best_val_mae = float('inf')
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-"*70)
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, False)
        
        # Validate with MAE
        val_mae = validate(model, val_loader, device, False)
        
        current_lr = scheduler.get_last_lr()[0]
        print(f"Train MSE Loss: {train_loss:.4f} | Val MAE: {val_mae:.4f} | LR: {current_lr:.6f}")
        
        # Save best model based on validation MAE
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  ✓ Saved new best model (Val MAE: {val_mae:.4f})")
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Best Validation MAE: {best_val_mae:.4f} g/dL")
    print(f"Model saved to: {os.path.join(args.output_dir, 'best_model.pt')}")
    print(f"Target MAE: 0.8000 g/dL")
    print(f"Gap: {best_val_mae - 0.8:+.4f} g/dL")
    print("="*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='../data/images')
    parser.add_argument('--labels_csv', type=str, default='../data/labels.csv')
    parser.add_argument('--output_dir', type=str, default='../weights')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=3e-4)
    
    args = parser.parse_args()
    main(args)