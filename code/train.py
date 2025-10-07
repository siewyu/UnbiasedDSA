import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from tqdm import tqdm
import argparse
import numpy as np
from sklearn.model_selection import KFold

from model import HemoglobinEstimator
from dataset import HemoglobinDataset, get_transforms

def train_epoch(model, dataloader, criterion, optimizer, device, use_metadata):
    model.train()
    total_loss = 0
    
    for batch in tqdm(dataloader, desc="Training", leave=False):
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
            predictions = model(images, None)
        
        loss = criterion(predictions, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion, device, use_metadata):
    model.eval()
    total_loss = 0
    
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
                predictions = model(images, None)
            
            loss = criterion(predictions, labels)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def train_single_fold(fold, train_dataset, val_dataset, args, device):
    """Train a single fold"""
    print(f"\n{'='*70}")
    print(f"FOLD {fold + 1}")
    print(f"{'='*70}")
    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Create model
    model = HemoglobinEstimator(use_metadata=False, metadata_dim=0)
    model = model.to(device)
    
    # Loss function - try Huber for robustness
    if args.loss == 'huber':
        criterion = nn.HuberLoss(delta=1.0)
    elif args.loss == 'mse':
        criterion = nn.MSELoss()
    else:
        criterion = nn.L1Loss()
    
    # Optimizer with lower learning rate
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler with warmup
    warmup_epochs = 10
    warmup_scheduler = LinearLR(
        optimizer, 
        start_factor=0.1, 
        total_iters=warmup_epochs
    )
    main_scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=args.epochs - warmup_epochs
    )
    scheduler = SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, main_scheduler], 
        milestones=[warmup_epochs]
    )
    
    best_val_loss = float('inf')
    patience = args.patience
    epochs_without_improvement = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, False)
        val_loss = validate(model, val_loader, criterion, device, False)
        scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{args.epochs} - Train MAE: {train_loss:.4f}, Val MAE: {val_loss:.4f}, LR: {current_lr:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            
            # Save fold-specific model
            os.makedirs(args.output_dir, exist_ok=True)
            model_path = os.path.join(args.output_dir, f'best_model_fold{fold+1}.pt')
            torch.save(model.state_dict(), model_path)
            print(f"  ✓ Saved new best model (Val MAE: {val_loss:.4f})")
        else:
            epochs_without_improvement += 1
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break
    
    print(f"\nFold {fold+1} complete! Best Val MAE: {best_val_loss:.4f}")
    return best_val_loss

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Loss function: {args.loss}")
    print(f"Learning rate: {args.lr}")
    print(f"Weight decay: {args.weight_decay}\n")
    
    # Load full dataset
    full_dataset = HemoglobinDataset(
        images_dir=args.images_dir,
        labels_csv=args.labels_csv,
        meta_csv=None,
        transform=get_transforms(train=True),
        use_metadata=False
    )
    
    print(f"Total samples: {len(full_dataset)}\n")
    
    if args.use_kfold:
        # K-Fold Cross-Validation
        print(f"{'='*70}")
        print(f"K-FOLD CROSS-VALIDATION (k={args.n_folds})")
        print(f"{'='*70}")
        
        kfold = KFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
        fold_maes = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(full_dataset)))):
            # Create subsets
            train_subset = Subset(full_dataset, train_idx)
            val_subset = Subset(full_dataset, val_idx)
            
            # Train this fold
            best_mae = train_single_fold(fold, train_subset, val_subset, args, device)
            fold_maes.append(best_mae)
        
        # Summary
        print(f"\n{'='*70}")
        print("K-FOLD RESULTS SUMMARY")
        print(f"{'='*70}")
        for i, mae in enumerate(fold_maes):
            print(f"Fold {i+1}: {mae:.4f} g/dL")
        print(f"\nAverage MAE: {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f} g/dL")
        print(f"Best Fold: {np.argmin(fold_maes)+1} with MAE: {np.min(fold_maes):.4f} g/dL")
        print(f"{'='*70}")
        
        # Save summary
        with open(os.path.join(args.output_dir, 'kfold_results.txt'), 'w') as f:
            f.write(f"K-Fold Cross-Validation Results (k={args.n_folds})\n")
            f.write("="*70 + "\n\n")
            for i, mae in enumerate(fold_maes):
                f.write(f"Fold {i+1}: {mae:.4f} g/dL\n")
            f.write(f"\nAverage MAE: {np.mean(fold_maes):.4f} ± {np.std(fold_maes):.4f} g/dL\n")
            f.write(f"Best Fold: {np.argmin(fold_maes)+1} with MAE: {np.min(fold_maes):.4f} g/dL\n")
        
    else:
        # Single train/val split
        print("Using single train/val split (80/20)")
        from torch.utils.data import random_split
        
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(
            full_dataset, 
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed)
        )
        
        best_mae = train_single_fold(0, train_dataset, val_dataset, args, device)
        print(f"\nTraining complete! Best Val MAE: {best_mae:.4f}")
        
        # Rename model to standard name
        os.rename(
            os.path.join(args.output_dir, 'best_model_fold1.pt'),
            os.path.join(args.output_dir, 'best_model.pt')
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Hemoglobin Estimator')
    
    # Data paths
    parser.add_argument('--images_dir', type=str, default='../data/images')
    parser.add_argument('--labels_csv', type=str, default='../data/labels.csv')
    parser.add_argument('--output_dir', type=str, default='../weights')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)
    
    # K-Fold settings
    parser.add_argument('--use_kfold', action='store_true', help='Use K-fold cross-validation')
    parser.add_argument('--n_folds', type=int, default=5)
    
    # Loss function
    parser.add_argument('--loss', type=str, default='l1', choices=['l1', 'mse', 'huber'])
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    main(args)