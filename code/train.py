import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import argparse

from model import HemoglobinEstimator
from dataset import HemoglobinDataset, get_transforms

def train_epoch(model, dataloader, criterion, optimizer, device, use_metadata):
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
                predictions = model(images)
            
            loss = criterion(predictions, labels)
            total_loss += loss.item()
    
    return total_loss / len(dataloader)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    full_dataset = HemoglobinDataset(
    images_dir=args.images_dir,
    labels_csv=args.labels_csv,
    meta_csv=args.meta_csv,  # Changed from None
    transform=get_transforms(train=True),
    use_metadata=args.use_metadata  # Changed from False
    )
    
    print(f"Total samples: {len(full_dataset)}")
    
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    val_dataset.dataset.transform = get_transforms(train=False)
    
    print(f"Training: {train_size}, Validation: {val_size}\n")
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    model = HemoglobinEstimator(
    use_metadata=args.use_metadata, 
    metadata_dim=len(full_dataset.metadata_cols) if args.use_metadata else 0
    )
    model = model.to(device)
    
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-3)
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, args.use_metadata)
        val_loss = validate(model, val_loader, criterion, device, args.use_metadata)
        
        val_mae = val_loss ** 0.5
        
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Est. MAE: {val_mae:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            os.makedirs(args.output_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  --> Saved new best model")
        else:
            patience_counter += 1
            
        if patience_counter >= 15:
            print("Early stopping triggered")
            break
    
    print(f"\nTraining complete! Best Val Loss: {best_val_loss:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images_dir', type=str, default='../data/images')
    parser.add_argument('--labels_csv', type=str, default='../data/labels.csv')
    parser.add_argument('--output_dir', type=str, default='../weights')
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--meta_csv', type=str, default='../data/meta.csv')
    parser.add_argument('--use_metadata', action='store_true')
    
    args = parser.parse_args()
    main(args)