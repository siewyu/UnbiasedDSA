import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import argparse
import albumentations as A
from albumentations.pytorch import ToTensorV2

from model import HemoglobinEstimator

def get_tta_transforms():
    """Test-Time Augmentation transforms"""
    base_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    flip_transform = A.Compose([
        A.Resize(224, 224),
        A.HorizontalFlip(p=1.0),  # Always flip
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    
    return [base_transform, flip_transform]

def predict_with_tta(model, image_array, transforms, device):
    """
    Predict with Test-Time Augmentation.
    Average predictions from multiple augmented versions.
    """
    predictions = []
    
    with torch.no_grad():
        for transform in transforms:
            img_transformed = transform(image=image_array)['image']
            img_tensor = img_transformed.unsqueeze(0).to(device)
            pred = model(img_tensor, None)
            predictions.append(pred.item())
    
    # Return mean of all predictions
    return np.mean(predictions)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Using Test-Time Augmentation: {args.use_tta}\n")
    
    # Load model
    model = HemoglobinEstimator(use_metadata=False, metadata_dim=0)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Find all images
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.heic', '.HEIC', '.JPG', '.JPEG', '.PNG']:
        image_files.extend([f for f in os.listdir(args.images) if f.endswith(ext)])
    
    print(f"Found {len(image_files)} images\n")
    
    # Get transforms
    if args.use_tta:
        transforms = get_tta_transforms()
    else:
        transforms = [A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])]
    
    predictions = []
    
    for img_file in image_files:
        img_path = os.path.join(args.images, img_file)
        image = Image.open(img_path).convert('RGB')
        image_array = np.array(image)
        
        # Predict with or without TTA
        pred = predict_with_tta(model, image_array, transforms, device)
        
        image_id = os.path.splitext(img_file)[0]
        predictions.append({'image_id': image_id, 'hgb': pred})
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(args.out, index=False)
    print(f"✓ Saved predictions to {args.out}")
    print(f"  Mean predicted HgB: {pred_df['hgb'].mean():.2f} g/dL")
    print(f"  Std predicted HgB: {pred_df['hgb'].std():.2f} g/dL")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True, help='Path to images directory')
    parser.add_argument('--weights', type=str, default='../weights/best_model.pt', help='Path to model weights')
    parser.add_argument('--out', type=str, required=True, help='Output CSV path')
    parser.add_argument('--use_tta', action='store_true', help='Use test-time augmentation')
    
    args = parser.parse_args()
    main(args)