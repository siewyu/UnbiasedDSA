import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import argparse
from glob import glob

from model import HemoglobinEstimator
from dataset import get_transforms

def load_model(model_path, device):
    """Load a single model"""
    model = HemoglobinEstimator(use_metadata=False, metadata_dim=0)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model

def predict_single_model(model, image_tensor, device):
    """Get prediction from a single model"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        pred = model(image_tensor, None)
    return pred.item()

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Find all model files
    if args.ensemble:
        model_files = glob(os.path.join(args.weights_dir, 'best_model_fold*.pt'))
        if len(model_files) == 0:
            print("No fold models found, falling back to single model...")
            model_files = [os.path.join(args.weights_dir, 'best_model.pt')]
    else:
        model_files = [os.path.join(args.weights_dir, 'best_model.pt')]
    
    print(f"Loading {len(model_files)} model(s):")
    for mf in model_files:
        print(f"  - {os.path.basename(mf)}")
    print()
    
    # Load all models
    models = []
    for model_path in model_files:
        if os.path.exists(model_path):
            models.append(load_model(model_path, device))
        else:
            print(f"Warning: Model not found: {model_path}")
    
    if len(models) == 0:
        print("ERROR: No models loaded!")
        return
    
    print(f"Successfully loaded {len(models)} model(s)\n")
    
    # Find all images
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.heic', '.HEIC', '.JPG', '.JPEG', '.PNG']:
        pattern = os.path.join(args.images, f'*{ext}')
        image_files.extend(glob(pattern))
    
    print(f"Found {len(image_files)} images\n")
    
    if len(image_files) == 0:
        print(f"ERROR: No images found in {args.images}")
        return
    
    transform = get_transforms(train=False)
    predictions = []
    
    print("Generating predictions...")
    for img_file in image_files:
        image_id = os.path.splitext(os.path.basename(img_file))[0]
        
        # Load and preprocess image
        image = Image.open(img_file).convert('RGB')
        image = np.array(image)
        image_tensor = transform(image=image)['image']
        image_tensor = image_tensor.unsqueeze(0)
        
        # Get predictions from all models
        model_preds = []
        for model in models:
            pred = predict_single_model(model, image_tensor, device)
            model_preds.append(pred)
        
        # Ensemble: average predictions
        if args.ensemble and len(model_preds) > 1:
            final_pred = np.mean(model_preds)
            std_pred = np.std(model_preds)
            
            predictions.append({
                'image_id': image_id, 
                'hgb_pred': final_pred,
                'hgb_std': std_pred  # Uncertainty estimate
            })
            print(f"  {image_id}: {final_pred:.2f} g/dL (±{std_pred:.3f})")
        else:
            predictions.append({
                'image_id': image_id, 
                'hgb_pred': model_preds[0]
            })
            print(f"  {image_id}: {model_preds[0]:.2f} g/dL")
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(args.out, index=False)
    
    print(f"\n{'='*70}")
    print(f"Predictions saved to: {args.out}")
    print(f"Total predictions: {len(pred_df)}")
    
    if args.ensemble and 'hgb_std' in pred_df.columns:
        print(f"Average uncertainty (std): {pred_df['hgb_std'].mean():.3f} g/dL")
        print(f"\nMost confident predictions (lowest std):")
        print(pred_df.nsmallest(3, 'hgb_std')[['image_id', 'hgb_pred', 'hgb_std']].to_string(index=False))
        print(f"\nLeast confident predictions (highest std):")
        print(pred_df.nlargest(3, 'hgb_std')[['image_id', 'hgb_pred', 'hgb_std']].to_string(index=False))
    
    print(f"{'='*70}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate hemoglobin predictions')
    parser.add_argument('--images', type=str, required=True, help='Path to images directory')
    parser.add_argument('--weights_dir', type=str, default='../weights', help='Directory containing model weights')
    parser.add_argument('--out', type=str, required=True, help='Output CSV file path')
    parser.add_argument('--ensemble', action='store_true', help='Use ensemble of all fold models')
    
    args = parser.parse_args()
    main(args)