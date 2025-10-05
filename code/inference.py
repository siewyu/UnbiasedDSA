import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import argparse

from model import HemoglobinEstimator

def get_inference_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load metadata if provided
    use_metadata = args.meta is not None and os.path.exists(args.meta)
    if use_metadata:
        meta_df = pd.read_csv(args.meta)
        metadata_cols = [col for col in meta_df.columns if col != 'image_id']
        metadata_dim = len(metadata_cols)
    else:
        metadata_dim = 0
    
    # Load model
    model = HemoglobinEstimator(use_metadata=use_metadata, metadata_dim=metadata_dim)
    model.load_state_dict(torch.load('../weights/best_model.pt', map_location=device))
    model = model.to(device)
    model.eval()
    
    # Get image files
    image_files = [f for f in os.listdir(args.images) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    transform = get_inference_transform()
    predictions = []
    
    with torch.no_grad():
        for img_file in image_files:
            # Load and transform image
            img_path = os.path.join(args.images, img_file)
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            image = transform(image=image)['image']
            image = image.unsqueeze(0).to(device)
            
            # Get metadata if available
            image_id = os.path.splitext(img_file)[0]
            if use_metadata:
                meta_row = meta_df[meta_df['image_id'] == image_id]
                if len(meta_row) > 0:
                    metadata = torch.tensor(meta_row[metadata_cols].values, dtype=torch.float32).to(device)
                    pred = model(image, metadata)
                else:
                    pred = model(image, None)
            else:
                pred = model(image, None)
            
            predictions.append({
                'image_id': image_id,
                'hgb': pred.item()
            })
    
    # Save predictions
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True, help='Path to images directory')
    parser.add_argument('--meta', type=str, default=None, help='Path to metadata CSV')
    parser.add_argument('--out', type=str, required=True, help='Output predictions CSV')
    
    args = parser.parse_args()
    main(args)