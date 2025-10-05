import os
import torch
import pandas as pd
from PIL import Image
import numpy as np
import argparse

from model import HemoglobinEstimator
from dataset import get_transforms

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = HemoglobinEstimator(use_metadata=False, metadata_dim=0)
    model.load_state_dict(torch.load('../weights/best_model.pt', map_location=device))
    model = model.to(device)
    model.eval()
    
    # Find all images
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.heic', '.HEIC', '.JPG', '.JPEG', '.PNG']:
        image_files.extend([f for f in os.listdir(args.images) if f.endswith(ext)])
    
    transform = get_transforms(train=False)
    predictions = []
    
    with torch.no_grad():
        for img_file in image_files:
            img_path = os.path.join(args.images, img_file)
            image = Image.open(img_path).convert('RGB')
            image = np.array(image)
            image = transform(image=image)['image']
            image = image.unsqueeze(0).to(device)
            
            pred = model(image, None)
            
            image_id = os.path.splitext(img_file)[0]
            predictions.append({'image_id': image_id, 'hgb': pred.item()})
    
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(args.out, index=False)
    print(f"Saved predictions to {args.out}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', type=str, required=True)
    parser.add_argument('--meta', type=str, default=None)
    parser.add_argument('--out', type=str, required=True)
    
    args = parser.parse_args()
    main(args)