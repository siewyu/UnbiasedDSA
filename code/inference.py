import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image

from model import HemoglobinEstimator
from dataset import get_transforms


def load_schema(data_dir):
    norm_path = os.path.join(data_dir, "normalization.json")
    meta_cols_path = os.path.join(data_dir, "metadata_columns.json")

    with open(norm_path, "r") as f:
        norm = json.load(f)
    hgb_min, hgb_max = float(norm["hgb_min"]), float(norm["hgb_max"])

    meta_cols = None
    if os.path.exists(meta_cols_path):
        with open(meta_cols_path, "r") as f:
            meta_cols = json.load(f)
    return hgb_min, hgb_max, meta_cols


def normalize_inv(y_norm, y_min, y_max):
    return y_norm * (y_max - y_min) + y_min


def build_metadata_vector(image_id, meta_df, meta_cols):
    if meta_df is None or meta_cols is None:
        return None
    row = meta_df[meta_df["image_id"] == image_id]
    if row.empty:
        return np.zeros(len(meta_cols), dtype=np.float32)

    onehot = np.zeros(len(meta_cols), dtype=np.float32)
    row0 = row.iloc[0]
    for i, key in enumerate(meta_cols):
        split_idx = key.find("_")
        if split_idx == -1:
            continue
        col = key[:split_idx]
        val = key[split_idx + 1:]
        if col in row0.index:
            v = str(row0[col]) if pd.notna(row0[col]) else "unknown"
            if f"{col}_{v}" == key:
                onehot[i] = 1.0
    return onehot


def main(args):
    device = torch.device("cpu")

    # load schema (JSONs) from data folder
    hgb_min, hgb_max, meta_cols = load_schema("../data")

    use_metadata = args.meta is not None and os.path.exists(args.meta) and (meta_cols is not None)
    meta_df = pd.read_csv(args.meta) if use_metadata else None

    model = HemoglobinEstimator(
        backbone_name="mobilenet_v3_small",
        use_metadata=use_metadata,
        metadata_dim=(len(meta_cols) if (use_metadata and meta_cols is not None) else 0),
    )

    state = torch.load(args.weights, map_location=device)
    model.load_state_dict(state)
    model.eval()

    image_files = [f for f in os.listdir(args.images) if f.lower().endswith((".jpg", ".jpeg", ".png", ".heic"))]
    tfm = get_transforms(train=False, size=args.img_size)

    preds = []
    with torch.no_grad():
        for fname in image_files:
            path = os.path.join(args.images, fname)
            image = Image.open(path).convert("RGB")
            image = np.array(image)
            image = tfm(image=image)["image"].unsqueeze(0)

            if use_metadata and meta_cols is not None:
                image_id = os.path.splitext(fname)[0]
                meta_vec = build_metadata_vector(image_id, meta_df, meta_cols)
                meta_t = torch.tensor(meta_vec).unsqueeze(0)
                y_norm = model(image, meta_t)
            else:
                y_norm = model(image)

            y = float(normalize_inv(y_norm.item(), hgb_min, hgb_max))
            preds.append({"image_id": os.path.splitext(fname)[0], "hgb": y})

    # ensure output folder exists
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    out_df = pd.DataFrame(preds)
    out_df.to_csv(args.out, index=False)
    print(f"✅ Saved predictions to {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--images", type=str, default="../data/images")
    p.add_argument("--weights", type=str, default="../weights/best_model.pt")
    p.add_argument("--meta", type=str, default="../data/meta.csv")
    p.add_argument("--out", type=str, default="../data/predictions.csv")
    p.add_argument("--img_size", type=int, default=192)
    args = p.parse_args()
    main(args)
