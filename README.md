# UnbiasedDSA - Hemoglobin Prediction from Lip Images

A machine learning pipeline for estimating hemoglobin levels from lip images using computer vision and deep learning.

## Project Structure
```
UnbiasedDSA/
├── code/
│   ├── extractdata.py    # Data extraction from images
│   ├── dataset.py         # PyTorch dataset loader
│   ├── model.py           # Neural network architecture
│   ├── train.py           # Training script
│   ├── inference.py       # Prediction script
│   ├── evaluate.py        # Model evaluation script
│   └── convert_heic.py    # HEIC to JPG converter
├── data/
│   ├── images/            # Input images (HEIC/JPG)
│   ├── labels.csv         # HgB labels
│   ├── meta.csv           # Image metadata
│   └── combined_data.csv  # Combined dataset
├── weights/
│   └── best_model.pt      # Trained model weights
├── requirements.txt       # Python dependencies
├── model_card.md          # Model documentation
└── report.pdf             # Technical report (2-3 pages)
```

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Required Python Packages
```
torch>=2.0.0
torchvision>=0.15.0
pillow>=10.0.0
pillow-heif>=0.13.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
albumentations>=1.3.0
opencv-python>=4.8.0
tqdm>=4.66.0
```

### 3. Prepare Your Data

**Step 1:** Place all lip images in a folder (e.g., `1. Randomised Files/`)

**Step 2:** Update the `IMAGE_FOLDER` path in `code/extractdata.py`:
```python
IMAGE_FOLDER = r"C:\path\to\your\images"
```

**Step 3:** Run data extraction:
```bash
cd code
python extractdata.py
```

This generates:
- `data/labels.csv` - HgB values extracted from filenames
- `data/meta.csv` - Camera metadata (device, ISO, exposure, etc.)
- `data/combined_data.csv` - Complete dataset

### 4. Handle HEIC Files (If Applicable)

If your images are in HEIC format, convert them to JPG:
```bash
cd code
python convert_heic.py
```

This converts all `.heic` files in `data/images/` to `.jpg` format.

## Model Training & Evaluation Pipeline

### Step 1: Train the Model
```bash
cd code
python train.py --epochs 50 --batch_size 4
```

**Training options:**
- `--epochs`: Number of training epochs (default: 50)
- `--batch_size`: Batch size (default: 2)
- `--lr`: Learning rate (default: 1e-4)
- `--images_dir`: Path to images (default: ../data/images)
- `--labels_csv`: Path to labels (default: ../data/labels.csv)
- `--output_dir`: Output directory for model (default: ../weights)

**Expected output:**
- Progress bars showing training/validation
- Epoch-by-epoch MAE metrics
- Best model saved to `weights/best_model.pt`

### Step 2: Run Inference

Generate predictions on images:
```bash
python inference.py --images ../data/images --out predictions.csv
```

Output format (`predictions.csv`):
```csv
image_id,hgb
HgB_10.7gdl_Individual01,10.8
HgB_12.0gdl_Individual02,11.9
...
```

### Step 3: Evaluate Model Performance

Run the evaluation script to assess your model:
```bash
python evaluate.py
```

This will:
- Calculate Mean Absolute Error (MAE) and other metrics
- Display sample predictions vs ground truth
- Save detailed results to `evaluation_results.csv`
- Compare performance against competition target (0.8 g/dL MAE)

## Model Architecture

- **Backbone**: MobileNetV3-Small (pre-trained on ImageNet)
- **Head**: Fully connected regression layers with dropout
- **Loss**: Mean Absolute Error (MAE)
- **Optimizer**: AdamW with cosine annealing schedule
- **Input**: 224x224 RGB images
- **Model Size**: ~5 MB (meets edge-lite requirements)

## Data Augmentation

Training augmentations:
- Random horizontal flip
- Random brightness/contrast adjustment
- Hue/saturation/value shifts
- Gaussian noise
- ImageNet normalization

## File Format Specifications

**labels.csv:**
```csv
image_id,hgb
HgB_10.7gdl_Individual01,10.7
HgB_12.0gdl_Individual02_1,12.0
```

**meta.csv:**
```csv
image_id,device_id,device_brand,device_model,camera_type,iso_bucket,exposure_bucket,wb_bucket,ambient_light,distance_band,skin_tone_proxy,age_band,gender
HgB_10.7gdl_Individual01,Apple_iPhone_15_Pro_Max,Apple,iPhone 15 Pro Max,front,medium,fast,auto,indoor,close,unknown,unknown,unknown
```

## Metadata Fields

**Automatically extracted:**
- `device_brand` - Camera manufacturer (Apple, Samsung, etc.)
- `device_model` - Specific device model
- `camera_type` - front/back camera detection
- `iso_bucket` - ISO sensitivity (low/medium/high)
- `exposure_bucket` - Exposure time (fast/medium/slow)
- `wb_bucket` - White balance setting (auto/manual/daylight)

**Inferred from context:**
- `ambient_light` - Lighting condition (indoor assumed for medical setting)
- `distance_band` - Capture distance (close assumed for lip close-ups)

**Requires manual annotation:**
- `skin_tone_proxy` - Fitzpatrick scale (I-VI)
- `age_band` - Age categories
- `gender` - Gender categories


## Troubleshooting

### HEIC Files Won't Open
```bash
pip install pillow-heif
```

### OpenCV Installation Fails (Windows)
```bash
pip install opencv-python-headless
```

### Out of Memory During Training

Reduce batch size:
```bash
python train.py --batch_size 2
```
