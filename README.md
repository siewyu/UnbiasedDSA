# UnbiasedDSA - Hemoglobin Prediction from Lip Images

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Required Python Packages

- **numpy** - Numerical computations
- **pandas** - Data manipulation and CSV handling
- **Pillow (PIL)** - Image processing, HEIC file support
- **opencv-python** - Computer vision and image analysis
- **pathlib2** - Path handling utilities

### 3. Data Setup

**Place your image files** in the appropriate folder (update the `IMAGE_FOLDER` path in `extractdata.py`).

### 4. Run Data Extraction

```bash
cd code
python extractdata.py
```

This will generate:

- `labels.csv` - HgB labels extracted from filenames
- `meta.csv` - Image metadata following competition convention
- `combined_data.csv` - Combined dataset

### 5. File Formats

**labels.csv:**

```
image_id,hgb
HgB_10.7gdl_Individual01,10.7
HgB_12.0gdl_Individual02_1,12.0
...
```

**meta.csv:**

```
image_id,device_id,device_brand,device_model,iso_bucket,exposure_bucket,wb_bucket,ambient_light,distance_band,skin_tone_proxy,age_band,gender
HgB_10.7gdl_Individual01,Apple_iPhone,Apple,iPhone,medium,fast,auto,unknown,unknown,unknown,unknown,unknown
...
```

### 6. Manual Annotation Required

Some metadata fields are set to "unknown" and require manual annotation:

- `ambient_light` - bright/indoor/warm/cool
- `distance_band` - close/medium/far
- `skin_tone_proxy` - Fitzpatrick I-VI proxy
- `age_band` - age ranges
- `gender` - gender categories

### 7. Git Best Practices

**📁 What's tracked in Git:**

- Source code (`extractdata.py`, etc.)
- Documentation (`README.md`)
- Dependencies (`requirements.txt`)
- Configuration files

**🚫 What's NOT tracked (in `.gitignore`):**

- Generated CSV files (`*.csv`)
- Image data files (`*.jpg`, `*.heic`, etc.)
- Python cache files (`__pycache__/`)
- Virtual environments

This keeps your repository clean and avoids pushing large data files to GitHub.

### Troubleshooting

If you encounter issues with HEIC files:

```bash
pip install pillow-heif
```

For Windows users, if OpenCV installation fails:

```bash
pip install opencv-python-headless
```
