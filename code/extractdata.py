"""
Extract labels and metadata from hemoglobin lip images
Creates: labels.csv and metadata.csv
"""

import os
import re
import pandas as pd
from pathlib import Path
from PIL import Image
from PIL.ExifTags import TAGS
import cv2
import numpy as np

# Enable HEIC support
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print("HEIC support enabled")
except ImportError:
    print("pillow-heif not installed. HEIC files may not work properly.")
    print("Install with: pip install pillow-heif")

def extract_hgb_from_filename(filename):
    """
    Extract HgB value from various filename patterns.
    
    Examples:
        'HgB_8.0gdl_Individual03_01.jpg' -> 8.0
        'HgB_10.7gdl_Individual01.heic' -> 10.7
        'Random_7.8gdl_ChineseOrigin.jpg' -> 7.8
        'Random_HgB_4.1gdl.jpg' -> 4.1
    """
    # Remove file extensions (including HEIC)
    name = filename.lower()
    name = re.sub(r'\.(jpg|jpeg|png|heic)$', '', name)
    
    # Pattern 1: Direct HgB patterns - HgB_X.Xgdl
    match = re.search(r'hgb[_\s]*(\d+\.?\d*)g?dl', name)
    if match:
        return float(match.group(1))
    
    # Pattern 2: Random with HgB - Random_HgB_X.Xgdl  
    match = re.search(r'random[_\s]*hgb[_\s]*(\d+\.?\d*)g?dl', name)
    if match:
        return float(match.group(1))
    
    # Pattern 3: Random with direct number - Random_X.Xgdl
    match = re.search(r'random[_\s]*(\d+\.?\d*)g?dl', name)
    if match:
        return float(match.group(1))
    
    # Pattern 4: Any number followed by gdl (fallback)
    match = re.search(r'(\d+\.?\d*)g?dl', name)
    if match:
        return float(match.group(1))
    
    return None

def create_labels_csv(image_folder, output_path='labels.csv'):
    """
    Create labels.csv with image_id and hgb columns.
    """
    image_folder = Path(image_folder)
    
    labels_data = []
    
    # Find all image files (including HEIC)
    image_extensions = ['.jpg', '.jpeg', '.png', '.heic', '.JPG', '.JPEG', '.PNG', '.HEIC']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_folder.glob(f'*{ext}')))
    
    print(f"Found {len(image_files)} image files")
    
    for img_path in sorted(image_files):
        # Extract image ID (filename without extension)
        image_id = img_path.stem
        
        # Extract HgB value
        hgb = extract_hgb_from_filename(img_path.name)
        
        if hgb is not None:
            labels_data.append({
                'image_id': image_id,
                'hgb': hgb
            })
            print(f"✓ {img_path.name} -> HgB: {hgb:.1f} g/dL")
        else:
            print(f"✗ {img_path.name} -> Could not extract HgB value")
    
    # Create DataFrame and save
    df = pd.DataFrame(labels_data)
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Created {output_path}")
    print(f"Total images with labels: {len(df)}")
    print(f"HgB range: {df['hgb'].min():.1f} - {df['hgb'].max():.1f} g/dL")
    print(f"{'='*60}\n")
    
    return df

def extract_exif_metadata(image_path):
    """
    Extract EXIF metadata from image following competition convention.
    Creates metadata matching: image_id, device_id, device_brand, device_model, 
    iso_bucket, exposure_bucket, wb_bucket, ambient_light, distance_band, 
    skin_tone_proxy, age_band, gender
    """
    metadata = {
        'image_id': image_path.stem
    }
    
    try:
        img = Image.open(image_path)
        
        # Handle different EXIF extraction methods, especially for HEIC
        exif_data = None
        
        # Method 1: Try modern getexif() first (better for HEIC)
        try:
            exif_dict = img.getexif()
            if exif_dict:
                exif_data = dict(exif_dict)
                print(f"Found {len(exif_data)} EXIF tags using getexif()")
        except:
            pass
            
        # Method 2: Try legacy _getexif() as fallback
        if not exif_data:
            try:
                legacy_exif = img._getexif()
                if legacy_exif:
                    exif_data = legacy_exif
                    print(f"Found {len(exif_data)} EXIF tags using _getexif()")
            except:
                pass
        
        # Method 3: Try to get all available info including lens data
        try:
            # Get additional metadata that might contain lens info
            if hasattr(img, 'tag'):
                additional_data = img.tag
                if additional_data and not exif_data:
                    exif_data = additional_data
                    print(f"Found metadata using img.tag")
        except:
            pass
        
        # Initialize with default values
        device_brand = "Unknown"
        device_model = "Unknown"
        iso_value = None
        exposure_time = None
        white_balance = None
        
        if exif_data:
            for tag_id, value in exif_data.items():
                tag = TAGS.get(tag_id, tag_id)
                if tag == 'Make':
                    device_brand = str(value).strip()
                elif tag == 'Model':
                    device_model = str(value).strip()
                elif tag == 'ISOSpeedRatings' or tag == 'ISO':
                    iso_value = value
                elif tag == 'ExposureTime':
                    exposure_time = value
                elif tag == 'WhiteBalance':
                    white_balance = value
                # Look for HostComputer tag which often contains detailed device info
                elif tag == 'HostComputer':
                    host_info = str(value).strip()
                    if 'iphone' in host_info.lower():
                        device_brand = "Apple"
                        device_model = host_info  # Use the full HostComputer string
        
        # Try additional methods for HEIC files to get device info
        if device_brand == "Unknown" and device_model == "Unknown" and str(image_path).lower().endswith('.heic'):
            try:
                # Try to get more detailed metadata for HEIC files                
                # Get all possible metadata
                info = img.getexif()
                if info:
                    # Look through all tags more thoroughly
                    for tag_id in info:
                        tag = TAGS.get(tag_id, tag_id)
                        value = info[tag_id]
                        
                        # Check various fields that might contain device info
                        if any(keyword in str(tag).lower() for keyword in ['make', 'brand', 'manufacturer']):
                            if 'apple' in str(value).lower():
                                device_brand = "Apple"
                        elif any(keyword in str(tag).lower() for keyword in ['model', 'camera', 'lens']):
                            if 'iphone' in str(value).lower():
                                device_brand = "Apple"
                                # Try to extract specific iPhone model
                                value_str = str(value).lower()
                                if '15 pro max' in value_str:
                                    device_model = "iPhone 15 Pro Max"
                                elif '15 pro' in value_str:
                                    device_model = "iPhone 15 Pro"  
                                elif '15' in value_str:
                                    device_model = "iPhone 15"
                                elif '14 pro max' in value_str:
                                    device_model = "iPhone 14 Pro Max"
                                elif '14 pro' in value_str:
                                    device_model = "iPhone 14 Pro"
                                elif '14' in value_str:
                                    device_model = "iPhone 14"
                                else:
                                    device_model = "iPhone"
                                    
                print(f"After HEIC analysis: {device_brand} {device_model}")
            except Exception as e:
                print(f"HEIC metadata extraction failed: {e}")
        
        # If still no EXIF data found, try to infer from filename or use reasonable defaults
        if device_brand == "Unknown" and device_model == "Unknown":
            filename = image_path.name.lower()
            
            # Try to infer device from common patterns
            if any(x in filename for x in ['iphone', 'apple']):
                device_brand = "Apple"
                device_model = "iPhone"
            elif any(x in filename for x in ['samsung', 'galaxy']):
                device_brand = "Samsung" 
                device_model = "Galaxy"
            elif any(x in filename for x in ['pixel', 'google']):
                device_brand = "Google"
                device_model = "Pixel"
            else:
                # For medical imaging, assume mobile device
                device_brand = "Mobile"
                device_model = "Camera"
                
        # Set required metadata fields
        metadata['device_id'] = f"{device_brand}_{device_model}".replace(" ", "_")
        metadata['device_brand'] = device_brand
        metadata['device_model'] = device_model
        
        # Create ISO bucket (with reasonable defaults for mobile photos)
        if iso_value:
            if iso_value <= 100:
                metadata['iso_bucket'] = "low"
            elif iso_value <= 400:
                metadata['iso_bucket'] = "medium"
            else:
                metadata['iso_bucket'] = "high"
        else:
            # Most mobile photos are taken in medium lighting conditions
            metadata['iso_bucket'] = "medium"
        
        # Create exposure bucket (with reasonable defaults)
        if exposure_time:
            if isinstance(exposure_time, str) and '/' in exposure_time:
                # Handle fractional exposure times like "1/60"
                parts = exposure_time.split('/')
                exp_val = float(parts[0]) / float(parts[1])
            else:
                exp_val = float(exposure_time)
            
            if exp_val >= 1/30:
                metadata['exposure_bucket'] = "slow"
            elif exp_val >= 1/125:
                metadata['exposure_bucket'] = "medium"
            else:
                metadata['exposure_bucket'] = "fast"
        else:
            # Most mobile photos use automatic exposure
            metadata['exposure_bucket'] = "medium"
        
        # Create white balance bucket (with reasonable defaults)
        if white_balance:
            if white_balance == 0:
                metadata['wb_bucket'] = "auto"
            elif white_balance == 1:
                metadata['wb_bucket'] = "daylight"
            else:
                metadata['wb_bucket'] = "manual"
        else:
            # Most mobile phones use auto white balance
            metadata['wb_bucket'] = "auto"
        
        # Try to infer some fields from filename patterns
        filename_lower = image_path.name.lower()
        
        # Ambient light inference (medical imaging context)
        metadata['ambient_light'] = "indoor"  # Medical imaging typically indoor
        
        # Distance band inference (lip close-ups)
        metadata['distance_band'] = "close"  # Lip images are typically close-up
        
        # Skin tone proxy - try to infer from filename if available
        if 'chinese' in filename_lower or 'asian' in filename_lower:
            metadata['skin_tone_proxy'] = "III-IV"  # Typical range for East Asian
        elif 'middleeastern' in filename_lower or 'middle' in filename_lower:
            metadata['skin_tone_proxy'] = "IV-V"  # Typical range for Middle Eastern
        elif 'african' in filename_lower or 'dark' in filename_lower:
            metadata['skin_tone_proxy'] = "V-VI"  # Darker skin tones
        elif 'european' in filename_lower or 'caucasian' in filename_lower:
            metadata['skin_tone_proxy'] = "I-III"  # Lighter skin tones
        else:
            metadata['skin_tone_proxy'] = "unknown"  # Cannot determine
        
        # Age and gender remain unknown without manual annotation
        metadata['age_band'] = "unknown"  # Need manual annotation
        metadata['gender'] = "unknown"  # Need manual annotation
        
    except Exception as e:
        print(f"Warning: Could not extract EXIF from {image_path.name}: {e}")
        # Set default values on error
        metadata.update({
            'device_id': "unknown",
            'device_brand': "unknown", 
            'device_model': "unknown",
            'iso_bucket': "unknown",
            'exposure_bucket': "unknown", 
            'wb_bucket': "unknown",
            'ambient_light': "unknown",
            'distance_band': "unknown",
            'skin_tone_proxy': "unknown", 
            'age_band': "unknown",
            'gender': "unknown"
        })
    
    return metadata

def extract_color_metadata(image_path):
    """
    Extract color-based metadata from image.
    Handles HEIC files by converting through PIL first.
    """
    try:
        # Handle HEIC files
        if str(image_path).lower().endswith('.heic'):
            # Use PIL to read HEIC, then convert to numpy array
            pil_img = Image.open(image_path).convert('RGB')
            img_rgb = np.array(pil_img)
            # Convert to BGR for OpenCV functions
            img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        else:
            # Read image normally
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"Could not read image: {image_path}")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # RGB statistics
        mean_r = np.mean(img_rgb[:, :, 0])
        mean_g = np.mean(img_rgb[:, :, 1])
        mean_b = np.mean(img_rgb[:, :, 2])
        
        std_r = np.std(img_rgb[:, :, 0])
        std_g = np.std(img_rgb[:, :, 1])
        std_b = np.std(img_rgb[:, :, 2])
        
        # HSV statistics
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(img_hsv[:, :, 0])
        mean_s = np.mean(img_hsv[:, :, 1])
        mean_v = np.mean(img_hsv[:, :, 2])
        
        # LAB color space (better for color perception)
        img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        mean_l = np.mean(img_lab[:, :, 0])
        mean_a = np.mean(img_lab[:, :, 1])
        mean_b_lab = np.mean(img_lab[:, :, 2])
        
        # Color ratios (important for hemoglobin)
        rg_ratio = mean_r / (mean_g + 1e-6)
        rb_ratio = mean_r / (mean_b + 1e-6)
        
        # Brightness estimate
        brightness = 0.299 * mean_r + 0.587 * mean_g + 0.114 * mean_b
        
        return {
            'mean_r': round(mean_r, 2),
            'mean_g': round(mean_g, 2),
            'mean_b': round(mean_b, 2),
            'std_r': round(std_r, 2),
            'std_g': round(std_g, 2),
            'std_b': round(std_b, 2),
            'mean_h': round(mean_h, 2),
            'mean_s': round(mean_s, 2),
            'mean_v': round(mean_v, 2),
            'mean_l': round(mean_l, 2),
            'mean_a': round(mean_a, 2),
            'mean_b_lab': round(mean_b_lab, 2),
            'rg_ratio': round(rg_ratio, 4),
            'rb_ratio': round(rb_ratio, 4),
            'brightness': round(brightness, 2)
        }
    except Exception as e:
        print(f"Warning: Could not extract color metadata from {image_path.name}: {e}")
        return {}

def create_metadata_csv(image_folder, output_path='meta.csv'):
    """
    Create meta.csv following competition convention:
    image_id, device_id, device_brand, device_model, iso_bucket, exposure_bucket, 
    wb_bucket, ambient_light, distance_band, skin_tone_proxy, age_band, gender
    """
    image_folder = Path(image_folder)
    
    metadata_list = []
    
    # Find all image files (including HEIC)
    image_extensions = ['.jpg', '.jpeg', '.png', '.heic', '.JPG', '.JPEG', '.PNG', '.HEIC']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_folder.glob(f'*{ext}')))
    
    print(f"Extracting metadata from {len(image_files)} images...\n")
    
    for img_path in sorted(image_files):
        print(f"Processing: {img_path.name}")
        
        # Get EXIF metadata in competition format
        metadata = extract_exif_metadata(img_path)
        metadata_list.append(metadata)
    
    # Create DataFrame with specific column order
    df = pd.DataFrame(metadata_list)
    
    # Ensure columns are in the expected order
    expected_columns = [
        'image_id', 'device_id', 'device_brand', 'device_model', 
        'iso_bucket', 'exposure_bucket', 'wb_bucket', 'ambient_light', 
        'distance_band', 'skin_tone_proxy', 'age_band', 'gender'
    ]
    
    # Reorder columns and fill missing ones
    for col in expected_columns:
        if col not in df.columns:
            df[col] = "unknown"
    
    df = df[expected_columns]
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Created {output_path} (following competition convention)")
    print(f"Total images: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"{'='*60}\n")
    
    # Show sample
    print("Sample metadata (first 3 rows):")
    print(df.head(3).to_string())
    
    return df

def create_combined_dataset(image_folder, labels_csv='labels.csv', 
                           metadata_csv='meta.csv', 
                           output_csv='combined_data.csv'):
    """
    Combine labels and metadata into a single CSV.
    """
    labels_df = pd.read_csv(labels_csv)
    metadata_df = pd.read_csv(metadata_csv)
    
    # Merge on image_id
    combined_df = labels_df.merge(metadata_df, on='image_id', how='inner')
    combined_df.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"Created {output_csv}")
    print(f"Total samples: {len(combined_df)}")
    print(f"Columns: {list(combined_df.columns)}")
    print(f"{'='*60}\n")
    
    return combined_df

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("HEMOGLOBIN DATA EXTRACTION")
    print("="*60)
    print()
    
    # SET YOUR IMAGE FOLDER PATH HERE
    IMAGE_FOLDER = r"C:\Users\kylie\Documents\SMU Y3S1\DSA Case Competition 2025\1. Randomised Files"
    
    # CREATE DATA FOLDER FOR OUTPUT
    DATA_FOLDER = Path(__file__).parent.parent / "data"
    DATA_FOLDER.mkdir(exist_ok=True)
    print(f"Output folder: {DATA_FOLDER}")
    
    # Check if folder exists
    if not os.path.exists(IMAGE_FOLDER):
        print(f"ERROR: Folder not found: {IMAGE_FOLDER}")
        print("Please update IMAGE_FOLDER path in the script.")
        exit(1)
    
    # STEP 1: Extract labels
    print("\n[STEP 1] Extracting HgB labels from filenames...")
    print("-" * 60)
    labels_df = create_labels_csv(IMAGE_FOLDER, output_path=DATA_FOLDER / 'labels.csv')
    
    # STEP 2: Extract metadata
    print("\n[STEP 2] Extracting image metadata...")
    print("-" * 60)
    metadata_df = create_metadata_csv(
        IMAGE_FOLDER, 
        output_path=DATA_FOLDER / 'meta.csv'
    )
    
    # STEP 3: Create combined dataset
    print("\n[STEP 3] Creating combined dataset...")
    print("-" * 60)
    combined_df = create_combined_dataset(
        IMAGE_FOLDER,
        labels_csv=DATA_FOLDER / 'labels.csv',
        metadata_csv=DATA_FOLDER / 'meta.csv',
        output_csv=DATA_FOLDER / 'combined_data.csv'
    )
    
    # Summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE!")
    print("="*60)
    print(f"\nFiles created in {DATA_FOLDER}:")
    print("  1. labels.csv - HgB labels for each image (image_id, hgb)")
    print("  2. meta.csv - Image metadata following competition convention")
    print("  3. combined_data.csv - Combined labels + metadata")
    print("\nYou can now use these CSV files for your analysis!")
    print("="*60)