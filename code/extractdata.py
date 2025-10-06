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

# CRITICAL: Enable HEIC support BEFORE any image operations
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    print("✓ HEIC support enabled\n")
except ImportError:
    print("ERROR: pillow-heif not installed. HEIC files will fail.")
    print("Install with: pip install pillow-heif")
    print()
    exit(1)  # Exit if HEIC support isn't available

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
    
    # Find all image files (including HEIC) - use case-insensitive matching
    image_files = []
    for pattern in ['*.jpg', '*.jpeg', '*.png', '*.heic']:
        # Add both lowercase and uppercase matches
        image_files.extend(list(image_folder.glob(pattern)))
        image_files.extend(list(image_folder.glob(pattern.upper())))
    
    # Remove duplicates that might occur from case variations
    image_files = list(set(image_files))
    
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
    if len(df) > 0:
        print(f"HgB range: {df['hgb'].min():.1f} - {df['hgb'].max():.1f} g/dL")
    print(f"{'='*60}\n")
    
    return df

def extract_exif_metadata(image_path):
    """
    Extract EXIF metadata from image with robust HEIC support.
    """
    metadata = {
        'image_id': image_path.stem
    }
    
    # Initialize all variables at the start
    camera_type = "unknown"
    device_brand = "Unknown"
    device_model = "Unknown"
    iso_value = None
    exposure_time = None
    white_balance = None
    lens_make = None
    lens_model = None
    focal_length = None
    
    try:
        img = Image.open(image_path)
        
        # Get EXIF data
        exif_data = {}
        
        try:
            exif_dict = img.getexif()
            if exif_dict:
                # Get main EXIF data
                for tag_id, value in exif_dict.items():
                    decoded_tag = TAGS.get(tag_id, tag_id)
                    exif_data[decoded_tag] = value
                
                # CRITICAL: Get IFD data which contains lens/camera info for iPhones
                try:
                    # EXIF IFD (0x8769) contains detailed camera settings
                    exif_ifd = exif_dict.get_ifd(0x8769)
                    for tag_id, value in exif_ifd.items():
                        decoded_tag = TAGS.get(tag_id, tag_id)
                        if decoded_tag not in exif_data:
                            exif_data[decoded_tag] = value
                except:
                    pass
                
                print(f"  Extracted {len(exif_data)} EXIF fields")
        except Exception as e:
            print(f"  EXIF extraction error: {e}")
        
        # Extract device info
        for tag, value in exif_data.items():
            tag_str = str(tag)
            value_str = str(value).strip()
            
            # Camera maker
            if tag in ['Make', 'make']:
                device_brand = value_str
            
            # Camera model - this is KEY for iPhones
            elif tag in ['Model', 'model']:
                device_model = value_str
            
            # Lens make
            elif tag in ['LensMake', 'lens_make']:
                lens_make = value_str
            
            # Lens model - CRITICAL for iPhone camera type detection
            elif tag in ['LensModel', 'lens_model', 'LensInfo']:
                lens_model = value_str
                print(f"  Found lens_model: {lens_model}")
            
            # Camera settings
            elif tag in ['ISOSpeedRatings', 'ISO', 'PhotographicSensitivity']:
                iso_value = value
            elif tag in ['ExposureTime', 'exposure_time']:
                exposure_time = value
            elif tag in ['WhiteBalance', 'white_balance']:
                white_balance = value
            elif tag in ['FocalLength', 'focal_length']:
                focal_length = value
        
        # Detect camera type from lens model (most reliable for iPhones)
        if lens_model:
            lens_lower = lens_model.lower()
            
            # iPhone patterns: "iPhone 15 Pro Max front TrueDepth camera"
            if 'front' in lens_lower or 'truedepth' in lens_lower or 'facetime' in lens_lower:
                camera_type = "front"
            elif 'back' in lens_lower or 'rear' in lens_lower:
                camera_type = "back"
            # Check for specific lens types
            elif any(x in lens_lower for x in ['main', 'wide', 'telephoto', 'ultra']):
                camera_type = "back"
        
        # Fallback: Use focal length to detect camera type
        if camera_type == "unknown" and focal_length:
            try:
                if isinstance(focal_length, tuple):
                    fl_value = focal_length[0] / focal_length[1]
                else:
                    fl_value = float(focal_length)
                
                # iPhone front cameras: ~2.71mm, back cameras: 1.5-9mm
                # 3mm is a good threshold
                if fl_value < 3.5:
                    camera_type = "front"
                else:
                    camera_type = "back"
                    
                print(f"  Focal length: {fl_value}mm -> camera_type: {camera_type}")
            except:
                pass
        
        # Clean up device brand/model
        if device_brand and 'apple' in device_brand.lower():
            device_brand = "Apple"
        
        if device_model and 'iphone' in device_model.lower():
            # Extract specific iPhone model
            model_lower = device_model.lower()
            
            if '15 pro max' in model_lower:
                device_model = "iPhone 15 Pro Max"
            elif '15 pro' in model_lower:
                device_model = "iPhone 15 Pro"
            elif '15 plus' in model_lower:
                device_model = "iPhone 15 Plus"
            elif '15' in model_lower:
                device_model = "iPhone 15"
            elif '14 pro max' in model_lower:
                device_model = "iPhone 14 Pro Max"
            elif '14 pro' in model_lower:
                device_model = "iPhone 14 Pro"
            elif '14 plus' in model_lower:
                device_model = "iPhone 14 Plus"
            elif '14' in model_lower:
                device_model = "iPhone 14"
            elif '13 pro max' in model_lower:
                device_model = "iPhone 13 Pro Max"
            elif '13 pro' in model_lower:
                device_model = "iPhone 13 Pro"
            elif '13' in model_lower:
                device_model = "iPhone 13"
            # Keep original if specific model not detected
        
        # If still unknown after all extraction attempts
        if device_brand == "Unknown" or device_model == "Unknown":
            print(f"  WARNING: Could not extract device info from EXIF")
            print(f"  Current: {device_brand} {device_model}")
            
            # Last resort: check filename
            filename_lower = image_path.name.lower()
            if 'iphone' in filename_lower:
                device_brand = "Apple"
                device_model = "iPhone (unknown model)"
        
        # Build device_id
        device_id_parts = [device_brand, device_model]
        if camera_type != "unknown":
            device_id_parts.append(camera_type)
        
        metadata['device_id'] = "_".join(device_id_parts).replace(" ", "_")
        metadata['device_brand'] = device_brand
        metadata['device_model'] = device_model
        metadata['camera_type'] = camera_type
        
        # Lens info
        if lens_make or lens_model:
            metadata['lens_info'] = f"{lens_make or ''} {lens_model or ''}".strip()
        else:
            metadata['lens_info'] = "unknown"
        
        # Focal length
        if focal_length:
            try:
                if isinstance(focal_length, tuple):
                    fl_value = focal_length[0] / focal_length[1]
                else:
                    fl_value = float(focal_length)
                metadata['focal_length_mm'] = round(fl_value, 2)
            except:
                metadata['focal_length_mm'] = "unknown"
        else:
            metadata['focal_length_mm'] = "unknown"
        
        # ISO bucket
        if iso_value:
            try:
                iso_val = int(iso_value) if not isinstance(iso_value, (list, tuple)) else int(iso_value[0])
                if iso_val <= 100:
                    metadata['iso_bucket'] = "low"
                elif iso_val <= 400:
                    metadata['iso_bucket'] = "medium"
                else:
                    metadata['iso_bucket'] = "high"
            except:
                metadata['iso_bucket'] = "medium"
        else:
            metadata['iso_bucket'] = "medium"
        
        # Exposure bucket
        if exposure_time:
            try:
                if isinstance(exposure_time, tuple):
                    exp_val = exposure_time[0] / exposure_time[1]
                elif isinstance(exposure_time, str) and '/' in str(exposure_time):
                    parts = str(exposure_time).split('/')
                    exp_val = float(parts[0]) / float(parts[1])
                else:
                    exp_val = float(exposure_time)
                
                if exp_val >= 1/30:
                    metadata['exposure_bucket'] = "slow"
                elif exp_val >= 1/125:
                    metadata['exposure_bucket'] = "medium"
                else:
                    metadata['exposure_bucket'] = "fast"
            except:
                metadata['exposure_bucket'] = "medium"
        else:
            metadata['exposure_bucket'] = "medium"
        
        # White balance
        if white_balance is not None:
            try:
                wb_val = int(white_balance)
                if wb_val == 0:
                    metadata['wb_bucket'] = "auto"
                elif wb_val == 1:
                    metadata['wb_bucket'] = "manual"
                else:
                    metadata['wb_bucket'] = "other"
            except:
                metadata['wb_bucket'] = "auto"
        else:
            metadata['wb_bucket'] = "auto"
        
        # Infer contextual metadata
        filename_lower = image_path.name.lower()
        
        metadata['ambient_light'] = "indoor"
        metadata['distance_band'] = "close"
        
        # Skin tone inference from filename
        if 'chinese' in filename_lower or 'asian' in filename_lower:
            metadata['skin_tone_proxy'] = "III-IV"
        elif 'middleeastern' in filename_lower or 'middle' in filename_lower:
            metadata['skin_tone_proxy'] = "IV-V"
        elif 'african' in filename_lower:
            metadata['skin_tone_proxy'] = "V-VI"
        elif 'european' in filename_lower or 'caucasian' in filename_lower:
            metadata['skin_tone_proxy'] = "I-III"
        else:
            metadata['skin_tone_proxy'] = "unknown"
        
        metadata['age_band'] = "unknown"
        metadata['gender'] = "unknown"
        
    except Exception as e:
        print(f"  ERROR extracting metadata: {e}")
        import traceback
        traceback.print_exc()
        
        metadata.update({
            'device_id': "unknown",
            'device_brand': "unknown",
            'device_model': "unknown",
            'camera_type': "unknown",
            'lens_info': "unknown",
            'iso_bucket': "medium",
            'exposure_bucket': "medium",
            'wb_bucket': "auto",
            'focal_length_mm': "unknown",
            'ambient_light': "indoor",
            'distance_band': "close",
            'skin_tone_proxy': "unknown",
            'age_band': "unknown",
            'gender': "unknown"
        })
    
    return metadata

def create_metadata_csv(image_folder, output_path='meta.csv'):
    """
    Create meta.csv with enhanced metadata extraction.
    """
    image_folder = Path(image_folder)
    
    metadata_list = []
    
    # Find all image files (including HEIC) - use case-insensitive matching
    image_files = []
    for pattern in ['*.jpg', '*.jpeg', '*.png', '*.heic']:
        # Add both lowercase and uppercase matches
        image_files.extend(list(image_folder.glob(pattern)))
        image_files.extend(list(image_folder.glob(pattern.upper())))
    
    # Remove duplicates that might occur from case variations
    image_files = list(set(image_files))
    
    print(f"Extracting metadata from {len(image_files)} images...\n")
    
    for img_path in sorted(image_files):
        print(f"Processing: {img_path.name}")
        
        # Get EXIF metadata
        metadata = extract_exif_metadata(img_path)
        
        # Show extracted info
        if metadata['device_brand'] != "Unknown":
            print(f"  Device: {metadata['device_brand']} {metadata['device_model']}")
        if metadata['camera_type'] != "unknown":
            print(f"  Camera: {metadata['camera_type']}")
        
        metadata_list.append(metadata)
        print()
    
    # Create DataFrame
    df = pd.DataFrame(metadata_list)
    
    # Define column order
    expected_columns = [
        'image_id', 'device_id', 'device_brand', 'device_model', 
        'camera_type', 'lens_info', 'focal_length_mm',
        'iso_bucket', 'exposure_bucket', 'wb_bucket', 
        'ambient_light', 'distance_band', 'skin_tone_proxy', 
        'age_band', 'gender'
    ]
    
    # Reorder columns and fill missing ones
    for col in expected_columns:
        if col not in df.columns:
            df[col] = "unknown"
    
    df = df[expected_columns]
    df.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(f"Created {output_path}")
    print(f"Total images: {len(df)}")
    print(f"{'='*60}\n")
    
    # Show sample
    if len(df) > 0:
        print("Sample metadata:")
        print(df[['image_id', 'device_brand', 'device_model', 'camera_type']].head(3).to_string())
    
    return df

def create_combined_dataset(image_folder, labels_csv, 
                           metadata_csv, 
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
    
    # Define paths - CHANGE THIS TO YOUR IMAGE FOLDER
    IMAGE_FOLDER = Path(r"../data/images")
    
    # Create data folder structure
    DATA_FOLDER = Path(__file__).parent.parent / "data"
    
    # Check and create data folder if it doesn't exist
    if not DATA_FOLDER.exists():
        print(f"Creating data folder: {DATA_FOLDER}")
        DATA_FOLDER.mkdir(parents=True, exist_ok=True)
        print("✓ Data folder created\n")
    else:
        print(f"✓ Data folder exists: {DATA_FOLDER}\n")
    
    # Check if images folder exists
    if not IMAGE_FOLDER.exists():
        print(f"ERROR: Images folder not found: {IMAGE_FOLDER}")
        print("Please update IMAGE_FOLDER path in the script.")
        exit(1)
    
    print(f"Reading images from: {IMAGE_FOLDER}")
    print(f"Output folder: {DATA_FOLDER}\n")
    
    # Define output paths in data folder
    labels_path = DATA_FOLDER / "labels.csv"
    meta_path = DATA_FOLDER / "meta.csv"
    combined_path = DATA_FOLDER / "combined_data.csv"
    
    # STEP 1: Extract labels
    print("\n[STEP 1] Extracting HgB labels from filenames...")
    print("-" * 60)
    labels_df = create_labels_csv(IMAGE_FOLDER, output_path=labels_path)
    
    # STEP 2: Extract metadata
    print("\n[STEP 2] Extracting image metadata...")
    print("-" * 60)
    metadata_df = create_metadata_csv(IMAGE_FOLDER, output_path=meta_path)
    
    # STEP 3: Create combined dataset
    print("\n[STEP 3] Creating combined dataset...")
    print("-" * 60)
    combined_df = create_combined_dataset(
        IMAGE_FOLDER,
        labels_csv=labels_path,
        metadata_csv=meta_path,
        output_csv=combined_path
    )
    
    # Summary
    print("\n" + "="*60)
    print("EXTRACTION COMPLETE!")
    print("="*60)
    print(f"\nFiles created in {DATA_FOLDER}:")
    print(f"  1. {labels_path.name} - HgB labels for each image")
    print(f"  2. {meta_path.name} - Image metadata")
    print(f"  3. {combined_path.name} - Combined labels + metadata")
    print("\nYou can now use these CSV files for training!")
    print("="*60)