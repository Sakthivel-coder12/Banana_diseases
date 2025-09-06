import os
import glob
import re
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.measure import shannon_entropy

def extract_original_id(augmented_name):
    """
    Extract the original image ID from augmented filename.
    Example: 'aug_228_0.png' -> '228'
    """
    match = re.search(r'aug_(\d+)_\d+', augmented_name)
    return match.group(1) if match else None

def create_augmentation_mapping(proc_files):
    """
    Create a mapping from original image ID to list of augmented versions
    """
    mapping = {}
    for proc_file in proc_files:
        filename = os.path.basename(proc_file)
        original_id = extract_original_id(filename)
        if original_id:
            if original_id not in mapping:
                mapping[original_id] = []
            mapping[original_id].append(proc_file)
    return mapping

def extract_original_number(original_name):
    """
    Extract the number from original filename.
    Example: 'disease228.png' -> '228'
    """
    match = re.search(r'(\d+)', original_name)
    return match.group(1) if match else None

def resize_to_match(original, target):
    """Resize target image to match original dimensions"""
    if original.shape != target.shape:
        target = cv2.resize(target, (original.shape[1], original.shape[0]))
    return target

def compute_custom_metrics(original, processed):
    # Resize processed to match original dimensions
    processed = resize_to_match(original, processed)
    
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # SSIM with fixed data range
    ssim_val = ssim(orig_gray, proc_gray, data_range=255)

    # PSNR
    psnr_val = psnr(orig_gray, proc_gray, data_range=255)

    # Sharpness (variance of Laplacian)
    sharpness = cv2.Laplacian(proc_gray, cv2.CV_64F).var()

    # Background std (using Otsu mask)
    _, thresh = cv2.threshold(proc_gray, 0, 255, cv2.THRESH_OTSU)
    bg_std = np.std(proc_gray[thresh == 0]) if np.any(thresh == 0) else 0

    # FG-BG contrast
    fg_mean = np.mean(proc_gray[thresh == 255]) if np.any(thresh == 255) else 0
    bg_mean = np.mean(proc_gray[thresh == 0]) if np.any(thresh == 0) else 0
    fg_bg_contrast = abs(fg_mean - bg_mean)

    # Black ratio
    black_ratio = np.sum(proc_gray == 0) / proc_gray.size

    # Entropy
    entropy_val = shannon_entropy(proc_gray)

    return {
        "SSIM": ssim_val,
        "PSNR": psnr_val,
        "Sharpness": sharpness,
        "Background Std": bg_std,
        "FG-BG Contrast": fg_bg_contrast,
        "Black Ratio": black_ratio,
        "Entropy": entropy_val
    }

# --- CONFIG ---
orig_dir = "dataset_custom/train/disease"
proc_dir = "preprocessed_custom_fixed_1/train/disease"

metrics = {
    "SSIM": [], "PSNR": [], "Sharpness": [], 
    "Background Std": [], "FG-BG Contrast": [], 
    "Black Ratio": [], "Entropy": []
}

# Collect files
exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
orig_files, proc_files = [], []
for e in exts:
    orig_files.extend(glob.glob(os.path.join(orig_dir, e)))
    proc_files.extend(glob.glob(os.path.join(proc_dir, e)))

print(f"Found {len(orig_files)} original files")
print(f"Found {len(proc_files)} processed files")

# Create mapping from original ID to augmented versions
augmentation_mapping = create_augmentation_mapping(proc_files)
print(f"Found {len(augmentation_mapping)} original images with augmentations")

# Process images
matched, missing, errors = 0, 0, 0

for orig_file in orig_files:
    orig_name = os.path.basename(orig_file)
    base_name = os.path.splitext(orig_name)[0]
    
    # Extract the number from original filename
    original_number = extract_original_number(orig_name)
    
    if not original_number:
        print(f"⚠️ Could not extract number from: {orig_name}")
        missing += 1
        continue
    
    # Find matching augmented files
    if original_number in augmentation_mapping:
        augmented_files = augmentation_mapping[original_number]
        
        # Use the first augmentation for metrics (you can modify this to use all)
        proc_file = augmented_files[0]
        
        try:
            orig_img = cv2.imread(orig_file)
            proc_img = cv2.imread(proc_file)
            
            if orig_img is None:
                print(f"❌ Could not read original: {base_name}")
                errors += 1
                continue
                
            if proc_img is None:
                print(f"❌ Could not read processed: {os.path.basename(proc_file)}")
                errors += 1
                continue
                
            # Compute metrics
            m = compute_custom_metrics(orig_img, proc_img)
            
            for key in metrics:
                metrics[key].append(m[key])
                
            matched += 1
            
            # Print progress every 100 files
            if matched % 100 == 0:
                print(f"Processed {matched} files...")
            
        except Exception as e:
            print(f"❌ Error processing {base_name}: {str(e)}")
            errors += 1
            continue
    else:
        print(f"⚠️ No augmentations found for: {base_name} (ID: {original_number})")
        missing += 1

# Print results
print(f"\nResults:")
print(f"✅ Matched pairs: {matched}")
print(f"⚠️  Missing pairs: {missing}")
print(f"❌ Error pairs: {errors}\n")

for metric_name, values in metrics.items():
    if values:
        print(f"{metric_name}: {np.mean(values):.4f} ± {np.std(values):.4f} (n={len(values)})")
    else:
        print(f"{metric_name}: No data")

# Optional: If you want to process ALL augmentations for each original image
print(f"\nAugmentation statistics:")
for orig_id, aug_files in list(augmentation_mapping.items())[:10]:  # Show first 10
    print(f"Original ID {orig_id}: {len(aug_files)} augmentations")