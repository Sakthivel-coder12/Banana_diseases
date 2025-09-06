import os, glob, re
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim, peak_signal_noise_ratio as psnr
from skimage.measure import shannon_entropy
from difflib import get_close_matches


def extract_number(name):
    """Extract first number from filename, else return full name."""
    match = re.search(r"\d+", name)
    return match.group(0) if match else name.lower()


def resize_to_match(original, target):
    """Resize target image to match original dimensions"""
    if original.shape != target.shape:
        target = cv2.resize(target, (original.shape[1], original.shape[0]))
    return target


def compute_online_metrics(original, processed):
    # Resize processed image to match original dimensions
    processed = resize_to_match(original, processed)
    
    orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)

    # Calculate SSIM with proper data range
    ssim_val = ssim(orig_gray, proc_gray, data_range=255)
    
    # Calculate PSNR
    psnr_val = psnr(orig_gray, proc_gray, data_range=255)
    
    # Calculate entropy
    entropy_val = shannon_entropy(proc_gray)

    # Calculate foreground-background contrast
    _, thresh = cv2.threshold(proc_gray, 0, 255, cv2.THRESH_OTSU)
    fg_mean = np.mean(proc_gray[thresh == 255]) if np.any(thresh == 255) else 0
    bg_mean = np.mean(proc_gray[thresh == 0]) if np.any(thresh == 0) else 0
    fg_bg_contrast = abs(fg_mean - bg_mean)

    return {"SSIM": ssim_val, "PSNR": psnr_val, "Entropy": entropy_val, "FG-BG Contrast": fg_bg_contrast}


# --- CONFIG ---
orig_dir = "dataset/train/disease"
proc_dir = "preprocessed_online_1/train/disease"

metrics = {"SSIM": [], "PSNR": [], "Entropy": [], "FG-BG Contrast": []}

# collect originals
exts = ["*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"]
orig_files, proc_files = [], []
for e in exts:
    orig_files.extend(glob.glob(os.path.join(orig_dir, e)))
    proc_files.extend(glob.glob(os.path.join(proc_dir, e)))

print("Found original files:", len(orig_files))
print("Found processed files:", len(proc_files))

# create lookup for processed files (by number OR basename)
proc_lookup = {}
for f in proc_files:
    base = os.path.splitext(os.path.basename(f))[0]
    num = extract_number(base)
    proc_lookup[num] = f

matched, missing, errors = 0, 0, 0

# match originals to processed
for file in orig_files:
    base = os.path.splitext(os.path.basename(file))[0]
    num = extract_number(base)

    if num in proc_lookup:
        proc_file = proc_lookup[num]
    else:
        # fuzzy fallback
        close = get_close_matches(base.lower(), [os.path.splitext(os.path.basename(x))[0].lower() for x in proc_files], n=1, cutoff=0.6)
        if close:
            proc_file = [x for x in proc_files if os.path.splitext(os.path.basename(x))[0].lower() == close[0]][0]
        else:
            print("⚠️ Missing pair for:", base)
            missing += 1
            continue

    orig = cv2.imread(file)
    proc = cv2.imread(proc_file)

    if orig is None:
        print("❌ Could not read original:", base)
        errors += 1
        continue
        
    if proc is None:
        print("❌ Could not read processed:", base)
        errors += 1
        continue

    try:
        matched += 1
        m = compute_online_metrics(orig, proc)
        for k in metrics:
            metrics[k].append(m[k])
    except Exception as e:
        print(f"❌ Error processing {base}: {str(e)}")
        errors += 1
        continue

# --- RESULTS ---
print(f"\n✅ Matched pairs: {matched}")
print(f"❌ Missing pairs: {missing}")
print(f"⚠️  Error pairs: {errors}\n")

for k in metrics:
    if metrics[k]:
        print(f"Average {k}: {np.mean(metrics[k]):.4f} ± {np.std(metrics[k]):.4f}")
    else:
        print(f"{k}: No data")