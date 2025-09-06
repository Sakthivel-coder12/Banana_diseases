import os
import glob
import re

orig_dir = "dataset/train/disease"
proc_dir = "preprocessed_custom_fixed/train/disease"

# Check what we actually have
orig_files = glob.glob(os.path.join(orig_dir, "*"))
proc_files = glob.glob(os.path.join(proc_dir, "*.png"))

print(f"Original files: {len(orig_files)}")
print(f"Processed files: {len(proc_files)}")

# Get all processed IDs
proc_ids = []
for f in proc_files:
    match = re.search(r'aug_(\d+)_', os.path.basename(f))
    if match:
        proc_ids.append(int(match.group(1)))

print(f"Processed IDs range: {min(proc_ids)} to {max(proc_ids)}")
print(f"Processed IDs sample: {sorted(proc_ids)[:20]}")