import os
import shutil
import random

# Paths
original_dataset = r"C://Users//sakth//OneDrive//Pictures//Documents//deep_learning_dataset//extract//dataset1"  # your current dataset folder
output_base = "dataset"           # new dataset folder

# Classes (Healthy, Diseased)
classes = ["Healthy", "disease"]

# Train/Val/Test split ratios
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Create directories
for split in ["train", "val", "test"]:
    for cls in classes:
        os.makedirs(os.path.join(output_base, split, cls), exist_ok=True)

# Split and copy files
for cls in classes:
    folder = os.path.join(original_dataset, cls)
    images = os.listdir(folder)
    random.shuffle(images)

    total = len(images)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)

    train_files = images[:train_end]
    val_files = images[train_end:val_end]
    test_files = images[val_end:]

    # Copy files
    for f in train_files:
        shutil.copy(os.path.join(folder, f), os.path.join(output_base, "train", cls, f))
    for f in val_files:
        shutil.copy(os.path.join(folder, f), os.path.join(output_base, "val", cls, f))
    for f in test_files:
        shutil.copy(os.path.join(folder, f), os.path.join(output_base, "test", cls, f))

print("✅ Dataset split completed!")
