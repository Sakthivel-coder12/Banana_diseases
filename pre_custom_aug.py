import cv2
import numpy as np
import random
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# --- Config ---
IMG_SIZE = (224, 224)
AUG_PER_IMAGE = 5   # Number of augmented versions per image

# --- Custom Functions ---
def random_cutout(img, size=50, mode="blur"):
    """Hide a random patch with blur instead of black (optional)."""
    h, w, _ = img.shape
    if h > size and w > size:
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)

        if mode == "blur":
            patch = img[y:y+size, x:x+size, :].copy()
            patch = cv2.GaussianBlur(patch, (15, 15), 0)
        else:
            return img

        img[y:y+size, x:x+size, :] = patch
    return img

def custom_preprocessing(img):
    """Resize and keep natural colors, with optional blur cutout."""
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # ✅ fix color shift
    if random.random() < 0.3:  # 30% chance cutout
        img = random_cutout(img, mode="blur")
    return img.astype(np.float32)


# --- Data Augmentation ---
datagen = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
    # 🚫 brightness_range removed (keeps original colors)
)

# --- Save Augmented Images ---
def save_augmented_images(input_dir, output_dir, subset_name):
    generator = datagen.flow_from_directory(
        input_dir,
        target_size=IMG_SIZE,
        batch_size=1,
        class_mode=None,
        shuffle=False,
        save_to_dir=None
    )

    total = generator.samples
    print(f"➡ Processing {total} images from {subset_name}...")

    for i in range(total):
        img = generator[i][0]  # already preprocessed
        label = generator.filenames[i].split(os.sep)[0]
        save_folder = os.path.join(output_dir, subset_name, label)
        os.makedirs(save_folder, exist_ok=True)

        # Save original once
        orig_path = os.path.join(save_folder, f"orig_{i}.png")
        cv2.imwrite(orig_path, cv2.cvtColor(np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


        # Save augmented versions
        for j in range(AUG_PER_IMAGE):
            aug_iter = datagen.flow(np.expand_dims(img, 0), batch_size=1)
            aug_img = next(aug_iter)[0]
            aug_path = os.path.join(save_folder, f"aug_{i}_{j}.png")
            cv2.imwrite(aug_path, cv2.cvtColor(np.clip(aug_img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))


    print(f"✅ Done saving for {subset_name}")

# --- Run for train, val, test ---
save_augmented_images("dataset_custom/train", "preprocessed_custom_fixed", "train")
save_augmented_images("dataset_custom/val", "preprocessed_custom_fixed", "val")
save_augmented_images("dataset_custom/test", "preprocessed_custom_fixed", "test")
