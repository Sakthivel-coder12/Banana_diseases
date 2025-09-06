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
    """Hide a random patch with blur instead of black."""
    h, w, _ = img.shape
    if h > size and w > size:
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)

        if mode == "blur":
            patch = img[y:y+size, x:x+size, :].copy()
            patch = cv2.GaussianBlur(patch, (15, 15), 0)
            img[y:y+size, x:x+size, :] = patch
    return img

def load_and_resize(img_path):
    """Load and resize image without other preprocessing"""
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB

# --- Data Augmentation (NO preprocessing) ---
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest"
    # 🚫 NO preprocessing_function here!
)

# --- Save Augmented Images ---
def save_augmented_images(input_dir, output_dir, subset_name):
    # Get all image paths
    image_paths = []
    labels = []
    
    for label in os.listdir(input_dir):
        label_dir = os.path.join(input_dir, label)
        if os.path.isdir(label_dir):
            for file in os.listdir(label_dir):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(label_dir, file))
                    labels.append(label)
    
    print(f"➡ Processing {len(image_paths)} images from {subset_name}...")
    
    for i, (img_path, label) in enumerate(zip(image_paths, labels)):
        # Load and resize original (no other preprocessing yet)
        orig_img = load_and_resize(img_path)
        if orig_img is None:
            continue
            
        save_folder = os.path.join(output_dir, subset_name, label)
        os.makedirs(save_folder, exist_ok=True)

        # Save TRUE original (just resized)
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        orig_path = os.path.join(save_folder, f"{base_name}_orig.png")
        cv2.imwrite(orig_path, cv2.cvtColor(orig_img, cv2.COLOR_RGB2BGR))

        # Apply cutout to create "preprocessed" version
        preprocessed_img = orig_img.copy()
        if random.random() < 0.3:  # 30% chance cutout
            preprocessed_img = random_cutout(preprocessed_img, mode="blur")
        
        # Save preprocessed version (without augmentation)
        preprocessed_path = os.path.join(save_folder, f"{base_name}_preprocessed.png")
        cv2.imwrite(preprocessed_path, cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2BGR))

        # Generate augmented versions from preprocessed image
        for j in range(AUG_PER_IMAGE):
            # Expand dimensions for Keras generator
            img_expanded = np.expand_dims(preprocessed_img, 0)
            
            # Get augmented version
            aug_iter = datagen.flow(img_expanded, batch_size=1)
            aug_img = next(aug_iter)[0]
            
            # Clip and convert back to uint8
            aug_img = np.clip(aug_img, 0, 255).astype(np.uint8)
            
            # Save augmented version
            aug_path = os.path.join(save_folder, f"{base_name}_aug_{j}.png")
            cv2.imwrite(aug_path, cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR))

        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1} images...")

    print(f"✅ Done saving for {subset_name}")

# --- Run for train, val, test ---
save_augmented_images("dataset_custom/train", "preprocessed_custom_fixed_1", "train")
save_augmented_images("dataset_custom/val", "preprocessed_custom_fixed_1", "val") 
save_augmented_images("dataset_custom/test", "preprocessed_custom_fixed_1", "test")