import cv2
import numpy as np
import random
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224,224)
BATCH_SIZE = 32

# --- Custom Functions ---
def apply_clahe(img):
    # Ensure uint8 (CLAHE requires 8-bit)
    img = np.clip(img, 0, 255).astype(np.uint8)

    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return final

def normalize_img(img):
    return img.astype(np.float32) / 255.0

def random_cutout(img, size=50):
    h, w, _ = img.shape
    if h > size and w > size:  # safety check
        y = random.randint(0, h - size)
        x = random.randint(0, w - size)
        img[y:y+size, x:x+size, :] = 0
    return img

def custom_preprocessing(img):
    img = cv2.resize(img, IMG_SIZE)
    img = apply_clahe(img)          # improve lighting
    img = normalize_img(img)        # normalize AFTER CLAHE
    if random.random() < 0.3:       # 30% chance cutout
        img = random_cutout(img)
    return img

# --- Data Augmentation ---
train_datagen_custom = ImageDataGenerator(
    preprocessing_function=custom_preprocessing,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

val_test_datagen_custom = ImageDataGenerator(
    preprocessing_function=custom_preprocessing
)

# Example: Create generators
train_custom = train_datagen_custom.flow_from_directory(
    "dataset_custom/train",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=True
)

val_custom = val_test_datagen_custom.flow_from_directory(
    "dataset_custom/val",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False
)

# --- Save augmented images ---
save_dir = "preprocessed_custom"
os.makedirs(save_dir, exist_ok=True)

num_batches = 5   # how many batches to save (each batch = 32 images here)

batch_index = 0
for X_batch, y_batch in train_custom:
    for i in range(len(X_batch)):
        # X_batch is float32 normalized (0-1), convert back to uint8 for saving
        img = (X_batch[i] * 255).astype(np.uint8)
        label = int(y_batch[i])  # 0 or 1

        # save in subfolder per label
        label_dir = os.path.join(save_dir, str(label))
        os.makedirs(label_dir, exist_ok=True)

        filename = os.path.join(label_dir, f"train_aug_{batch_index}_{i}.jpg")
        cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

    batch_index += 1
    if batch_index >= num_batches:  # stop after saving N batches
        break

print(f"✅ Augmented images saved in '{save_dir}'")


# It assigns labels in alphabetical order:

# disease → class 0

# healthy → class 1