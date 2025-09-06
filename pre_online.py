import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = (224,224)

# Where to save augmented images
output_base = "preprocessed_dataset"

# Same classes as in your dataset
classes = ["Healthy", "disease"]

# Make output directories
for split in ["train", "val"]:
    for cls in classes:
        os.makedirs(os.path.join(output_base, split, cls), exist_ok=True)

# Preprocessing generator (augmentation)
datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode="nearest"
)

# Loop through train & val
for split in ["train", "val"]:
    for cls in classes:
        folder = os.path.join("dataset", split, cls)
        save_folder = os.path.join(output_base, split, cls)

        generator = datagen.flow_from_directory(
            "dataset/" + split,
            target_size=IMG_SIZE,
            batch_size=1,
            classes=[cls],            # only process one class
            class_mode=None,
            save_to_dir=save_folder,  # save augmented images
            save_prefix="aug",
            save_format="jpg"
        )

        # Save exactly as many augmented images as originals
        num_images = len(os.listdir(folder))
        for i in range(num_images):
            next(generator)

print("✅ Preprocessed images saved in 'preprocessed_dataset/'")
