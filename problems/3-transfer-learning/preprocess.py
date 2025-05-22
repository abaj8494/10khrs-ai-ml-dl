import os
import shutil
from pathlib import Path

# Animal categories of interest
animals = [
    'bear', 'chimp', 'giraffe', 'gorilla', 'llama',
    'ostrich', 'porcupine', 'skunk', 'triceratops', 'zebra'
]

# Paths
base_dir = Path(__file__).parent
data_dir = base_dir / 'data'
output_dir = base_dir / 'caltech_10'
splits = ['train', 'valid', 'test']

# Ensure output folders exist
for split in splits:
    for animal in animals:
        (output_dir / split / animal).mkdir(parents=True, exist_ok=True)

# Go through each folder in data/
for folder in data_dir.iterdir():
    if not folder.is_dir():
        continue

    for animal in animals:
        if f".{animal}" in folder.name:  # match e.g. "009.bear"
            image_paths = sorted(folder.glob("*.jpg"))
            total = len(image_paths)

            if total < 70:
                print(f"Warning: Not enough images for '{animal}' in {folder.name}. Found {total}. Skipping...")
                continue

            # Split the images
            train_imgs = image_paths[:60]
            valid_imgs = image_paths[60:70]
            test_imgs = image_paths[70:]

            # Move images to respective folders
            for img in train_imgs:
                shutil.move(str(img), output_dir / 'train' / animal / img.name)
            for img in valid_imgs:
                shutil.move(str(img), output_dir / 'valid' / animal / img.name)
            for img in test_imgs:
                shutil.move(str(img), output_dir / 'test' / animal / img.name)

            print(f"Moved {len(train_imgs)} train, {len(valid_imgs)} valid, {len(test_imgs)} test images for '{animal}'.")
            break  # Once matched, no need to check other animals for this folder

print("All done.")
