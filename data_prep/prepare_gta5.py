#!/usr/bin/env python3
"""
Prepare GTA5 dataset for 8-class Semantic Segmentation.

Downloads are expected to be extracted into --raw-dir.
The expected structure of --raw-dir is:
GTA5/
  ├── images/ (RGB screenshots)
  └── labels/ (Cityscapes-colored semantic masks)

This script performs:
1. Train/val split copying
2. TrainId conversion (RGB -> Cityscapes 19-class)
3. 19-class to 8-class mapping
4. Center-crop and resize to 1024x1024
"""

import argparse
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from class_mappings import map_cityscapes_to_8class

# Cityscapes color map
CITYSCAPES_COLORS = {
    (128, 64, 128): 0,   # road
    (244, 35, 232): 1,   # sidewalk
    (70, 70, 70): 2,     # building
    (102, 102, 156): 3,  # wall
    (190, 153, 153): 4,  # fence
    (153, 153, 153): 5,  # pole
    (250, 170, 30): 6,   # traffic light
    (220, 220, 0): 7,    # traffic sign
    (107, 142, 35): 8,   # vegetation
    (152, 251, 152): 9,  # terrain
    (70, 130, 180): 10,  # sky
    (220, 20, 60): 11,   # person
    (255, 0, 0): 12,     # rider
    (0, 0, 142): 13,     # car
    (0, 0, 70): 14,      # truck
    (0, 60, 100): 15,    # bus
    (0, 80, 100): 16,    # train
    (0, 0, 230): 17,     # motorcycle
    (119, 11, 32): 18,   # bicycle
}

def prepare_image(image, width=1024, height=1024):
    """Resize and crop image to target dimensions."""
    w, h = image.size
    if w < h:
        new_w = width
        new_h = int(height * h / w)
    else:
        new_h = height
        new_w = int(width * w / h)
    image = image.resize((new_w, new_h), resample=Image.LANCZOS)
    left = (new_w - width) / 2
    top = (new_h - height) / 2
    right = (new_w + width) / 2
    bottom = (new_h + height) / 2
    return image.crop((left, top, right, bottom))


def prepare_mask(image, width=1024, height=1024):
    """Resize and crop mask to target dimensions."""
    w, h = image.size
    if w < h:
        new_w = width
        new_h = int(height * h / w)
    else:
        new_h = height
        new_w = int(width * w / h)
    image = image.resize((new_w, new_h), resample=Image.NEAREST)
    left = (new_w - width) / 2
    top = (new_h - height) / 2
    right = (new_w + width) / 2
    bottom = (new_h + height) / 2
    return image.crop((left, top, right, bottom))

def color_to_trainid(img_array):
    """Convert RGB mask to Cityscapes 19-class using exact match."""
    h, w, _ = img_array.shape
    out_mask = np.full((h, w), 255, dtype=np.uint8)
    
    for color, label in CITYSCAPES_COLORS.items():
        mask = (img_array[:, :, 0] == color[0]) & \
               (img_array[:, :, 1] == color[1]) & \
               (img_array[:, :, 2] == color[2])
        out_mask[mask] = label
        
    return out_mask

def process_dataset(raw_dir, out_dir, splits_dir):
    """Process images and masks."""
    print("Processing dataset...")
    
    for split in ['train', 'val']:
        split_file = splits_dir / f"{split}.txt"
        if not split_file.exists():
            print(f"  Warning: {split_file} not found.")
            continue
            
        with open(split_file, 'r') as f:
            filenames = [line.strip() for line in f if line.strip()]
            
        # Create output dirs
        out_img_dir = out_dir / 'images' / split
        out_lbl_dir = out_dir / 'labels_8class' / split
        out_img_dir.mkdir(parents=True, exist_ok=True)
        out_lbl_dir.mkdir(parents=True, exist_ok=True)
        
        for filename in tqdm(filenames, desc=f"Processing {split}"):
            # Paths
            img_path = raw_dir / 'images' / filename
            lbl_path = raw_dir / 'labels' / filename
            
            if not img_path.exists() or not lbl_path.exists():
                continue
                
            # Process Image
            img = Image.open(img_path).convert('RGB')
            img_processed = prepare_image(img)
            img_processed.save(out_img_dir / filename)
            
            # Process Label
            lbl = Image.open(lbl_path).convert('RGB')
            lbl_processed = prepare_mask(lbl)
            
            # Convert to TrainId
            lbl_arr = np.array(lbl_processed)
            trainid_mask = color_to_trainid(lbl_arr)
            
            # Map 19-class to 8-class
            class8_mask = map_cityscapes_to_8class(trainid_mask)
            
            # Save
            Image.fromarray(class8_mask).save(out_lbl_dir / filename)

def main():
    parser = argparse.ArgumentParser(description="Prepare GTA5 dataset")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Path to raw GTA5 dataset")
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to output prepared dataset")
    parser.add_argument("--splits-dir", type=Path, default=Path(__file__).parent / 'splits' / 'gta5', help="Path to splits lists")
    args = parser.parse_args()

    process_dataset(args.raw_dir, args.output_dir, args.splits_dir)
    print(f"\nPreparation complete. Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
