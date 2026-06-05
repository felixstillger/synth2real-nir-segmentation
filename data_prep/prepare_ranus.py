#!/usr/bin/env python3
"""
Prepare RANUS dataset for 8-class Semantic Segmentation.

Downloads are expected to be extracted into --raw-dir.
The expected structure of --raw-dir is:
RANUS_v1.0/
  ├── GT/  (01 to 50 folders with _gt.png)
  ├── NIR/ (01 to 50 folders with _nir.png)
  └── RGB/ (01 to 50 folders with _rgb.png)

This script performs:
1. Flattening and filename normalization
2. Train/val/test splitting
3. Fuzzy mask conversion (RGB to indices)
4. 10-class to 8-class mapping
5. Nearest-neighbor resizing to 1024x1024
"""

import argparse
import shutil
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm

from class_mappings import map_ranus_to_8class

def normalize_and_flatten(raw_dir, temp_dir):
    """Flatten 50-folder structure and normalize filenames sequentially."""
    print("Step 1: Normalizing and flattening dataset...")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    for modality, suffix in [('RGB', '_rgb.png'), ('NIR', '_nir.png'), ('GT', '_gt.png')]:
        mod_dir = raw_dir / modality
        out_dir = temp_dir / modality
        out_dir.mkdir(exist_ok=True)
        
        if not mod_dir.exists():
            print(f"  Warning: {mod_dir} not found. Skipping.")
            continue
            
        # Get all png files and group by city folder
        png_files = list(mod_dir.rglob("*.png"))
        folder_groups = {}
        for f in png_files:
            # The parent folder name is the city id (01-50)
            city_id = f.parent.name
            if city_id not in folder_groups:
                folder_groups[city_id] = []
            folder_groups[city_id].append(f)
            
        # Rename sequentially within each city folder
        for city_id in sorted(folder_groups.keys()):
            files = folder_groups[city_id]
            files.sort(key=lambda x: x.name)  # sort to maintain deterministic order
            
            for i, old_file in enumerate(files, 1):
                new_filename = f"{city_id}_{i:04d}.png"
                shutil.copy2(old_file, out_dir / new_filename)
                
    print("  Done flattening.")

def apply_splits(temp_dir, out_dir, splits_dir):
    """Move files into train/val/test subdirectories based on split lists."""
    print("Step 2: Applying train/val/test splits...")
    
    for split in ['train', 'val', 'test']:
        split_file = splits_dir / f"{split}.txt"
        if not split_file.exists():
            print(f"  Warning: {split_file} not found.")
            continue
            
        with open(split_file, 'r') as f:
            # Files in split lists contain the suffix (e.g. 01_0001_rgb.png)
            # We need the base name without suffix
            basenames = [line.strip().replace('_rgb.png', '').replace('_nir.png', '').replace('_gt.png', '') 
                         for line in f if line.strip()]
            
        for modality, old_suffix in [('RGB', '_rgb.png'), ('NIR', '_nir.png'), ('GT', '_gt.png')]:
            src_dir = temp_dir / modality
            dst_dir = out_dir / modality / split
            dst_dir.mkdir(parents=True, exist_ok=True)
            
            if not src_dir.exists():
                continue
                
            for basename in basenames:
                filename = f"{basename}.png"
                src_file = src_dir / filename
                if src_file.exists():
                    shutil.copy2(src_file, dst_dir / filename)
                    
    print("  Done splitting.")

def fuzzy_convert(img_array, colors_to_labels, threshold=10):
    """Convert RGB mask to indices using L1 distance fuzzy mapping."""
    h, w, _ = img_array.shape
    out_mask = np.full((h, w), 255, dtype=np.uint8)
    
    for color, label in colors_to_labels.items():
        diff = np.abs(img_array.astype(np.int32) - np.array(color).astype(np.int32))
        dist = np.sum(diff, axis=-1)
        mask = dist <= threshold
        out_mask[mask] = label
        
    return out_mask

def process_masks(out_dir):
    """Convert GT masks: fuzzy RGB -> index -> 8-class mapping -> resize."""
    print("Steps 3-5: Converting, mapping, and resizing masks...")
    
    # RANUS v1.0 standard colors
    ranus_colors = {
        (0, 0, 0): 0,       # Ignore
        (128, 128, 128): 1, # Sky
        (128, 0, 0): 2,     # Ground
        (0, 128, 128): 3,   # Water
        (128, 128, 0): 4,   # Mountain
        (128, 64, 128): 5,  # Road
        (128, 0, 128): 6,   # Construction
        (0, 128, 0): 7,     # Vegetation
        (64, 64, 0): 8,     # Object
        (0, 0, 128): 9,     # Vehicle
        (64, 0, 128): 10    # Human
    }

    # Combined map: RANUS (1-10) -> 8-class (0-7)
    # 8-class: sky(0), ground(1), road(2), construction(3), vegetation(4), object(5), vehicle(6), human(7)
    # Water(3) & Mountain(4) -> 255 (ignored)
    
    gt_base_dir = out_dir / 'GT'
    gt_8class_dir = out_dir / 'GT_8class'
    
    for split in ['train', 'val', 'test']:
        src_dir = gt_base_dir / split
        dst_dir = gt_8class_dir / split
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        if not src_dir.exists():
            continue
            
        files = list(src_dir.glob("*.png"))
        for f in tqdm(files, desc=f"Processing {split} masks"):
            img = Image.open(f).convert('RGB')
            img_arr = np.array(img)
            
            # 1. Fuzzy convert RGB to 10-class index (0-10)
            indexed_10class = fuzzy_convert(img_arr, ranus_colors, threshold=10)
            
            # 2. Map RANUS 10-class directly to 8-class (0-7)
            mapped_8class = map_ranus_to_8class(indexed_10class)
            
            # 3. Resize to 1024x1024
            mask_img = Image.fromarray(mapped_8class)
            mask_img = mask_img.resize((1024, 1024), Image.NEAREST)
            
            # Save
            mask_img.save(dst_dir / f.name)

    print("  Done mask processing.")

def main():
    parser = argparse.ArgumentParser(description="Prepare RANUS dataset")
    parser.add_argument("--raw-dir", type=Path, required=True, help="Path to raw extracted RANUS_v1.0 dataset")
    parser.add_argument("--output-dir", type=Path, required=True, help="Path to output prepared dataset")
    parser.add_argument("--splits-dir", type=Path, default=Path(__file__).parent / 'splits' / 'ranus', help="Path to splits lists")
    parser.add_argument("--step", type=str, choices=['all', 'flatten', 'split', 'process'], default='all', help="Step to run")
    args = parser.parse_args()

    temp_dir = args.output_dir / "temp_flattened"
    
    if args.step in ['all', 'flatten']:
        normalize_and_flatten(args.raw_dir, temp_dir)
        
    if args.step in ['all', 'split']:
        apply_splits(temp_dir, args.output_dir, args.splits_dir)
        
    if args.step in ['all', 'process']:
        process_masks(args.output_dir)
        
    if args.step == 'all':
        # Cleanup
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        print(f"\nPreparation complete. Output saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
