import os
import shutil
import subprocess
from pathlib import Path
import numpy as np
from PIL import Image

def backup_and_truncate_splits(splits_dir, backup_dir):
    splits = Path(splits_dir)
    backup = Path(backup_dir)
    if backup.exists():
        shutil.rmtree(backup)
    backup.mkdir(parents=True, exist_ok=True)
    
    for split_file in splits.rglob('*.txt'):
        # backup
        rel_path = split_file.relative_to(splits)
        bk_path = backup / rel_path
        bk_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(split_file, bk_path)
        
        # truncate to 2 lines
        lines = split_file.read_text().splitlines()
        split_file.write_text("\n".join(lines[:2]) + "\n")

def restore_splits(splits_dir, backup_dir):
    shutil.copytree(backup_dir, splits_dir, dirs_exist_ok=True)
    shutil.rmtree(backup_dir)

def check_pixels(img_path, lbl_path):
    print(f"\nChecking {img_path.name}")
    if img_path.exists():
        img = Image.open(img_path)
        print(f"  Image Size: {img.size}")
    else:
        print("  Image NOT FOUND!")
        
    if lbl_path.exists():
        lbl = Image.open(lbl_path)
        arr = np.array(lbl)
        print(f"  Label Size: {lbl.size}")
        print(f"  Unique Classes present (TrainIds): {np.unique(arr)}")
    else:
        print("  Label NOT FOUND!")

if __name__ == '__main__':
    splits_dir = Path('/home/hamscher/projects/synth2real-nir-segmentation/data/splits')
    backup_dir = Path('/home/hamscher/projects/synth2real-nir-segmentation/data/splits_backup')
    
    gta5_out = Path('/home/hamscher/projects/synth2real-nir-segmentation/data/out_GTA5_real')
    ranus_out = Path('/home/hamscher/projects/synth2real-nir-segmentation/data/out_RANUS_real')
    
    if gta5_out.exists(): shutil.rmtree(gta5_out)
    if ranus_out.exists(): shutil.rmtree(ranus_out)
    
    try:
        print("Backing up and truncating splits...")
        backup_and_truncate_splits(splits_dir, backup_dir)
        
        print("\nRunning GTA5 Preprocessing...")
        subprocess.run(["python", "prepare_gta5.py", "--raw-dir", "/home/hamscher/datasets/GTA5", "--output-dir", str(gta5_out)], cwd="/home/hamscher/projects/synth2real-nir-segmentation/data")
        
        print("\nRunning RANUS Preprocessing...")
        subprocess.run(["python", "prepare_ranus.py", "--raw-dir", "/home/hamscher/datasets/RANUS_v1.0", "--output-dir", str(ranus_out)], cwd="/home/hamscher/projects/synth2real-nir-segmentation/data")
        
        print("\n=== PIXEL LEVEL VERIFICATION ===")
        print("GTA5 Test Sample:")
        gta5_lbls = list((gta5_out / 'labels_8class' / 'train').glob('*.png'))
        gta5_imgs = list((gta5_out / 'images' / 'train').glob('*.png'))
        if gta5_lbls and gta5_imgs:
            check_pixels(gta5_imgs[0], gta5_lbls[0])
        else:
            print("GTA5 outputs not generated!")
            
        print("RANUS Test Sample:")
        ranus_lbls = list((ranus_out / 'GT_8class' / 'train').glob('*.png'))
        ranus_imgs = list((ranus_out / 'RGB' / 'train').glob('*.png'))
        if ranus_lbls and ranus_imgs:
            check_pixels(ranus_imgs[0], ranus_lbls[0])
        else:
            print("RANUS outputs not generated!")
            
    finally:
        print("\nRestoring original splits...")
        restore_splits(splits_dir, backup_dir)
        print("Done.")
