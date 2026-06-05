import os
import shutil
import subprocess
from pathlib import Path
from PIL import Image
import numpy as np

def create_gta5_mock(base_dir):
    base = Path(base_dir)
    img_dir = base / 'images'
    lbl_dir = base / 'labels'
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    
    # 00001.png (train), 00002.png (train)
    # size 1920x1080
    for name in ['00001.png', '00002.png']:
        # RGB image: random noise
        img = Image.fromarray(np.random.randint(0, 255, (1080, 1920, 3), dtype=np.uint8))
        img.save(img_dir / name)
        
        # Label: road (128, 64, 128) -> trainId 0 -> 8class 2
        lbl_arr = np.full((1080, 1920, 3), (128, 64, 128), dtype=np.uint8)
        lbl = Image.fromarray(lbl_arr)
        lbl.save(lbl_dir / name)

def create_ranus_mock(base_dir):
    base = Path(base_dir)
    # 01_0001 (train), 01_0002 (train)
    # For RANUS, the script expects folder names to be city IDs like 01
    city_gt = base / 'GT' / '01'
    city_rgb = base / 'RGB' / '01'
    city_nir = base / 'NIR' / '01'
    
    for d in [city_gt, city_rgb, city_nir]:
        d.mkdir(parents=True, exist_ok=True)
        
    for name in ['01_0001', '01_0002']:
        # Original size might be 1024x1024
        img = Image.fromarray(np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8))
        img.save(city_rgb / f"{name}_rgb.png")
        img.save(city_nir / f"{name}_nir.png")
        
        # GT: road (128, 64, 128)
        lbl_arr = np.full((1024, 1024, 3), (128, 64, 128), dtype=np.uint8)
        lbl = Image.fromarray(lbl_arr)
        lbl.save(city_gt / f"{name}_gt.png")

def main():
    print("Setting up mock datasets...")
    gta5_raw = Path('mock_GTA5')
    gta5_out = Path('out_GTA5')
    ranus_raw = Path('mock_RANUS')
    ranus_out = Path('out_RANUS')
    
    if gta5_raw.exists(): shutil.rmtree(gta5_raw)
    if gta5_out.exists(): shutil.rmtree(gta5_out)
    if ranus_raw.exists(): shutil.rmtree(ranus_raw)
    if ranus_out.exists(): shutil.rmtree(ranus_out)
    
    create_gta5_mock(gta5_raw)
    create_ranus_mock(ranus_raw)
    
    print("Running prepare_gta5.py...")
    res = subprocess.run(["python", "prepare_gta5.py", "--raw-dir", str(gta5_raw), "--output-dir", str(gta5_out)])
    if res.returncode != 0:
        print("prepare_gta5.py failed!")
        return
        
    print("Running prepare_ranus.py...")
    res = subprocess.run(["python", "prepare_ranus.py", "--raw-dir", str(ranus_raw), "--output-dir", str(ranus_out)])
    if res.returncode != 0:
        print("prepare_ranus.py failed!")
        return
        
    print("\nValidating Outputs...")
    # Validate GTA5
    gta5_lbl = gta5_out / 'labels_8class' / 'train' / '00001.png'
    gta5_img = gta5_out / 'images' / 'train' / '00001.png'
    
    if gta5_lbl.exists():
        lbl_img = Image.open(gta5_lbl)
        print(f"GTA5 Label Size: {lbl_img.size} (Expected: 1024x1024)")
        arr = np.array(lbl_img)
        print(f"GTA5 Label Unique values: {np.unique(arr)} (Expected: [2] since road maps to 2)")
    
    if gta5_img.exists():
        img_val = Image.open(gta5_img)
        print(f"GTA5 Image Size: {img_val.size} (Expected: 1024x1024)")
        
    # Validate RANUS
    ranus_lbl = ranus_out / 'GT_8class' / 'train' / '01_0001_gt.png'
    ranus_img = ranus_out / 'RGB' / 'train' / '01_0001_rgb.png'
    
    if ranus_lbl.exists():
        lbl_img = Image.open(ranus_lbl)
        print(f"RANUS Label Size: {lbl_img.size} (Expected: 1024x1024)")
        arr = np.array(lbl_img)
        print(f"RANUS Label Unique values: {np.unique(arr)} (Expected: [2] since road maps to 2)")
        
    if ranus_img.exists():
        img_val = Image.open(ranus_img)
        print(f"RANUS Image Size: {img_val.size} (Expected: 1024x1024)")

    print("\nAll preprocessing tests completed.")

if __name__ == '__main__':
    main()
