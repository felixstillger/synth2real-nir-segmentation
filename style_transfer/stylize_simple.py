import subprocess
import datetime
import time
from pathlib import Path

# Base paths
CONTENT_DIRS = [
    # str(Path("~/datasets/Ranus_splits/RGB/train").expanduser()),
    # str(Path("~/datasets/Ranus_splits/NIR/train").expanduser()),
    str(Path("~/datasets/GTA5/Semseg/images/train").expanduser()),
    # str(Path("~/datasets/GTA5/Semseg/nir_synth/train").expanduser()),
    # str(Path("~/datasets/Ranus_splits/RGB/val").expanduser()),
    # str(Path("~/datasets/Ranus_splits/NIR/val").expanduser()),
    str(Path("~/datasets/GTA5/Semseg/images/val").expanduser()),
    # str(Path("~/datasets/GTA5/Semseg/nir_synth/val").expanduser())
]

STYLE_DIR = str(Path("~/datasets/train").expanduser())

# Parameter combinations
NUM_POINTS = [4, 8, 16] # [4, 16]  # Different numbers of Voronoi cells
STYLIZE_PROPS = [0.25, 0.75]   # Different proportions to stylize
ALPHAS = [1.0]     # Different style strength values
GRAYSCALE = False #True  # Whether to convert style images to grayscale before stylization

# Generate all parameter combinations
PARAMS = []
for np in NUM_POINTS:
    for sp in STYLIZE_PROPS:
        for alpha in ALPHAS:
            # Create output paths for each content directory
            output_dirs = []
            for content_dir in CONTENT_DIRS:
                parts = Path(content_dir).parts
                
                if "Ranus_splits" in parts:
                    modality = parts[-2]  # RGB or NIR
                    split = parts[-1]     # train, val, or test
                    base_path = Path(content_dir).parent.parent
                    modality_str = modality.lower()
                    sp_str = f"{int(sp*100):03d}"
                    output_dir = f"{base_path}/v{np}_p{sp_str}/{modality_str}/{split}"
                elif "GTA5" in parts:
                    modality = parts[-2]  # images or nir_synth
                    split = parts[-1]     # train, val, or test
                    base_path = Path(content_dir).parent.parent
                    modality_str = 'nir' if modality == 'nir_synth' else 'rgb' if modality == 'images' else modality
                    sp_str = f"{int(sp*100):03d}"
                    output_dir = f"{base_path}/v{np}_p{sp_str}/{modality_str}/{split}"
                else:
                    dataset_name = Path(content_dir).parent.name
                    sp_str = f"{int(sp*100):03d}"
                    output_dir = f"{Path(content_dir).parent.parent}/stylized/{dataset_name}_train_v{np}_p{sp_str}"
                
                output_dirs.append(output_dir)
            
            # Convert lists to space-separated strings for command line
            content_dirs_str = " ".join(CONTENT_DIRS)
            output_dirs_str = " ".join(output_dirs)
            
            param_str = (f"--num_points {np} "
                        f"--output_dirs {output_dirs_str} "
                        f"--content_dirs {content_dirs_str} "
                        f"--style_dir {STYLE_DIR} "
                        f"--stylize_proportion {sp} "
                        f"--alpha {alpha}")
            if GRAYSCALE:
                param_str += " --grayscale"

            PARAMS.append(param_str)

SCRIPT_PATH = "voronoi_style_transfer_simple.py"

def run_style_transfers():
    for i, params in enumerate(PARAMS, 1):
        print(f"\n[{datetime.datetime.now()}] Starting run {i}/{len(PARAMS)}")
        print(f"Parameters: {params}")
        
        try:
            cmd = f"python {SCRIPT_PATH} {params}"
            subprocess.run(cmd, shell=True, check=True)
            
            print(f"[{datetime.datetime.now()}] Completed run {i}/{len(PARAMS)}")
            
            # Add small delay between runs
            if i < len(PARAMS):
                time.sleep(5)
                
        except subprocess.CalledProcessError as e:
            print(f"Error in run {i}: {e}")
            continue

if __name__ == "__main__":
    print(f"Will run {len(PARAMS)} parameter combinations")
    print(f"Processing {len(CONTENT_DIRS)} datasets:")
    for content_dir in CONTENT_DIRS:
        print(f"  - {content_dir}")
    run_style_transfers()
