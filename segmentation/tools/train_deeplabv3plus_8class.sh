#!/bin/bash
# Training script for DeepLabV3+ ResNet50-D8 models (8-class, 60k iterations, AdamW)
# Run from the segmentation/ directory of synth2real-nir-segmentation:
#   cd <repo>/segmentation
#   nohup bash tools/train_deeplabv3plus_8class.sh > train_deeplabv3plus_8class.log 2>&1 &

set -e  # Exit on error

# GPU configuration
export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1

# Version suffixes for work_dir - will train each config with all suffixes (3 seeds)
VERSION_SUFFIXES=("" "_v1" "_v2")

# Base directory: mmsegmentation submodule root (for tools/train.py)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MMSEG_ROOT="$SCRIPT_DIR/../mmsegmentation"
cd "$MMSEG_ROOT"

# Log file for this script
SCRIPT_LOG="$SCRIPT_DIR/../train_deeplabv3plus_8class_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee -a "$SCRIPT_LOG"
echo "DeepLabV3+ ResNet50-D8 Training (8-class, 60k iterations, AdamW)" | tee -a "$SCRIPT_LOG"
echo "Started: $(date)" | tee -a "$SCRIPT_LOG"
echo "MMSEG root: $MMSEG_ROOT" | tee -a "$SCRIPT_LOG"
echo "========================================" | tee -a "$SCRIPT_LOG"

# Config paths relative to the mmsegmentation root
CONFIGS_BASE="$SCRIPT_DIR/../configs/deeplabv3plus"

CONFIGS=(
    # GTA5 NIR synthesized baseline
    "$CONFIGS_BASE/deeplabv3plus_r50-d8_4xb2-60k_gta5_8class_nir.py"

    # GTA5 NIR + Voronoi style diversification variants
    "$CONFIGS_BASE/deeplabv3plus_r50-d8_4xb2-60k_gta5_8class_nir_voronoi1.py"

    # GTA5 RGB baseline and Voronoi variants (ablation / comparison)
    "$CONFIGS_BASE/deeplabv3plus_r50-d8_4xb2-60k_gta5_8class_rgb.py"
    "$CONFIGS_BASE/deeplabv3plus_r50-d8_4xb2-60k_gta5_8class_rgb_v1_100.py"
    "$CONFIGS_BASE/deeplabv3plus_r50-d8_4xb2-60k_gta5_8class_rgb_voronoi4_025.py"
    "$CONFIGS_BASE/deeplabv3plus_r50-d8_4xb2-60k_gta5_8class_rgb_voronoi4_075.py"
    "$CONFIGS_BASE/deeplabv3plus_r50-d8_4xb2-60k_gta5_8class_rgb_voronoi8_025.py"
    "$CONFIGS_BASE/deeplabv3plus_r50-d8_4xb2-60k_gta5_8class_rgb_voronoi8_075.py"
    "$CONFIGS_BASE/deeplabv3plus_r50-d8_4xb2-60k_gta5_8class_rgb_v16_025.py"
    "$CONFIGS_BASE/deeplabv3plus_r50-d8_4xb2-60k_gta5_8class_rgb_v16_075.py"


    # RANUS supervised oracle baselines
    "$CONFIGS_BASE/deeplabv3plus_r50-d8_4xb2-60k_ranus_8class_rgb.py"
    "$CONFIGS_BASE/deeplabv3plus_r50-d8_4xb2-60k_ranus_8class_nir.py"
)

# Calculate total training runs (configs × versions)
TOTAL_CONFIGS=${#CONFIGS[@]}
TOTAL_VERSIONS=${#VERSION_SUFFIXES[@]}
TOTAL_RUNS=$((TOTAL_CONFIGS * TOTAL_VERSIONS))
CURRENT_RUN=0

echo "Total configs: $TOTAL_CONFIGS" | tee -a "$SCRIPT_LOG"
echo "Total seeds: $TOTAL_VERSIONS" | tee -a "$SCRIPT_LOG"
echo "Total training runs: $TOTAL_RUNS" | tee -a "$SCRIPT_LOG"

# Nested loop: for each config, train all seed versions
for CONFIG in "${CONFIGS[@]}"; do
    CONFIG_NAME=$(basename "$CONFIG" .py)

    for VERSION_SUFFIX in "${VERSION_SUFFIXES[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        VERSION_NAME="${VERSION_SUFFIX:-original}"

        # Extract work_dir from config and append VERSION_SUFFIX for seed variants
        ORIGINAL_WORKDIR=$(python3 -c "
import re, sys
text = open('$CONFIG').read()
m = re.search(r\"work_dir\s*=\s*['\\\"](.+?)['\\\"]\", text)
print(m.group(1) if m else sys.exit(1))
")
        NEW_WORKDIR="${ORIGINAL_WORKDIR}${VERSION_SUFFIX}"

        echo "" | tee -a "$SCRIPT_LOG"
        echo "========================================" | tee -a "$SCRIPT_LOG"
        echo "[$CURRENT_RUN/$TOTAL_RUNS] $CONFIG_NAME (seed: $VERSION_NAME)" | tee -a "$SCRIPT_LOG"
        echo "Started: $(date)" | tee -a "$SCRIPT_LOG"
        echo "Work dir: $NEW_WORKDIR" | tee -a "$SCRIPT_LOG"
        echo "========================================" | tee -a "$SCRIPT_LOG"

        if python tools/train.py "$CONFIG" --launcher none --work-dir "$NEW_WORKDIR"; then
            echo "✓ SUCCESS: $CONFIG_NAME ($VERSION_NAME) at $(date)" | tee -a "$SCRIPT_LOG"
        else
            echo "✗ FAILED: $CONFIG_NAME ($VERSION_NAME) at $(date)" | tee -a "$SCRIPT_LOG"
            echo "Continuing with next training..." | tee -a "$SCRIPT_LOG"
        fi
    done
done

echo "" | tee -a "$SCRIPT_LOG"
echo "========================================" | tee -a "$SCRIPT_LOG"
echo "All trainings completed!" | tee -a "$SCRIPT_LOG"
echo "Finished: $(date)" | tee -a "$SCRIPT_LOG"
echo "========================================" | tee -a "$SCRIPT_LOG"
