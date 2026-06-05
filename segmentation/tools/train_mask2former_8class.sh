#!/bin/bash
# Training script for Mask2Former Swin-L models (8-class, 60k iterations)
# Usage: nohup bash train_mask2former_8class.sh > train_mask2former_8class.log 2>&1 &

set -e  # Exit on error

# GPU configuration
export CUDA_VISIBLE_DEVICES=1
NUM_GPUS=1

# Version suffixes for work_dir - will train each config with all suffixes
VERSION_SUFFIXES=("_v1" "_v2") # "_v2"

# Base directory
MMSEG_ROOT="/home/hamscher/projects/NIR/mmsegmentation"
cd $MMSEG_ROOT

# Log file for this script
SCRIPT_LOG="train_mask2former_8class_$(date +%Y%m%d_%H%M%S).log"

echo "========================================" | tee -a $SCRIPT_LOG
echo "Mask2Former Swin-L Training (8-class, 60k)" | tee -a $SCRIPT_LOG
echo "Started: $(date)" | tee -a $SCRIPT_LOG
echo "========================================" | tee -a $SCRIPT_LOG

# Training configurations
CONFIGS=(
    # GTA5 RGB baseline and Voronoi variants
    # "configs/mask2former/mask2former_swin-l_8xb2-60k_gta5_8class_rgb.py"
    # "configs/mask2former/mask2former_swin-l_8xb2-60k_gta5_8class_rgb_v1_100.py"
    # "configs/mask2former/mask2former_swin-l_8xb2-60k_gta5_8class_rgb_v4_025.py"
    # "configs/mask2former/mask2former_swin-l_8xb2-60k_gta5_8class_rgb_v4_075.py"
    # "configs/mask2former/mask2former_swin-l_8xb2-60k_gta5_8class_rgb_v8_025.py"
    # "configs/mask2former/mask2former_swin-l_8xb2-60k_gta5_8class_rgb_v8_075.py"
    # "configs/mask2former/mask2former_swin-l_8xb2-60k_gta5_8class_rgb_v16_025.py"
    # "configs/mask2former/mask2former_swin-l_8xb2-60k_gta5_8class_rgb_v16_075.py"
    
    # GTA5 NIR baseline and Voronoi variants
    # "configs/mask2former/mask2former_swin-l_8xb2-60k_gta5_8class_nir.py"
    # "configs/mask2former/mask2former_swin-l_8xb2-60k_gta5_8class_nir_v1_100.py"
    # "configs/mask2former/mask2former_swin-l_8xb2-60k_gta5_8class_nir_v4_025.py"
    # "configs/mask2former/mask2former_swin-l_8xb2-60k_gta5_8class_nir_v4_075.py"
    # "configs/mask2former/mask2former_swin-l_8xb2-60k_gta5_8class_nir_v8_025.py"
    # "configs/mask2former/mask2former_swin-l_8xb2-60k_gta5_8class_nir_v8_075.py"
    # "configs/mask2former/mask2former_swin-l_8xb2-60k_gta5_8class_nir_v16_025.py"
    # "configs/mask2former/mask2former_swin-l_8xb2-60k_gta5_8class_nir_v16_075.py"
    
    # Ranus RGB and NIR
    "configs/mask2former/mask2former_swin-l_8xb2-60k_ranus_8class_rgb.py"
    # "configs/mask2former/mask2former_swin-l_8xb2-60k_ranus_8class_nir.py"
    # Bias Oracle networks RANUS NIR
    # "/home/hamscher/projects/NIR/mmsegmentation/configs/mask2former/mask2former_swin-l_8xb2-60k_ranus_8class_nir_eed.py"
    # "/home/hamscher/projects/NIR/mmsegmentation/configs/mask2former/mask2former_swin-l_8xb2-60k_ranus_8class_nir_voronoi_shuffled64.py"
)

# Calculate total training runs (configs × versions)
TOTAL_CONFIGS=${#CONFIGS[@]}
TOTAL_VERSIONS=${#VERSION_SUFFIXES[@]}
TOTAL_RUNS=$((TOTAL_CONFIGS * TOTAL_VERSIONS))
CURRENT_RUN=0

# Nested loop: for each version suffix, train all configs
for VERSION_SUFFIX in "${VERSION_SUFFIXES[@]}"; do
    VERSION_NAME="${VERSION_SUFFIX:-'original'}"  # Display name for logging
    
    echo "" | tee -a "$SCRIPT_LOG"
    echo "========================================" | tee -a "$SCRIPT_LOG"
    echo "Starting version: $VERSION_NAME" | tee -a "$SCRIPT_LOG"
    echo "========================================" | tee -a "$SCRIPT_LOG"

    for CONFIG in "${CONFIGS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        CONFIG_NAME=$(basename "$CONFIG" .py)

        # Extract work_dir from config and append VERSION_SUFFIX
        ORIGINAL_WORKDIR=$(grep "^work_dir = " "$CONFIG" | sed "s/work_dir = '//" | sed "s/'$//")
        NEW_WORKDIR="${ORIGINAL_WORKDIR}${VERSION_SUFFIX}"
        
        echo "" | tee -a "$SCRIPT_LOG"
        echo "========================================" | tee -a "$SCRIPT_LOG"
        echo "[$CURRENT_RUN/$TOTAL_RUNS] Training: $CONFIG_NAME (version: $VERSION_NAME)" | tee -a "$SCRIPT_LOG"
        echo "Started: $(date)" | tee -a "$SCRIPT_LOG"
        echo "Work dir: $NEW_WORKDIR" | tee -a "$SCRIPT_LOG"
        echo "========================================" | tee -a "$SCRIPT_LOG"
        
        # Run training
        if python tools/train.py "$CONFIG" --launcher none --work-dir "$NEW_WORKDIR"; then
            echo "✓ SUCCESS: $CONFIG_NAME ($VERSION_NAME) completed at $(date)" | tee -a "$SCRIPT_LOG"
        else
            echo "✗ FAILED: $CONFIG_NAME ($VERSION_NAME) failed at $(date)" | tee -a "$SCRIPT_LOG"
            echo "Continuing with next training..." | tee -a "$SCRIPT_LOG"
        fi
    done
done

echo "" | tee -a "$SCRIPT_LOG"
echo "========================================" | tee -a "$SCRIPT_LOG"
echo "All trainings completed!" | tee -a "$SCRIPT_LOG"
echo "Finished: $(date)" | tee -a "$SCRIPT_LOG"
echo "========================================" | tee -a "$SCRIPT_LOG"
