#!/bin/bash
# RANUS Voronoi Shuffled Dataset Generation Commands

# RANUS - filtered_t10_agree80 (Voronoi 128)
# -----------------------------
# 10. Generate RGB images + GT_8class labels (all splits)
echo "Command 10: RANUS RGB with 8-class labels (Voronoi 128)" > voronoi_ranus_rgb_128.log
nohup python voronoi_shuffled_ranus.py \
  --cell_number 128 \
  --img_root ../../data/Ranus_prepared/RGB/ \
  --label_root ../../data/Ranus_prepared/GT_8class/ \
  --splits train val test \
  --result_dir ../../data/Ranus_prepared/Voronoi_shuffled_128 \
  --image_subfolder RGB \
  --label_subfolder GT_8class/ \
  --seed 4224 \
  >> voronoi_ranus_rgb_128.log 2>&1 &
echo "Started RANUS RGB job (Voronoi 128, PID: $!)"

# 11. Generate NIR images only (reuse labels from step 10)
echo "Command 11: RANUS NIR (images only, Voronoi 128)" > voronoi_ranus_nir_128.log
nohup python voronoi_shuffled_ranus.py \
  --cell_number 128 \
  --img_root ../../data/Ranus_prepared/NIR/ \
  --label_root ../../data/Ranus_prepared/GT_8class/ \
  --splits train val test \
  --result_dir ../../data/Ranus_prepared/Voronoi_shuffled_128 \
  --image_subfolder NIR \
  --images_only \
  --seed 4224 \
  >> voronoi_ranus_nir_128.log 2>&1 &
echo "Started RANUS NIR job (Voronoi 128, PID: $!)"


echo "All jobs started. Monitor with: tail -f voronoi_*_.log"
