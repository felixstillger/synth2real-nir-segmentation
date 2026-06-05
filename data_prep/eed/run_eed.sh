#!/usr/bin/env bash
set -eo pipefail
export CUDA_VISIBLE_DEVICES=1
SCRIPT="$(dirname "$0")/process_directory.py"

SOURCE_DIRS=(
  "../../data/Ranus_prepared/NIR"
  "../../data/Ranus_prepared/RGB"
  # "/home/hamscher/datasets/GTA5/Semseg/images/test"
  # "/home/hamscher/datasets/GTA5/Semseg/nir_synth/test"
  # "/home/hamscher/datasets/GTA5/Semseg/images/val"
  # "/home/hamscher/datasets/GTA5/Semseg/nir_synth/val"
  "../../data/GTA5_prepared/images/train"
  "../../data/GTA5_prepared/nir/train"
)

for SRC in "${SOURCE_DIRS[@]}"; do
  if [[ ! -d "$SRC" ]]; then
    echo "Source not found: $SRC; skipping."
    continue
  fi

  TARGET="${SRC%/}_EED"
  if [[ -d "$TARGET" ]]; then
    echo "Target exists: $TARGET - skipping to avoid overwriting."
    continue
  fi

  echo "Running: $SCRIPT $SRC --target_dir $TARGET"
  python "$SCRIPT" "$SRC" --target_dir "$TARGET" --device cuda:0 --N_processes 1
  echo "Finished: $SRC -> $TARGET"
done

echo "All done."