#!/bin/bash
# Generate distortions for RANUS GT, NIR, and RGB

# Ensure we are in the correct directory
cd "$(dirname "$0")" || exit

echo "Generating GT distortions..."
python dataset_distortions_3channel.py --config ranus_gt_distortions_config.yaml

echo "Generating NIR distortions..."
python dataset_distortions_3channel.py --config ranus_nir_distortions_config.yaml

echo "Generating RGB distortions..."
python dataset_distortions_3channel.py --config ranus_rgb_distortions_config.yaml

echo "All distortions generated successfully."
