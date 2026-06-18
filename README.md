# Texture-Shape Bias Balancing for Robust Synthetic-to-Real Semantic Segmentation in Automotive NIR Imagery

> **Accepted to ECML-PKDD 2026**  
> F. Stillger\*, B. Hamscher\*, L. Hahn, A. Mütze, T. Meisen, K. Maag

This repository contains the code for synthesizing Near-Infrared (NIR) automotive datasets from RGB images and evaluating semantic segmentation models' robustness using texture and shape bias metrics.

## Repository Structure

- `environments/`: Conda environment exports (`mmseg` and `synthesis`).
- `data/`: Automated data preparation scripts for RANUS and GTA5.
- `synthesis/`: SDXL + ControlNet Union inference and LoRA training code.
- `style_transfer/`: Voronoi Style Diversification method for training augmentation.
- `segmentation/`: mmsegmentation configurations, custom datasets, metrics, and evaluation framework.
- `eed/`: Edge Enhancing Diffusion (EED) data generation for shape bias evaluation.

## Environment Setup

The project uses two distinct Conda environments due to dependency conflicts between HuggingFace Diffusers and MMSegmentation.

1. **Synthesis Environment** (for SDXL synthesis and LoRA):
   ```bash
   conda env create -f environments/synthesis_environment.yml
   conda activate sdxl
   ```

2. **Segmentation Environment** (for mmsegmentation):
   ```bash
   conda env create -f environments/mmseg_environment.yml
   conda activate nir
   ```

## Data Preparation

### 1. RANUS Dataset
The RANUS dataset contains paired RGB and NIR images.
1. Download the dataset from the [official source](https://drive.google.com/file/d/1WJ7rcCeMBPy9Qb2c_pwI0rUIlFqotkzT/view).
2. Run the preparation script:
   ```bash
   python data/prepare_ranus.py --raw-dir /path/to/extracted/RANUS_v1.0 --output-dir data/Ranus_prepared
   ```

### 2. GTA5 Dataset
The GTA5 dataset provides synthetic RGB images and Cityscapes-format labels.
1. Download the dataset from the [official source](https://download.visinf.tu-darmstadt.de/data/from_games/).
2. Run the preparation script:
   ```bash
   python data/prepare_gta5.py --raw-dir /path/to/extracted/GTA5 --output-dir data/GTA5_prepared
   ```

## NIR Image Synthesis (SDXL)

We synthesize pseudo-NIR images from GTA5 RGB using Stable Diffusion XL, ControlNet Union (Promax), and a custom LoRA trained on RANUS.

1. Review `synthesis/lora_training/README.md` for our exact LoRA training parameters.
2. Generate the synthetic dataset:
   ```bash
   python synthesis/synthesize_nir.py \
     --img_dir data/GTA5_prepared/images/train \
     --out_dir data/GTA5_prepared/nir_synth/train
   ```

## Voronoi Style Diversification

To reduce texture bias, we apply local AdaIN style transfer within random Voronoi regions.

1. Download the pre-trained AdaIN weights (`decoder.pth`, `vgg_normalised.pth`) into `style_transfer/models/`.
2. Generate stylized training datasets:
   ```bash
   cd style_transfer
   python stylize_simple.py
   ```

## Semantic Segmentation

### Training
Models are defined in `segmentation/configs/`. We train DeepLabV3+, Mask2Former, and SegFormer on the synthesized NIR data and our Voronoi stylized variants.

> [!IMPORTANT]
> For SegFormer, you must first download the pre-trained `mit_b5.pth` weights from [Google Drive](https://drive.google.com/file/d/1ZXvpu5B3EcxSuDbV2fXAq_z_cbZIWzkX/view) and save the file to `segmentation/pretrained/mit_b5.pth`.

```bash
cd segmentation
bash tools/train_deeplabv3plus_8class.sh
```

### Evaluation
Our evaluation framework measures standard cross-domain IoU as well as shape and texture biases.

```bash
cd segmentation
python tools/run_evaluations_scheduler.py --gpus 0 1 
```

After evaluations finish, generate the final normalized robustness and shape bias scores using the post-processing scripts:

```bash
# Calculate Distortion Robustness (mPC-AUC)
python tools/compute_robustness_scores.py \
  --input-csv /path/to/results_timestamp.csv \
  --real-baseline-config mask2former_swin-l_8xb2-60k_gta5_8class_nir

# Calculate Shape Bias (S_cd) and Robustness (R_cd)
python tools/compute_shape_bias_scores.py \
  --shape-bias-csv /path/to/detailed_results.csv \
  --cross-domain-csv /path/to/summary_results.csv
```


## Acknowledgements & Credits

- **Edge Enhancing Diffusion (EED)**: The EED code is credited to Edgar Heinert and his paper *Reducing Texture Bias of Deep Neural Networks via Edge Enhancing Diffusion*. The original code is provided under the MIT License.
- **Voronoi Shuffling**: The Voronoi dataset shuffling scripts are credited to Edgar Heinert. If you use the Voronoi shuffling code, please cite the following paper:
```bibtex
@misc{heinert2025shapebiasrobustnessevaluation,
      title={Shape Bias and Robustness Evaluation via Cue Decomposition for Image Classification and Segmentation}, 
      author={Edgar Heinert and Thomas Gottwald and Annika Mütze and Matthias Rottmann},
      year={2025},
      eprint={2503.12453},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2503.12453}, 
}
```
- **Dataset Distortions**: The common image corruption methods (distortions) found in `data/distortions/` are adapted from the [generalisation-humans-DNNs](https://github.com/rgeirhos/generalisation-humans-DNNs) repository by Robert Geirhos et al. The original license is included in that directory.

## Citation

```bibtex
@inproceedings{stillger2026texture,
  title={Texture-Shape Bias Balancing for Robust Synthetic-to-Real Semantic Segmentation in Automotive NIR Imagery},
  author={Stillger, Felix and Hamscher, Ben and M{\"u}tze, Annika and Hahn, Lukas and Meisen, Tobias and Maag, Kira},
  booktitle={European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD)},
  year={2026}
}
```
