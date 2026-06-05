# Voronoi Style Diversification

This directory contains the code for the Voronoi Style Diversification data augmentation method. This method applies local style transfer within randomly generated Voronoi cells to reduce texture bias in training datasets.

## Requirements

The style transfer implementation uses the AdaIN method. You need to download the pre-trained VGG encoder and AdaIN decoder weights:

## Usage
- Dependencies:
    - python >= 3.6
    - Pillow
    - torch
    - torchvision
    - tqdm  
- Download the models:
    - download the models (vgg/decoder) manually from [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN) and move both files to the `models/` directory
    - Get style images: Download train.zip from [Kaggle's painter-by-numbers dataset](https://www.kaggle.com/c/painter-by-numbers/data)


### 1. Training Augmentation

To generate augmented training datasets using Voronoi Style Diversification:

```bash
python stylize_simple.py
```

This script will run `voronoi_style_transfer_simple.py` over your training dataset with predefined parameter combinations (e.g., number of points, stylization proportion). Ensure you update the `CONTENT_DIRS` and `STYLE_DIR` paths in `stylize_simple.py` before running.

### 2. Global Style Transfer (NIR Stylized)

To generate the fully stylized NIR dataset (used in our "nir_voronoi1" experiment), which applies a single global style rather than Voronoi patches, and uses grayscale content/style:

```bash
python stylize.py \
  --content-dir /path/to/NIR/train \
  --style-dir /path/to/style_images \
  --output-dir /path/to/output \
  --grayscale-style \
  --grayscale-content
```

## License & Acknowledgements

Our code in this directory extends the original PyTorch implementation of AdaIN from [pytorch-AdaIN](https://github.com/naoto0804/pytorch-AdaIN). The original code is provided under the following MIT License:

```text
Most files in this directory (code/) are either directly copied from the 
pytorch-AdaIN repository (https://github.com/naoto0804/pytorch-AdaIN)
or adapted slightly. The following license applies to these files:

MIT License

Copyright (c) 2018 Naoto Inoue

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

```
