# UNet Lung CT Segmentation

A PyTorch implementation of UNet for automated lung segmentation in CT scans.

## Overview

This project implements a UNet convolutional neural network for segmenting lung regions in computed tomography (CT) images. The model is trained to accurately identify and isolate lung tissue from surrounding anatomical structures.

## Features

- UNet architecture for medical image segmentation
- PyTorch implementation
- Efficient 2D convolution-based approach
- Pre-processing pipeline for CT images
- Evaluation metrics (Dice coefficient, IoU)

## Installation

```bash
git clone https://github.com/yourusername/unet-lung-ct-segmentation.git
cd unet-lung-ct-segmentation
pip install -r requirements.txt
```

## Usage

```python
from model import UNet
model = UNet()
# Load and preprocess CT scan
segmentation = model.predict(ct_image)
```

## Dataset

Specify the dataset used (e.g., LUNA16, LIDC-IDRI, or custom dataset).

## Training

```bash
python train.py --epochs 100 --batch-size 32
```

## Results

Document model performance and example segmentation outputs.

## References

- UNet: Ronneberger et al. (https://arxiv.org/abs/1505.04597)

## License

MIT License
