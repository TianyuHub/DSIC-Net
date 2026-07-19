<div align="center">
  <h1>🔎 DSIC-Net: A Dual-Space Information Cascaded Correction Network for Brain Age Prediction from Structural MRI</h1>
</div>

> Tianyu Sun, xxx, xxx, Tong Zhang</br>
> *Medical Image Analysis (MedIA)*, 2026
> 
---

## 👀 Overview

DSIC-Net is a two-stage framework for brain age prediciton using brain T1-weighted Magnetic Resonance Imaging （MRI） scans.

1. **Image Space Net**  
   Stage 1 learns age-sensitive representations directly from the full 3D T1-weighted sMRI volume.

2. **Bounded Graph-Guided Residual Correction Net**  
   Stage 2 corrected brain age prediciton predicted from stage 1 using brain net information. 

DSIC-Net was validated on 10 diverse public datasets, achieving an average Dice only **3.08%** lower than the fully supervised baseline.

---

## 🚀 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/iMED-Lab/DSIC-Net.git
cd DSIC-Net
```

### 2. Create environments

#### Create a conda environment

```bash
conda create -n DSIC-Net python=3.10 -y
conda activate DSIC-Net
```

#### Install PyTorch

```bash
pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu118
```

#### Install SAM2 and other dependencies

```bash
pip install -e ./code_pl
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation

Please organize the dataset as follows:

```text
DatasetName/
├── file_list/
│   ├── train_all_frames.txt
│   ├── train_label_frames.txt
│   └── test_all_frames.txt
├── Train/
│   ├── JPEGImages/
│   │   ├── XXX001.png
│   │   ├── XXX002.png
│   │   └── ...
│   └── Annotations/
│       ├── XXX001.png
│       ├── XXX002.png
│       └── ...
└── Test/
    ├── JPEGImages/
    │   ├── XXX101.png
    │   ├── XXX102.png
    │   └── ...
    └── Annotations/
        ├── XXX101.png
        ├── XXX102.png
        └── ...
```

Notes:

* `train_all_frames.txt`: all training image filenames
* `train_label_frames.txt`: the selected labeled image filename for the one-shot setting
* `test_all_frames.txt`: all test image filenames
* Filenames in `file_list/*.txt` should be plain filenames, for example: `XXX001.png`
* Input images should be 3-channel `.png`
* Ground-truth labels should be single-channel `.png`
* Pseudo labels generated in Stage 1 are also single-channel `.png`

We provide a dataset structure example in: `DSIC-Net/data/ISIC2016`

---

## 🤖 Model Preparation

Please download SAM2 checkpoints from the official repository: [SAM2 Official Repository](https://github.com/facebookresearch/sam2)

Then place the downloaded checkpoint files under: `DSIC-Net/code_pl/checkpoints/`

For example:

```text
DSIC-Net/
└── code_pl/
    └── checkpoints/
        ├── sam2.1_hiera_tiny.pt
        ├── sam2.1_hiera_small.pt
        ├── sam2.1_hiera_base_plus.pt
        └── sam2.1_hiera_large.pt
```

We recommend using `sam2.1_hiera_small.pt` by default.

---

## ⚡ Run DSIC-Net

#### 1. Stage 1: Multi-view Pseudo-label Generation

Run Stage 1 to generate:

* `pl_original`
* `pl_rotate`
* `pl_flip`
* `divergence_mask`

```bash
python code_pl/multi_view_inference.py \
  --data-root /path/to/DatasetName \
  --checkpoint code_pl/checkpoints/sam2.1_hiera_small.pt \
  --cfg code_pl/configs/sam2.1/sam2.1_hiera_s.yaml
```

After Stage 1, the `Train/` directory will be automatically updated as:

```text
Train/
├── JPEGImages/
├── Annotations/
├── pl_original/
├── pl_rotate/
├── pl_flip/
└── divergence_mask/
```

#### 2. Stage 2: Segmentation Training

Train the segmentation model with the generated pseudo labels:

```bash
python code_seg/train.py \
  --data-root /path/to/DatasetName \
  --exp-name DSIC-Net_unet \
  --num-classes 2 \
  --image-size 256 \
  --batch-size 4 \
  --epochs 100
```

#### 3. Testing

Run testing with the trained model:

```bash
python code_seg/test.py \
  --data-root /path/to/DatasetName \
  --checkpoint checkpoints/DSIC-Net_unet/best.pth \
  --output-dir outputs/DSIC-Net_unet_test \
  --num-classes 2 \
  --image-size 256
```

---

## 🙏 Acknowledgements

We would like to thank the authors of the following open-source projects:

- [SAM2](https://github.com/facebookresearch/sam2)
- [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)

Their excellent work has greatly inspired and supported this project.

---

## 📜 Citation

If you find DSIC-Net useful, please cite:

```bibtex
xxxxxxxxxxxxxxxxxxx
```

---

## 🧠 Questions

If you have any questions, feel free to contact: aaaaa@nxxxx
