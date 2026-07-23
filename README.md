<div align="center">
  <h1>🔎 DSIC-Net: A Dual-Space Information Cascaded Correction Network for Brain Age Prediction from Structural MRI</h1>
</div>

> Tianyu Sun, Xinyao Zhao, Zhaoqin Huang, Ximing Wang, Cui-Na Jiao, Zehua Zhang, Ning Zhao, Hao Zhou, Junhao Nie, Xuan Wang, Qiang Zheng, Tong Zhang</br>
> *Medical Image Analysis (MedIA)*, 2026
> 
---
![Graph Abstract](./figure/GraphAbstracts.svg)
## 👀 Overview

DSIC-Net is a two-stage framework for brain age prediciton using brain T1-weighted Magnetic Resonance Imaging （MRI） scans.

1. **Image Space Net**  
   Stage 1 learns age-sensitive representations directly from the full 3D T1-weighted sMRI volume.

2. **Bounded Graph-Guided Residual Correction Net**  
   Stage 2 corrected brain age prediciton predicted from stage 1 using brain net information. 

DSIC-Net was validated on 10 diverse public datasets, achieving an average Dice only **3.08%** lower than the fully supervised baseline.

---

## Repository layout

~~~text
.
├── src/dsic_net/
│   ├── models/
│   │   ├── image_space_net.py
│   │   ├── graph_correction_net.py
│   │   └── dsic_net.py
│   ├── data.py
│   ├── demo_data.py
│   └── training.py
├── scripts/
│   ├── make_demo_data.py
│   ├── validate_data.py
│   ├── run_demo.py
│   ├── train_stage1.py
│   ├── train_stage2.py
│   └── predict.py
├── data/demo/
├── tests/
├── .github/workflows/ci.yml
└── pyproject.toml
~~~

## Installation

Python 3.10 or newer is required. A GPU is optional for the demo but recommended for
full-resolution training.

~~~bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip

# CPU-only PyTorch:
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
~~~

For CUDA, install the PyTorch build appropriate for the local driver first, then run
the editable-install command.

## Run the verified example

The repository already contains six deterministic synthetic subjects (4 train, 1 val,
1 test). Recreate them, validate every NIfTI/NPZ file, and run one end-to-end forward
pass:

~~~bash
python scripts/make_demo_data.py --overwrite
python scripts/validate_data.py --manifest data/demo/manifest.csv
python scripts/run_demo.py --manifest data/demo/manifest.csv --device cpu
~~~

The prediction is written to outputs/demo_predictions.csv. Because the demo uses
randomly initialized weights, the predicted ages are not meaningful. The command is
successful when it reports status=ok and correction_bound_satisfied=true.

## Train the two stages

The following commands are deliberately small enough for a software smoke test:

~~~bash
python scripts/train_stage1.py \
  --manifest data/demo/manifest.csv \
  --epochs 1 \
  --device cpu \
  --output outputs/stage1.pt

python scripts/train_stage2.py \
  --manifest data/demo/manifest.csv \
  --stage1-checkpoint outputs/stage1.pt \
  --epochs 1 \
  --device cpu \
  --output outputs/dsic_net.pt

python scripts/predict.py \
  --manifest data/demo/manifest.csv \
  --checkpoint outputs/dsic_net.pt \
  --split test \
  --device cpu \
  --output outputs/predictions.csv
~~~

For real experiments, increase the input resolution, channel widths, epochs, and batch
size according to available memory. Stage 1 is frozen during the provided Stage-2
training routine so its output remains an explicit anchor, matching the cascaded
prediction-and-correction design.

## Data format

Use a UTF-8 CSV file with one row per subject:

| Column     | Meaning                                                      |
| ---------- | ------------------------------------------------------------ |
| subject_id | Unique subject identifier; duplicates across splits are rejected |
| image_path | Path to a preprocessed 3D NIfTI image, relative to the manifest |
| graph_path | Path to an NPZ R2SN file, relative to the manifest           |
| age        | Chronological age in years                                   |
| sex        | Binary metadata field (0 or 1); retained for cohort auditing |
| split      | train, val, or test                                          |

Each graph NPZ must contain:

- node_features: float array with shape (N, d), one radiomics descriptor per ROI;
- adjacency: symmetric float array with shape (N, N).

All subjects must use the same N and d. Self-loops are added inside the model, so the
stored adjacency may have a zero diagonal. The loader z-scores nonzero voxels and
resamples each volume to the requested image_size. For paper experiments, keep the
same preprocessing, atlas, radiomics extraction, R2SN construction, and target size
across all cohorts.

Validate a real manifest before training:

~~~bash
python scripts/validate_data.py --manifest /path/to/manifest.csv
~~~

The validator checks required columns, labels, split leakage, readable 3D NIfTI
volumes, finite values, graph dimensions, and adjacency symmetry.

## Tests and quality checks

~~~bash
pytest
ruff check .
python -m compileall -q src scripts tests
~~~

Tests cover tensor shapes, gradients, normalized adjacency, Methods Eq. (12), the
strict graph-correction bound, and the complete data loader.

## Reproducibility notes

- The source archive's six NIfTI files were truncated and could not be decompressed.
  They were replaced in this clean repository by generated, valid synthetic images.
- The legacy XLS/XLSX cohort tables used abbreviated columns and included rows for
  data not present in the archive. The runnable example therefore uses an explicit,
  portable CSV manifest.
- The supplied Methods excerpt refers to R2SN construction in Section 2.2.2, which
  was not included. This repository consumes precomputed R2SN node features and
  adjacency matrices; it does not invent a paper preprocessing pipeline.
- The training loss and optimizer are practical defaults because the supplied excerpt
  specifies the architecture but not the optimization protocol. Replace these values
  with the final manuscript settings before reporting results.
- No pretrained weights or original study images are distributed.

## Citation and license

Add the manuscript citation, authors, DOI, and a CITATION.cff file once the paper
metadata are final. No open-source license was supplied with the input material; see
LICENSE-NOTICE.md before publishing.

---

## Questions

If you have any questions, feel free to contact: aaaaa@nxxxx
