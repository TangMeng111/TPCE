# Target-Private Categories Expansion for Cross-Domain Few-Shot Medical Image Segmentation

**Meng Tang and Yunyun Yang**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.8.20](https://img.shields.io/badge/Python-3.8.20-blue.svg)](https://www.python.org/)
[![PyTorch 2.4.1](https://img.shields.io/badge/PyTorch-2.4.1-red.svg)](https://pytorch.org/)
[![CUDA 12.1](https://img.shields.io/badge/CUDA-12.1-green.svg)](https://developer.nvidia.com/cuda-toolkit)

## Overview

A dual-branch structural prototype framework for cross-domain few-shot medical image segmentation. The shared branch learns from source annotations; the novel branch learns from target supervoxel pseudo-tasks. Their predictions are fused into a joint multi-class segmentation.

![Overall framework](method_framework.png)

This package covers the **abdominal main model**: CHAOS T2-SPIR MRI to Synapse/SABS CT.

## Installation

Run all commands from the project root on Linux.

```bash
conda create -n abdomen-main python=3.8.20 pip -y
conda activate abdomen-main
python -m pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
```

### COCO Initialization

Torchvision automatically downloads the [COCO initialization weights](https://download.pytorch.org/models/deeplabv3_resnet101_coco-586e9e4e.pth) on first training. For offline use, place the file in `torch.hub.get_dir()/checkpoints/`, or `$TORCH_HOME/hub/checkpoints/` if `TORCH_HOME` is set. See the [official model documentation](https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.segmentation.deeplabv3_resnet101.html).

## Data Preparation

| Dataset | Role | Download |
|---|---|---|
| CHAOS MR | Source: 20 T2-SPIR volumes with four-organ annotations | [CHAOS Challenge](https://chaos.grand-challenge.org/) |
| Synapse / SABS CT | Target: 30 annotated abdominal CT volumes | [Synapse](https://www.synapse.org/Synapse:syn3193805/wiki/89480) |

Organize the raw data outside the repository:

```text
/path/to/CHAOS/MR/<patient_id>/T2SPIR/
  DICOM_anon/                         # DICOM images
  Ground/                            # PNG annotations
/path/to/Synapse/img/imgNNNN.nii.gz
/path/to/Synapse/label/labelNNNN.nii.gz
```

Convert and preprocess the images, then generate target supervoxels:

```bash
python -B -m data.prepare chaos-dicom --mr-root /path/to/CHAOS/MR \
  --out outputs/preprocessing/chaos_converted
python -B -m data.prepare chaos \
  --images outputs/preprocessing/chaos_converted --labels outputs/preprocessing/chaos_converted \
  --out data/chaos
python -B -m data.prepare synapse --images /path/to/Synapse/img \
  --labels /path/to/Synapse/label --out data/synapse
python -B data/preprocessing/felzenszwalb_3d/setup.py build_ext --inplace
python -B -m data.generate_supervoxels --data-root data/synapse \
  --out data/synapse/supervoxels_felzenszwalb_5000
```

Use the complete datasets to preserve case numbering. Synapse IDs 0–29 follow lexicographic filename order. Settings are recorded in `data/config.json`; Felzenszwalb's `min_size=5000` is a small-region merge threshold in voxels.

## Training

```bash
python -B -m train shared --gpu 5 --out outputs/training/shared
python -B -m train novel --gpu 5 \
  --init-checkpoint outputs/training/shared/checkpoints/epoch_005.pth \
  --out outputs/training/novel
```

Both shot settings reuse the same pair of checkpoints. Training configurations are in `train/configs/shared.json` and `train/configs/novel.json`.

`--gpu` selects a physical GPU; examples use GPU5.
## Evaluation

```bash
python -B -m evaluation run --shots 1 --gpu 5 \
  --shared-checkpoint outputs/training/shared/checkpoints/epoch_005.pth \
  --novel-checkpoint outputs/training/novel/checkpoints/epoch_100.pth \
  --out outputs/evaluation/full
python -B -m evaluation run --shots 5 --gpu 5 \
  --shared-checkpoint outputs/training/shared/checkpoints/epoch_005.pth \
  --novel-checkpoint outputs/training/novel/checkpoints/epoch_100.pth \
  --out outputs/evaluation/full
# Optional: regenerate the five-fold summary
python -B -m evaluation summarize --shots 1,5 --out outputs/evaluation/full
```

Each run evaluates only the selected shot setting. Five-fold results are summarized automatically; use `--folds 0,1` to run selected folds.

Per-fold predictions and FPD tables are saved under `fold_N/Kshot/attempt_NNN/`. `summary.json` contains group means and sample standard deviations; `per_class.csv` contains per-fold, per-class FPD.

**Fused Pos-Dice (FPD)** measures class-wise Dice from joint predictions on ground-truth-positive slices, macro-averaged within each category group and then equally across the five folds. Evaluation settings are in `evaluation/configs/`.

## Abdominal Results

Five-fold mean FPD on a 0–1 scale:

| Setting | Shared FPD | Novel FPD | All FPD |
|---|---:|---:|---:|
| 1-shot | 0.73815 | 0.36712 | **0.48128** |
| 5-shot | 0.73109 | 0.39053 | **0.49532** |

## Project Structure

```text
data/          # Preprocessing, supervoxels and data loading
model/         # Network definitions
train/         # Shared/novel training and configurations
evaluation/    # Joint prediction and five-fold FPD evaluation
utils/         # I/O, seeds, support processing, thresholds and refinement
outputs/       # Generated checkpoints, predictions and reports
```

> **Code coming soon.** The code will be publicly released upon acceptance of the paper.

## Acknowledgements

We thank the CHAOS and Synapse providers, PyTorch/torchvision, scikit-image, and the medical few-shot segmentation projects referenced in the source code.

## License

Original project code is licensed under the [MIT License](LICENSE).
Third-party supervoxel code retains its [upstream license notices](data/preprocessing/felzenszwalb_3d/LICENSE).
