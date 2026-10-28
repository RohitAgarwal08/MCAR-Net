# MCAR-Net: A Lightweight Multi-scale Context-Aware Residual Network for Brain Tumor Segmentation

Code for the manuscript *"A Lightweight Multi-scale Context-Aware Residual (MCAR) Network for
Brain Tumor Segmentation"* (Molecular Imaging and Biology, under revision).

| Manuscript item | Script |
|---|---|
| Architecture (Sec. 3, Fig. 1) | `model.py` |
| Parameters and MACs (Sec. 5.1): 6.57 M, 141.91 GMac | `complexity.py` |
| Preprocessing and data partitioning (Sec. 4, 4.1) | `preprocess.py`, `data.py` |
| Training configuration (Sec. 5.2) | `train.py` |
| WT / TC / ET Dice, Jaccard, HD95 (Table 1) | `evaluate.py` |
| Paired t-tests, 95% CI, Cohen's d_z, Benjamini-Hochberg (Table 2) | `stats.py` |
| Component-wise and loss-function ablations (Table 3) | `train.py --variant ...`, `--lambda-dice/--lambda-ce` |

## Installation

```bash
pip install -r requirements.txt   # Python 3.12, PyTorch 2.6, MONAI 1.4.0
```

`ptflops` is pinned to 0.7.4. From version 0.7.5 onwards its default counting backend changed and
it reports a lower MAC count (~127.8 GMac) for the same network.

## 1. Preprocessing and data partitioning

```bash
python preprocess.py --raw-dir <MICCAI_BraTS2020_TrainingData> --out-dir data/brats2020
```

For every patient, the four modalities and the segmentation are read from the patient's own
folder, and each file name is checked to carry that patient's ID. Cases with a missing or
misnamed file are reported and skipped (in BraTS 2020 this applies to `BraTS20_Training_355`,
whose segmentation is distributed as `W39_1998.09.19_Segm.nii`). Then:

1. each modality is min-max normalised to [0, 1] with `sklearn.preprocessing.MinMaxScaler`
   applied to the volume reshaped as (240·240, 155), i.e. independently for each axial slice;
2. the modalities are stacked in the order FLAIR, T1ce, T2, T1;
3. a fixed-coordinate crop [56:184, 56:184, 13:141] gives a 128 × 128 × 128 × 4 input;
4. label 4 (ET) is relabelled to 3 and the labels are one-hot encoded (4 classes).

The BraTS volumes are distributed co-registered to a common template at 1 mm isotropic
resolution. In the loading pipeline (`data.py`), MONAI `Orientationd(RAS)` and `Spacingd(1 mm)`
ensure a consistent orientation and spacing, and `ScaleIntensityRanged([0, 3000] → [0, 255])`
is applied. Since the normalised intensities lie in [0, 1], the latter performs no clipping and
multiplies all intensities by the constant 255/3000.

**Partitioning:** the cases are shuffled with seed 42; 10% (rounded) go to validation, 10%
(rounded) to test and the rest to training: BraTS 2019 267/34/34, BraTS 2020 294/37/37,
BraTS 2021 1001/125/125. The same partitions are used for MCAR-Net and all comparator models.

**Augmentation (training only):** random flips along each axis (p = 0.5), random affine
(p = 0.3; rotation ±0.1 rad, scaling 0.9–1.1), random intensity scaling (p = 0.5, factor 0.9–1.1)
and shifting (p = 0.5, offset ±0.1).

## 2. Training

```bash
python train.py --data-dir data/brats2020 --out-dir runs/brats2020 --epochs 200
```

* batch size 1, whole 128³ volumes, seed 42
* loss: MONAI `DiceCELoss(softmax=True)` with equal Dice and cross-entropy weights (λ = 0.5)
* AdamW (lr 1e-2, weight decay 1e-5, β = (0.9, 0.999), ε = 1e-8)
* `ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-7)` stepped on the validation loss with
  `mode='max'`, the setting used for all reported models. As the validation loss decreases,
  this halves the learning rate every four epochs until 1e-7 (a step decay).
  `--scheduler-mode min` gives the conventional plateau behaviour.
* early stopping after 25 epochs without improvement of the validation IoU (`--patience 25`);
  epoch limits: BraTS 2019 = 300, BraTS 2020 = 200, BraTS 2021 = 100
* `best.pth` (highest validation IoU) is used for testing. The validation IoU is a monitoring
  quantity only; it is not used for the reported results.
* `last.pth` stores the full training state; `--resume runs/brats2020/last.pth` continues an
  interrupted run.

Ablations (Table 3):

| Table 3 row | option |
|---|---|
| Baseline (encoder–decoder) | `--variant baseline` |
| + AMCF (channel attention only) | `--variant amcf_channel` |
| + AMCF (spatial attention only) | `--variant amcf_spatial` |
| + AMCF (channel + spatial) | `--variant amcf` |
| + AMCF + CGSE (SE only) | `--variant amcf_cgse_se` |
| + AMCF + CGSE (gated attention only) | `--variant amcf_cgse_ga` |
| + AMCF + CGSE (GA + SE) = MCAR-Net | `--variant mcar_net` (default) |
| Dice loss only / CE only / λ | `--lambda-dice 1 --lambda-ce 0` / `--lambda-dice 0 --lambda-ce 1` / `--lambda-dice λ --lambda-ce 1-λ` |

## 3. Evaluation

```bash
python evaluate.py --data-dir data/brats2020 --checkpoint runs/brats2020/best.pth \
                   --dataset BraTS2020 --model-name MCAR-Net --out-dir results
```

* prediction = argmax over the four output channels, no post-processing
* WT = labels {1, 2, 3}, TC = {1, 3}, ET = {3}
* Dice and HD95 (`medpy.metric.binary.hd95`, mm) per case and region, averaged over the test
  cases; Jaccard index = D / (2 − D) from the mean Dice
* both masks empty → Dice = 1, HD95 = 0; exactly one mask empty → HD95 = largest possible
  distance in the 128³ volume, 127·√3 = 219.97 mm (did not occur for MCAR-Net in the reported
  test sets)
* writes `results/<dataset>_<model>.csv` with per-case values

## 4. Statistics (Table 2)

```bash
python stats.py --results-dir results --proposed MCAR-Net \
    --baselines "3D U-Net" UNETR Swin-Unet LATUP-Net VcaNet \
    --datasets BraTS2019 BraTS2020 BraTS2021
```

Two-sided paired t-test on the per-case Dice of each test set (n = 34, 37, 125), mean paired
difference with 95% CI, Cohen's d_z, Benjamini-Hochberg adjustment over the 45 comparisons.

## 5. Complexity

```bash
python complexity.py   # MCAR-Net: 6.57 M parameters, 141.91 GMac (4 × 128³ input)
```

## Data

BraTS 2019, 2020 and 2021 are available from the challenge organisers and UCSF-PDGM from The
Cancer Imaging Archive. The data are not redistributed here.
