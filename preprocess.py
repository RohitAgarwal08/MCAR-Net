"""
Offline preprocessing of BraTS 2019 / 2020 / 2021 into 4-channel 128^3 NIfTI volumes.

This is the cleaned version of the original `dataset_preperation.py`; the numerical
operations are unchanged:

  1. For each patient, the four modalities and the segmentation are located inside the
     patient's own folder and checked to carry that patient's ID (explicit pairing check).
  2. Each modality is min-max normalised to [0, 1] with sklearn's MinMaxScaler applied
     to the volume reshaped as (240*240, 155). MinMaxScaler scales each column
     independently, so the normalisation is effectively per axial slice.
  3. Modalities are stacked in the fixed order FLAIR, T1ce, T2, T1 (channel-last).
  4. A fixed-coordinate crop [56:184, 56:184, 13:141] gives 128 x 128 x 128 x 4.
  5. Label 4 (enhancing tumor) is relabelled to 3 and the mask is one-hot encoded (4 classes).
  6. Volumes are saved as NIfTI with an identity affine.

The BraTS volumes are already co-registered, skull-stripped and resampled to 1 mm
isotropic by the challenge organisers, so no further resampling is required. (The
Orientationd/Spacingd transforms in train.py/evaluate.py therefore leave the data unchanged.)

Data partitioning: the cases are randomly shuffled with a fixed seed (42); 10% (rounded)
are assigned to validation, 10% (rounded) to test and the remainder to training.
This gives 267/34/34 (BraTS 2019, 335 cases), 294/37/37 (BraTS 2020, 368 cases) and
1001/125/125 (BraTS 2021, 1251 cases). The assignment is written to <out-dir>/split.csv.
The same partition is used for MCAR-Net and all comparator models.
Optionally, an existing assignment can be supplied with --split-csv.

BraTS 2020: case BraTS20_Training_355 is skipped automatically because its segmentation
file is distributed under a non-standard name (W39_1998.09.19_Segm.nii).

Usage:
    python preprocess.py --raw-dir /path/to/MICCAI_BraTS2020_TrainingData --out-dir data/brats2020
"""

import argparse
import csv
import glob
import os

import nibabel as nib
import numpy as np
from sklearn.preprocessing import MinMaxScaler

CROP = (slice(56, 184), slice(56, 184), slice(13, 141))
MODALITY_ORDER = ["flair", "t1ce", "t2", "t1"]

# File-name suffixes per dataset family
NAMING = {
    "brats": {"flair": "_flair", "t1ce": "_t1ce", "t2": "_t2", "t1": "_t1", "seg": "_seg"},
    "ucsf":  {"flair": "_FLAIR", "t1ce": "_T1c", "t2": "_T2", "t1": "_T1", "seg": "_tumor_segmentation"},
}


def find_file(case_dir, case_id, suffix):
    """Return the unique file <case_id><suffix>.nii(.gz) inside case_dir, else None."""
    hits = []
    for ext in (".nii", ".nii.gz"):
        hits += glob.glob(os.path.join(case_dir, f"{case_id}{suffix}{ext}"))
    return hits[0] if len(hits) == 1 else None


def discover_cases(raw_dir, naming):
    """Every folder that contains a FLAIR file is treated as one patient folder.

    Returns (cases, skipped): cases = list of (case_id, {modality: path}).
    A case is skipped (and reported) if any of the five files is missing or does not
    carry the folder's patient ID - this is the explicit image/mask pairing check.
    """
    suffixes = NAMING[naming]
    flair_files = sorted(glob.glob(os.path.join(raw_dir, "**", f"*{suffixes['flair']}.nii*"),
                               recursive=True))
    case_dirs = sorted({os.path.dirname(f) for f in flair_files})
    cases, skipped = [], []
    for d in case_dirs:
        case_id = os.path.basename(d.rstrip("/"))
        if naming == "ucsf":  # UCSF-PDGM folders are named <ID>_nifti
            case_id = case_id.replace("_nifti", "")
        files = {k: find_file(d, case_id, s) for k, s in suffixes.items()}
        missing = [k for k, v in files.items() if v is None]
        if missing:
            skipped.append((case_id, missing))
            continue
        cases.append((case_id, files))
    return cases, skipped


def normalise(volume):
    """Exactly the original operation: MinMaxScaler on (H*W, D) -> per-axial-slice [0, 1]."""
    scaler = MinMaxScaler()
    return scaler.fit_transform(volume.reshape(-1, volume.shape[-1])).reshape(volume.shape)


def prepare_case(files):
    mods = [normalise(nib.load(files[m]).get_fdata()) for m in MODALITY_ORDER]
    image = np.stack(mods, axis=3)[CROP]                      # 128 x 128 x 128 x 4

    mask = nib.load(files["seg"]).get_fdata().astype(np.uint8)
    mask[mask == 4] = 3                                       # ET: 4 -> 3
    mask = mask[CROP]
    mask_onehot = np.eye(4, dtype=np.float32)[mask]           # same as keras to_categorical
    return image, mask_onehot


def random_split(case_ids, seed=42, fractions=(0.8, 0.1, 0.1)):
    rng = np.random.RandomState(seed)
    ids = list(case_ids)
    rng.shuffle(ids)
    n = len(ids)
    n_val = int(round(fractions[1] * n))
    n_test = int(round(fractions[2] * n))
    n_train = n - n_val - n_test
    return ({i: "Train" for i in ids[:n_train]}
            | {i: "Val" for i in ids[n_train:n_train + n_val]}
            | {i: "Test" for i in ids[n_train + n_val:]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split-csv", default=None,
                    help="optional CSV with columns case_id,split (Train/Val/Test); default: random 80/10/10 with --seed")
    ap.add_argument("--naming", default="brats", choices=list(NAMING))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude", nargs="*", default=[], help="case IDs to exclude")
    args = ap.parse_args()

    cases, skipped = discover_cases(args.raw_dir, args.naming)
    for cid, missing in skipped:
        print(f"[skip] {cid}: missing/misnamed {missing}")
    cases = [(c, f) for c, f in cases if c not in set(args.exclude)]
    print(f"{len(cases)} complete cases found")

    if args.split_csv:
        with open(args.split_csv) as fh:
            assignment = {r["case_id"]: r["split"] for r in csv.DictReader(fh)}
        unknown = [c for c, _ in cases if c not in assignment]
        cases = [(c, f) for c, f in cases if c in assignment]
        if unknown:
            print(f"[info] {len(unknown)} cases not in split CSV are ignored")
    else:
        assignment = random_split([c for c, _ in cases], seed=args.seed)
        split_path = os.path.join(args.out_dir, "split.csv")
        os.makedirs(args.out_dir, exist_ok=True)
        with open(split_path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["case_id", "split"])
            for c, _ in cases:
                w.writerow([c, assignment[c]])
        print(f"random split written to {split_path}")

    for part in ("Train", "Val", "Test"):
        os.makedirs(os.path.join(args.out_dir, part, "Images"), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, part, "Mask"), exist_ok=True)

    for case_id, files in cases:
        part = assignment[case_id]
        image, mask = prepare_case(files)
        nib.save(nib.Nifti1Image(image, affine=np.eye(4)),
                 os.path.join(args.out_dir, part, "Images", f"image_{case_id}.nii"))
        nib.save(nib.Nifti1Image(mask, affine=np.eye(4)),
                 os.path.join(args.out_dir, part, "Mask", f"mask_{case_id}.nii"))
        print(f"{part:5s} {case_id}")

    counts = {p: sum(1 for c, _ in cases if assignment[c] == p) for p in ("Train", "Val", "Test")}
    print("split sizes:", counts)


if __name__ == "__main__":
    main()
