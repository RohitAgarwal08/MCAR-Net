"""Shared data loading for train.py and evaluate.py (transforms exactly as used in the experiments)."""

import glob
import os

from monai.data import DataLoader, Dataset
from monai.transforms import (
    Compose, EnsureChannelFirstd, LoadImaged, Orientationd, RandAffined, RandFlipd,
    RandScaleIntensityd, RandShiftIntensityd, ScaleIntensityRanged, Spacingd, ToTensord,
)

# Note on intensities: images arrive from preprocess.py already min-max normalised to
# [0, 1]. ScaleIntensityRanged(0..3000 -> 0..255, clip=True) therefore performs no
# clipping; it only multiplies every voxel by the constant 255/3000 (~0.085).
# It is kept because all reported models were trained and evaluated with it.
_common = [
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=3000, b_min=0, b_max=255, clip=True),
]

train_transforms = Compose(_common + [
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    RandAffined(keys=["image", "label"], mode=("bilinear", "nearest"), prob=0.3,
                rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1), padding_mode="border"),
    RandScaleIntensityd(keys="image", factors=0.1, prob=0.5),
    RandShiftIntensityd(keys="image", offsets=0.1, prob=0.5),
    ToTensord(keys=["image", "label"]),
])

eval_transforms = Compose(_common + [ToTensord(keys=["image", "label"])])


def _case_id(path, prefix):
    name = os.path.basename(path)
    for ext in (".nii.gz", ".nii"):
        if name.endswith(ext):
            name = name[: -len(ext)]
    return name[len(prefix):]


def paired_files(split_dir):
    """Pair Images/image_<id> with Mask/mask_<id> by explicit ID match (not by list position)."""
    imgs = sorted(glob.glob(os.path.join(split_dir, "Images", "*.nii*")))
    msks = sorted(glob.glob(os.path.join(split_dir, "Mask", "*.nii*")))
    img_by_id = {_case_id(p, "image_"): p for p in imgs}
    msk_by_id = {_case_id(p, "mask_"): p for p in msks}
    if set(img_by_id) != set(msk_by_id):
        only_i = sorted(set(img_by_id) - set(msk_by_id))
        only_m = sorted(set(msk_by_id) - set(img_by_id))
        raise ValueError(f"Unpaired files in {split_dir}: images only {only_i[:5]}, masks only {only_m[:5]}")
    return [{"image": img_by_id[i], "label": msk_by_id[i], "case_id": i} for i in sorted(img_by_id)]


def make_loader(split_dir, train=False):
    files = paired_files(split_dir)
    ds = Dataset(files, train_transforms if train else eval_transforms)
    return DataLoader(ds, batch_size=1, shuffle=train), files
