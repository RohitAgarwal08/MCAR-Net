"""
Test-set evaluation producing the WT / TC / ET metrics reported in Table 1.

Procedure (as used for the reported results):
  * the best-validation-IoU checkpoint is applied to every test case
  * prediction = argmax over the 4 output channels (no thresholding, no post-processing)
  * BraTS regions (labels after 4 -> 3 relabelling):
        WT = {1, 2, 3},  TC = {1, 3},  ET = {3}
  * Dice per case and region. If prediction and ground truth are both empty for a
    region, Dice = 1.
  * HD95 per case and region with medpy.metric.binary.hd95 on the 1 mm isotropic grid
    (values in mm). If both are empty, HD95 = 0.
    If exactly one of them is empty, HD95 is undefined (there is no boundary to measure
    from). This script then assigns the largest distance that can occur inside the
    evaluated volume (the volume diagonal, see max_distance_mm) and reports how often it
    happened. This situation did not occur for MCAR-Net in any of the reported test sets.
  * Reported values are the means over test cases.
  * Jaccard index: computed from the mean Dice with the Dice-Jaccard identity
    J = D / (2 - D).

A per-case CSV is written so that the statistical tests (stats.py) can be reproduced.

Usage:
    python evaluate.py --data-dir data/brats2020 --checkpoint runs/brats2020/best.pth \
                       --dataset BraTS2020 --model-name MCAR-Net --out-dir results
"""

import argparse
import csv
import os

import numpy as np
import torch
from medpy.metric import binary
from tqdm import tqdm

from data import make_loader
from model import ABLATION_VARIANTS, build_model

REGIONS = {"WT": (1, 2, 3), "TC": (1, 3), "ET": (3,)}
VOXEL_SPACING_MM = (1.0, 1.0, 1.0)


def max_distance_mm(shape, spacing=VOXEL_SPACING_MM):
    """Largest possible distance between two voxel centres inside the evaluated volume.

    Voxel indices along each axis run from 0 to (size - 1), so the farthest pair of
    voxel centres is separated by (size - 1) * spacing along every axis. The distance is
    the length of the volume diagonal:

        sqrt( sum_k ((size_k - 1) * spacing_k)^2 )

    For the 128 x 128 x 128 volumes at 1 mm used here:
        sqrt(3 * 127^2) = 127 * sqrt(3) = 219.97 mm

    No HD95 value can exceed this, so it is used as the worst-case HD95 when exactly
    one of the prediction / ground truth masks is empty.
    """
    return float(np.sqrt(sum(((s - 1) * d) ** 2 for s, d in zip(shape, spacing))))


def region_masks(lab):
    return {r: np.isin(lab, v) for r, v in REGIONS.items()}


def dice(p, g):
    s = p.sum() + g.sum()
    if s == 0:
        return 1.0
    return 2.0 * np.logical_and(p, g).sum() / s


def jaccard_from_dice(d):
    """Jaccard index from Dice via the identity J = D / (2 - D)."""
    return d / (2.0 - d)


def hd95(p, g):
    if not p.any() and not g.any():
        return 0.0, False
    if not p.any() or not g.any():
        return max_distance_mm(p.shape), True
    return float(binary.hd95(p, g, voxelspacing=VOXEL_SPACING_MM)), False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True, help="folder with Test/ (or the split given by --split)")
    ap.add_argument("--split", default="Test")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--variant", default="mcar_net", choices=list(ABLATION_VARIANTS))
    ap.add_argument("--dataset", required=True, help="label used in output file names, e.g. BraTS2020")
    ap.add_argument("--model-name", default="MCAR-Net")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.variant).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    loader, files = make_loader(os.path.join(args.data_dir, args.split))
    rows, undefined = [], 0
    with torch.no_grad():
        for batch in tqdm(loader):
            out = model(batch["image"].to(device))
            pred = torch.argmax(out, dim=1)[0].cpu().numpy()
            gt = torch.argmax(batch["label"], dim=1)[0].numpy()
            pm, gm = region_masks(pred), region_masks(gt)
            row = {"case_id": batch["case_id"][0]}
            for r in REGIONS:
                row[f"dice_{r}"] = dice(pm[r], gm[r])
                row[f"hd95_{r}"], und = hd95(pm[r], gm[r])
                undefined += int(und)
            rows.append(row)

    os.makedirs(args.out_dir, exist_ok=True)
    out_csv = os.path.join(args.out_dir, f"{args.dataset}_{args.model_name}.csv")
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"{args.model_name} on {args.dataset} {args.split} (n = {len(rows)})")
    for r in REGIONS:
        d = np.mean([x[f"dice_{r}"] for x in rows])
        h = np.mean([x[f"hd95_{r}"] for x in rows])
        print(f"  {r}: Dice {100 * d:.2f}%  Jaccard {100 * jaccard_from_dice(d):.2f}%  HD95 {h:.2f} mm")
    print(f"  HD95 undefined (exactly one mask empty): {undefined} case-regions")
    print(f"per-case results -> {out_csv}")


if __name__ == "__main__":
    main()
