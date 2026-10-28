"""
Paired statistical comparison of Dice between MCAR-Net and each retrained baseline (Table 2).

For every dataset, region (WT/TC/ET) and baseline:
  * two-sided paired t-test on per-case Dice (same test cases for both models)
    n = number of test cases of that dataset (BraTS 2019: 34, BraTS 2020: 37, BraTS 2021: 125)
  * mean paired difference (MCAR-Net - baseline, in Dice %) with its 95% CI
  * Cohen's d_z = mean difference / SD of the paired differences
  * Benjamini-Hochberg adjustment across all comparisons (5 baselines x 3 regions x 3 datasets = 45)

Input: per-case CSVs written by evaluate.py, named <dataset>_<model>.csv, in --results-dir.

Usage:
    python stats.py --results-dir results --proposed MCAR-Net \
        --baselines "3D U-Net" UNETR Swin-Unet LATUP-Net VcaNet \
        --datasets BraTS2019 BraTS2020 BraTS2021 --out results/table2.csv
"""

import argparse
import csv
import os

import numpy as np
from scipy import stats


def load(path):
    with open(path) as fh:
        return {r["case_id"]: r for r in csv.DictReader(fh)}


def bh_adjust(p):
    """Benjamini-Hochberg adjusted p-values (NaN entries are left as NaN and not counted)."""
    p = np.asarray(p, dtype=float)
    out = np.full(len(p), np.nan)
    ok = ~np.isnan(p)
    q = p[ok]
    m = len(q)
    order = np.argsort(q)
    ranked = q[order] * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1]
    tmp = np.empty(m)
    tmp[order] = np.minimum(adj, 1.0)
    out[ok] = tmp
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--proposed", default="MCAR-Net")
    ap.add_argument("--baselines", nargs="+", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--out", default="results/table2.csv")
    args = ap.parse_args()

    rows = []
    for ds in args.datasets:
        prop = load(os.path.join(args.results_dir, f"{ds}_{args.proposed}.csv"))
        for bl in args.baselines:
            base = load(os.path.join(args.results_dir, f"{ds}_{bl}.csv"))
            if set(prop) != set(base):
                raise ValueError(f"{ds}: {args.proposed} and {bl} were not evaluated on the same cases")
            ids = sorted(prop)
            for region in ("WT", "TC", "ET"):
                a = np.array([float(prop[i][f"dice_{region}"]) for i in ids]) * 100
                b = np.array([float(base[i][f"dice_{region}"]) for i in ids]) * 100
                d = a - b
                n = len(d)
                mean, sd = d.mean(), d.std(ddof=1)
                se = sd / np.sqrt(n)
                tcrit = stats.t.ppf(0.975, n - 1)
                t, p = stats.ttest_rel(a, b)
                rows.append(dict(dataset=ds, baseline=bl, region=region, n=n, df=n - 1,
                                 delta=mean, ci_low=mean - tcrit * se, ci_high=mean + tcrit * se,
                                 d_z=mean / sd if sd > 0 else np.nan, t=t, p=p))

    for r, padj in zip(rows, bh_adjust([r["p"] for r in rows])):
        r["p_BH"] = padj

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    for r in rows:
        print(f"{r['dataset']:10s} {r['baseline']:10s} {r['region']}  n={r['n']:3d}  "
              f"delta {r['delta']:6.2f} [{r['ci_low']:.2f}, {r['ci_high']:.2f}]  "
              f"d_z {r['d_z']:.2f}  p {r['p']:.3f}  p_BH {r['p_BH']:.3f}")
    print(f"{len(rows)} comparisons -> {args.out}")


if __name__ == "__main__":
    main()
