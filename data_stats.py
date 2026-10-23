#!/usr/bin/env python3
"""Compute dataset statistics for the Data section of the report.

Reports, per case and in aggregate:
  * volume shape (x, y, z)
  * voxel spacing in mm (from the NIfTI header)
  * HU intensity range of the CT
  * number of infection voxels
  * infection fraction of the total volume, and of the foreground (body)

Usage (from the proj_code directory):
    python data_stats.py --data_root ./data/zenodo
"""

import argparse
import glob
import os

import numpy as np
import nibabel as nib

HU_BODY_THRESHOLD = -175  # matches the lung-window lower bound used in preprocessing


def main():
    p = argparse.ArgumentParser(description="COVID-19-CT-Seg dataset statistics")
    p.add_argument("--data_root", type=str, default="./data/zenodo")
    args = p.parse_args()

    base = os.path.join(args.data_root, "COVID-19-CT-Seg")
    img_dir = os.path.join(base, "COVID-19-CT-Seg_20cases")
    lbl_dir = os.path.join(base, "Infection_Mask")

    images = sorted(
        glob.glob(os.path.join(img_dir, "*.nii.gz")) + glob.glob(os.path.join(img_dir, "*.nii"))
    )
    if not images:
        raise FileNotFoundError(f"No images found under {img_dir!r}")

    rows = []
    for img_path in images:
        stem = os.path.basename(img_path).split(".nii")[0]
        lbl_path = os.path.join(lbl_dir, f"{stem}.nii.gz")
        if not os.path.exists(lbl_path):
            lbl_path = os.path.join(lbl_dir, os.path.basename(img_path))

        img = nib.load(img_path)
        data = img.get_fdata()
        zoom = tuple(float(z) for z in img.header.get_zooms()[:3])

        foreground = int((data >= HU_BODY_THRESHOLD).sum())
        if os.path.exists(lbl_path):
            lbl = nib.load(lbl_path).get_fdata()
            infection = int((lbl > 0).sum())
            frac_total = infection / data.size
            frac_foreground = infection / max(foreground, 1)
        else:
            infection, frac_total, frac_foreground = -1, -1.0, -1.0

        rows.append(
            {
                "name": stem,
                "shape": data.shape,
                "spacing": zoom,
                "hu_min": float(data.min()),
                "hu_max": float(data.max()),
                "infection": infection,
                "frac_total": frac_total,
                "frac_foreground": frac_foreground,
            }
        )
        print(
            f"{stem:34s} shape={data.shape}  spacing={tuple(round(z, 2) for z in zoom)}  "
            f"HU=[{data.min():.0f},{data.max():.0f}]  infection={infection} voxels "
            f"({frac_total * 100:.2f}% of volume / {frac_foreground * 100:.2f}% of foreground)"
        )

    shapes = np.array([r["shape"] for r in rows])
    spacings = np.array([r["spacing"] for r in rows])
    frac_total = [r["frac_total"] for r in rows if r["frac_total"] >= 0]
    frac_fg = [r["frac_foreground"] for r in rows if r["frac_foreground"] >= 0]

    print("\n================ SUMMARY ================")
    print(f"cases: {len(rows)}")
    print(f"shape (x,y,z): min={tuple(shapes.min(axis=0))}  max={tuple(shapes.max(axis=0))}")
    print(
        f"spacing (mm): min={tuple(spacings.min(axis=0).round(2))}  "
        f"max={tuple(spacings.max(axis=0).round(2))}"
    )
    if frac_total:
        print(
            f"infection / total volume (%): mean={np.mean(frac_total) * 100:.2f}  "
            f"min={np.min(frac_total) * 100:.2f}  max={np.max(frac_total) * 100:.2f}"
        )
    if frac_fg:
        print(
            f"infection / foreground  (%): mean={np.mean(frac_fg) * 100:.2f}  "
            f"min={np.min(frac_fg) * 100:.2f}  max={np.max(frac_fg) * 100:.2f}"
        )
    print("=========================================")


if __name__ == "__main__":
    main()
