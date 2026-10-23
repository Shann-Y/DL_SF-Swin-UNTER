#!/usr/bin/env python3
"""Visualize inference results: CT slice + ground truth + prediction."""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    EnsureTyped,
)
from monai.data import Dataset, DataLoader

from data.dataset import COVID19Dataset


def visualize(args):
    # Use the validation split, matching inference.py so prediction files line up.
    print(f"Loading data from {args.data_root}...")
    dataset = COVID19Dataset(
        data_root=args.data_root,
        dataset_type="zenodo",
        split="val",
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    samples = [dataset.samples[i] for i in dataset.indices]

    # Same preprocessing as inference.py, so the prediction aligns spatially.
    transforms = Compose(
        [
            LoadImaged(keys=["image", "label"]),
            EnsureChannelFirstd(keys=["image", "label"]),
            Orientationd(keys=["image", "label"], axcodes="RAS"),
            Spacingd(
                keys=["image", "label"],
                pixdim=(1.5, 1.5, 2.0),
                mode=("bilinear", "nearest"),
            ),
            ScaleIntensityRanged(
                keys=["image"],
                a_min=-175,
                a_max=250,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            CropForegroundd(keys=["image", "label"], source_key="image"),
            EnsureTyped(keys=["image", "label"]),
        ]
    )

    ds = Dataset(data=samples, transform=transforms)
    loader = DataLoader(ds, batch_size=1, num_workers=0)

    output_dir = args.output_dir
    pred_dir = args.pred_dir
    os.makedirs(output_dir, exist_ok=True)

    print("Generating visualizations...")

    for i, batch_data in enumerate(loader):
        original_path = samples[i]["image"]
        filename = os.path.basename(original_path).split(".")[0]
        print(f"Processing {filename}...")

        # (1, 1, H, W, D) -> (H, W, D)
        image = batch_data["image"][0, 0].cpu().numpy()
        gt_label = batch_data["label"][0, 0].cpu().numpy()

        pred_path = os.path.join(pred_dir, f"{filename}_pred.nii.gz")
        if not os.path.exists(pred_path):
            print(f"Warning: Prediction file {pred_path} not found")
            continue
        pred = nib.load(pred_path).get_fdata()

        if image.shape != pred.shape:
            print(f"Warning: Shape mismatch for {filename}. Image: {image.shape}, Pred: {pred.shape}")

        # Pick the slice with the largest infection area (fall back to GT).
        infection_pixels = np.sum(pred == 1, axis=(0, 1))
        if np.max(infection_pixels) > 0:
            slice_idx = np.argmax(infection_pixels)
        else:
            gt_pixels = np.sum(gt_label == 1, axis=(0, 1))
            slice_idx = np.argmax(gt_pixels) if np.max(gt_pixels) > 0 else image.shape[2] // 2

        print(f"  Selected slice: {slice_idx} (max infection area)")

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.imshow(np.rot90(image[:, :, slice_idx]), cmap="gray")
        plt.title(f"CT Image (Slice {slice_idx})")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(np.rot90(image[:, :, slice_idx]), cmap="gray")
        gt_mask = np.rot90(gt_label[:, :, slice_idx])
        if np.max(gt_mask) > 0:
            plt.imshow(np.ma.masked_where(gt_mask == 0, gt_mask), cmap="Reds", alpha=0.5, vmin=0, vmax=1)
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(np.rot90(image[:, :, slice_idx]), cmap="gray")
        pred_mask = np.rot90(pred[:, :, slice_idx])
        if np.max(pred_mask) > 0:
            plt.imshow(np.ma.masked_where(pred_mask == 0, pred_mask), cmap="Blues", alpha=0.5, vmin=0, vmax=1)
        plt.title("Prediction")
        plt.axis("off")

        plt.suptitle(f"Case: {filename}", fontsize=16)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{filename}_viz.png")
        plt.savefig(save_path)
        plt.close()
        print(f"  Saved visualization to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data/zenodo")
    parser.add_argument("--pred_dir", type=str, default="./outputs/baseline_swin_unetr/inference_results")
    parser.add_argument("--output_dir", type=str, default="./outputs/baseline_swin_unetr/visualizations")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="must match training")
    parser.add_argument("--seed", type=int, default=42, help="must match training")
    args = parser.parse_args()

    visualize(args)
