#!/usr/bin/env python3
"""Run inference with a trained model and save predictions for visualization."""

import os
import sys
import argparse
import numpy as np
import torch
import nibabel as nib
from monai.inferers import sliding_window_inference
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged,
    CropForegroundd,
    EnsureTyped,
    AsDiscrete,
)
from monai.data import Dataset, DataLoader, decollate_batch

from models import create_model
from configs import Config
from data import get_dataloaders


def save_nifti(data, affine, output_path):
    """Save a prediction as a NIfTI file."""
    img = nib.Nifti1Image(data.astype(np.uint8), affine)
    nib.save(img, output_path)
    print(f"Saved: {output_path}")


def inference(args):
    config = Config.from_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Creating model...")
    model = create_model(
        img_size=config.spatial_size,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        feature_size=config.feature_size,
        use_checkpoint=config.use_checkpoint,
        use_sf_gate=not args.no_sf_gate,  # must match the trained model
        sf_gate_reduction=config.sf_gate_reduction,
    )

    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        possible_path = os.path.join(config.exp_dir, "best_model.pth")
        if os.path.exists(possible_path):
            checkpoint_path = possible_path
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading weights from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Use the validation split (unseen cases), with val_ratio/seed read from the
    # checkpoint so they match training.
    from data.dataset import COVID19Dataset
    val_ratio = checkpoint.get("config", {}).get("val_ratio", config.val_ratio)
    seed = checkpoint.get("config", {}).get("seed", config.seed)
    val_dataset = COVID19Dataset(
        data_root=config.data_root,
        dataset_type=config.dataset_type,
        split="val",
        val_ratio=val_ratio,
        seed=seed,
    )
    inference_files = [val_dataset.samples[i] for i in val_dataset.indices]
    print(f"Found {len(inference_files)} validation samples for inference "
          f"(val_ratio={val_ratio}, seed={seed})")

    # Same preprocessing as validation (no augmentation).
    val_transforms = Compose(
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

    inf_ds = Dataset(data=inference_files, transform=val_transforms)
    inf_loader = DataLoader(inf_ds, batch_size=1, num_workers=1)

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=config.out_channels)])

    output_dir = os.path.join(config.exp_dir, "inference_results")
    os.makedirs(output_dir, exist_ok=True)

    print("Starting inference...")
    with torch.no_grad():
        for i, batch_data in enumerate(inf_loader):
            inputs = batch_data["image"].to(device)

            original_path = inference_files[i]["image"]
            filename = os.path.basename(original_path).split(".")[0]
            print(f"Processing {i+1}/{len(inference_files)}: {filename}")

            outputs = sliding_window_inference(
                inputs,
                roi_size=config.spatial_size,
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
            )

            pred = torch.argmax(outputs, dim=1).detach().cpu().numpy()[0]

            # The prediction is saved in the preprocessed space (RAS, 1.5x1.5x2.0,
            # cropped), which is what visualize.py expects.
            save_path = os.path.join(output_dir, f"{filename}_pred.nii.gz")
            try:
                affine = inputs.affine[0].cpu().numpy()
            except Exception:
                affine = np.eye(4)
            save_nifti(pred, affine, save_path)

    print(f"Inference completed. Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for COVID-19 CT segmentation")
    parser.add_argument("--data_root", type=str, required=True, help="Data root directory")
    parser.add_argument("--checkpoint", type=str, default="outputs/baseline_swin_unetr/best_model.pth", help="Path to checkpoint")
    parser.add_argument("--exp_name", type=str, default="baseline_swin_unetr", help="Experiment name")
    parser.add_argument("--no_sf_gate", action="store_true", help="Disable SF-Gate (must match training)")

    # Defaults for the remaining Config fields, to avoid missing-arg errors.
    parser.add_argument("--dataset_type", type=str, default="zenodo")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--use_sf_gate", action="store_true", default=True)

    args = parser.parse_args()

    if isinstance(args.spatial_size, list):
        args.spatial_size = tuple(args.spatial_size)

    inference(args)
