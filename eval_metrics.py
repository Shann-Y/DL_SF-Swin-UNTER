#!/usr/bin/env python3
"""Post-hoc evaluation: Dice + 95% Hausdorff distance on the validation split.

No retraining needed: loads a saved best_model.pth, runs sliding-window inference
over the *validation* cases (same split/seed as training), and reports Dice and HD95.

Usage (from the proj_code directory, on the GPU machine):

    # SF-Swin-UNETR  (use_sf_gate is read from the checkpoint config automatically)
    python eval_metrics.py --checkpoint ./outputs/sf_swin_unetr/best_model.pth

    # baseline Swin-UNETR  (override, in case the checkpoint lacks the flag)
    python eval_metrics.py --checkpoint ./outputs/baseline_swin_unetr/best_model.pth --no_sf_gate
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import torch
from monai.data import decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.transforms import AsDiscrete, Compose

from data import get_dataloaders
from models import create_model


def main():
    p = argparse.ArgumentParser(description="Compute Dice + HD95 on the validation split")
    p.add_argument("--checkpoint", type=str, required=True, help="path to best_model.pth")
    p.add_argument("--data_root", type=str, default=None, help="override data_root from the checkpoint")
    p.add_argument("--no_sf_gate", action="store_true", help="force baseline (disable SF-Gate)")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config", {}) or {}

    # ---- model config (from the checkpoint; sensible defaults as fallback) ----
    spatial_size = tuple(cfg.get("spatial_size", (96, 96, 96)))
    feature_size = cfg.get("feature_size", 48)
    in_channels = cfg.get("in_channels", 1)
    out_channels = cfg.get("out_channels", 2)
    use_checkpoint = cfg.get("use_checkpoint", False)
    use_sf_gate = cfg.get("use_sf_gate", True) and not args.no_sf_gate
    sf_gate_reduction = cfg.get("sf_gate_reduction", 16)

    # ---- data config ----
    data_root = args.data_root or cfg.get("data_root", "./data/zenodo")
    dataset_type = cfg.get("dataset_type", "zenodo")
    val_ratio = cfg.get("val_ratio", 0.2)
    seed = cfg.get("seed", 42)
    num_workers = cfg.get("num_workers", 4)

    print(f"[eval] checkpoint: {args.checkpoint}")
    print(f"[eval] model: feature_size={feature_size}, use_sf_gate={use_sf_gate}, spatial={spatial_size}")
    print(f"[eval] data: {data_root} (val_ratio={val_ratio}, seed={seed})")

    model = create_model(
        img_size=spatial_size,
        in_channels=in_channels,
        out_channels=out_channels,
        feature_size=feature_size,
        use_checkpoint=use_checkpoint,
        use_sf_gate=use_sf_gate,
        sf_gate_reduction=sf_gate_reduction,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    _, val_loader = get_dataloaders(
        data_root=data_root,
        dataset_type=dataset_type,
        batch_size=1,
        spatial_size=spatial_size,
        num_workers=num_workers,
        val_ratio=val_ratio,
        seed=seed,
    )
    print(f"[eval] validation cases: {len(val_loader)}")

    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=out_channels)])
    post_label = Compose([AsDiscrete(to_onehot=out_channels)])
    dice_metric = DiceMetric(include_background=False, reduction="mean")
    # HD95 in *voxel* units of the resampled grid (spacing 1.5 x 1.5 x 2.0 mm).
    # Both models share the same grid, so this is a fair head-to-head comparison.
    hd95_metric = HausdorffDistanceMetric(percentile=95, include_background=False, reduction="mean")

    per_case_dice = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            out = sliding_window_inference(
                x, roi_size=spatial_size, sw_batch_size=4, predictor=model, overlap=0.5
            )

            out_conv = [post_pred(o) for o in decollate_batch(out)]
            y_conv = [post_label(l) for l in decollate_batch(y)]
            dice_metric(y_pred=out_conv, y=y_conv)
            hd95_metric(y_pred=out_conv, y=y_conv)

            # per-case foreground Dice for the log (batch size is 1)
            pred = out.argmax(dim=1)
            tgt = y.squeeze(1).round().long()
            inter = (pred * tgt).sum().float()
            denom = pred.sum().float() + tgt.sum().float() + 1e-6
            per_case_dice.append((2.0 * inter / denom).item())
            print(f"  case {i + 1}/{len(val_loader)}: Dice = {per_case_dice[-1]:.4f}")

    mean_dice = dice_metric.aggregate().item()
    mean_hd95 = hd95_metric.aggregate().item()

    print("\n================ RESULTS ================")
    print(f"per-case Dice: {[f'{d:.4f}' for d in per_case_dice]}")
    print(f"Mean Dice: {mean_dice:.4f}  ({mean_dice * 100:.2f}%)")
    print(f"Mean HD95: {mean_hd95:.4f}  (voxel units; resampled grid 1.5 x 1.5 x 2.0 mm)")
    print("=========================================")


if __name__ == "__main__":
    main()
