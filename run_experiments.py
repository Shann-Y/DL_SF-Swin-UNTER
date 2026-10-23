"""Run the baseline-vs-SF-Gate comparison (the core empirical experiment).

Trains the two models back-to-back with an identical data split, seed and
schedule so the only difference is the SF-Gate:

    1. baseline  : Swin-UNETR           (--no_sf_gate)
    2. proposed  : SF-Swin-UNETR        (default)

Each run writes its own subdirectory under ``--output_dir`` (best_model.pth,
final_model.pth, training_curves.png, TensorBoard logs).

Usage (real data):
    python run_experiments.py --data_root ./data/zenodo --epochs 500

Usage (quick smoke on synthetic data, CPU):
    python run_experiments.py --dataset_type synthetic --spatial_size 64 64 64 \\
        --feature_size 24 --epochs 3 --val_interval 1 --no_ssl_pretrained --no_amp

For an ablation study / robustness beyond this single split, run it again with
a different ``--seed`` (or extend this script to K-fold over seeds).
"""

import argparse
import subprocess
import sys


def build_command(cfg, name, use_sf_gate):
    cmd = [
        sys.executable, "train.py",
        "--data_root", cfg.data_root,
        "--dataset_type", cfg.dataset_type,
        "--spatial_size", *map(str, cfg.spatial_size),
        "--feature_size", str(cfg.feature_size),
        "--batch_size", str(cfg.batch_size),
        "--epochs", str(cfg.max_epochs),
        "--lr", str(cfg.base_lr),
        "--val_interval", str(cfg.val_interval),
        "--val_ratio", str(cfg.val_ratio),
        "--seed", str(cfg.seed),
        "--num_workers", str(cfg.num_workers),
        "--output_dir", cfg.output_dir,
        "--exp_name", name,
    ]
    if not use_sf_gate:
        cmd.append("--no_sf_gate")
    if cfg.no_amp:
        cmd.append("--no_amp")
    if cfg.no_ssl_pretrained:
        cmd.append("--no_ssl_pretrained")
    return cmd


def main():
    parser = argparse.ArgumentParser(description="Baseline vs SF-Gate comparison")
    parser.add_argument("--data_root", type=str, default="./data/zenodo")
    parser.add_argument("--dataset_type", type=str, default="zenodo")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[96, 96, 96])
    parser.add_argument("--feature_size", type=int, default=48)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=500, dest="max_epochs")
    parser.add_argument("--lr", type=float, default=2e-4, dest="base_lr")
    parser.add_argument("--val_interval", type=int, default=10)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--no_ssl_pretrained", action="store_true")
    args = parser.parse_args()

    runs = [
        ("baseline_swin_unetr", False),
        ("sf_swin_unetr", True),
    ]
    for name, use_sf in runs:
        cmd = build_command(args, name, use_sf)
        print(f"\n{'=' * 70}\nRunning: {name}\n{'=' * 70}")
        print(" ".join(cmd), "\n")
        subprocess.run(cmd, check=True)

    print("\nComparison done. Compare per-run Dice in the training logs / TensorBoard:")
    print(f"  baseline : {args.output_dir}/baseline_swin_unetr")
    print(f"  proposed : {args.output_dir}/sf_swin_unetr")


if __name__ == "__main__":
    main()
