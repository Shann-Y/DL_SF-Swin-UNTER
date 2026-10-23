"""Zero-data smoke test for the SF-Swin-UNETR pipeline.

Runs entirely on synthetic data (no Zenodo download needed) and checks, in
order, that:

  1. the model builds and does forward/backward (both SF and baseline),
  2. the dataloaders produce correctly-shaped ``{"image","label"}`` batches,
  3. the model can *overfit* a single tiny sample (loss -> 0, Dice -> 1),
     which proves the training loop, loss, optimizer and gradients all work,
  4. checkpoints round-trip (save / reload).

Usage:
    python smoke_test.py
    python smoke_test.py --spatial_size 64 64 64 --feature_size 24 --overfit_steps 40

CPU is fine: everything here uses tiny volumes and a small backbone.
"""

import argparse
import os
import sys
import tempfile
from pathlib import Path

# Make the project package importable when run as `python smoke_test.py`.
sys.path.insert(0, str(Path(__file__).parent))

import torch

from data import get_dataloaders
from models import create_model
from train import get_loss_function, get_optimizer, set_seed


def check(cond: bool, msg: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        sys.exit(1)


def part1_model(device, spatial_size, feature_size):
    print("\n=== Part 1: model forward / backward ===")
    x = torch.randn(1, 1, *spatial_size, device=device)

    for name, use_sf in [("SF-Swin-UNETR", True), ("baseline Swin-UNETR", False)]:
        model = create_model(
            img_size=spatial_size,
            in_channels=1,
            out_channels=2,
            feature_size=feature_size,
            use_checkpoint=False,
            use_sf_gate=use_sf,
            sf_gate_reduction=16,
        ).to(device)
        y = model(x)
        check(tuple(y.shape) == (1, 2, *spatial_size), f"{name}: output shape {tuple(y.shape)}")
        y.sum().backward()
        grads_ok = any(p.grad is not None for p in model.parameters() if p.requires_grad)
        check(grads_ok, f"{name}: backward produced gradients")
    print("Part 1 done.")


def part2_dataloaders(spatial_size):
    print("\n=== Part 2: synthetic dataloaders ===")
    train_loader, val_loader = get_dataloaders(
        data_root="./data/zenodo",  # ignored for synthetic
        dataset_type="synthetic",
        batch_size=2,
        spatial_size=spatial_size,
        num_workers=0,
        val_ratio=0.25,
        seed=42,
    )
    check(len(train_loader) > 0, "train_loader is non-empty")
    check(len(val_loader) > 0, "val_loader is non-empty")

    batch = next(iter(train_loader))
    check("image" in batch and "label" in batch, "batch has 'image' and 'label' keys")
    check(
        tuple(batch["image"].shape) == (2, 1, *spatial_size),
        f"train batch image shape {tuple(batch['image'].shape)}",
    )
    check(
        batch["label"].min() >= 0 and batch["label"].max() <= 1,
        "label values are in [0, 1]",
    )
    print("Part 2 done.")


def part3_overfit(device, spatial_size, feature_size, steps):
    print(f"\n=== Part 3: overfit one synthetic sample ({steps} steps) ===")
    # A single sample -> if the loop is wired correctly the model memorizes it.
    train_loader, _ = get_dataloaders(
        data_root="./data/zenodo",
        dataset_type="synthetic",
        batch_size=1,
        spatial_size=spatial_size,
        num_workers=0,
        val_ratio=0.5,  # 1 sample -> 1 train / 1 val
        seed=0,
    )
    set_seed(42, deterministic=False)
    model = create_model(
        img_size=spatial_size,
        in_channels=1,
        out_channels=2,
        feature_size=feature_size,
        use_checkpoint=False,
        use_sf_gate=True,
        sf_gate_reduction=16,
    ).to(device)

    # A minimal Config stand-in with the fields get_loss_function/get_optimizer read.
    class Cfg:
        pass

    cfg = Cfg()
    cfg.loss_type = "dice_ce"
    cfg.use_sf_gate = True
    cfg.sf_gate_lr_multiplier = 10.0
    cfg.base_lr = 1e-3
    cfg.weight_decay = 1e-5
    cfg.optimizer = "adamw"

    loss_fn = get_loss_function(cfg)
    optimizer = get_optimizer(model, cfg)

    batch = next(iter(train_loader))
    x = batch["image"].to(device)
    y = batch["label"].to(device)

    def binary_dice(pred_logits: torch.Tensor, target: torch.Tensor) -> float:
        """Foreground Dice from raw logits + a 0/1 integer mask."""
        pred = pred_logits.argmax(dim=1)
        tgt = target.squeeze(1).round().long()
        inter = (pred * tgt).sum().float()
        denom = pred.sum().float() + tgt.sum().float() + 1e-6
        return (2.0 * inter / denom).item()

    first_loss = last_loss = float("nan")
    last_dice = 0.0
    for step in range(steps):
        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        optimizer.step()
        if step == 0:
            first_loss = loss.item()
        last_loss = loss.item()

        with torch.no_grad():
            last_dice = binary_dice(out, y)

    print(f"  loss: {first_loss:.4f} -> {last_loss:.4f}")
    print(f"  final Dice: {last_dice:.4f}")
    check(last_loss < first_loss, "loss decreased during overfit")
    check(last_dice > 0.9, f"model overfit the single sample (Dice {last_dice:.3f} > 0.9)")


def part4_checkpoint(device, spatial_size, feature_size):
    print("\n=== Part 4: checkpoint round-trip ===")
    model = create_model(
        img_size=spatial_size,
        in_channels=1,
        out_channels=2,
        feature_size=feature_size,
        use_checkpoint=False,
        use_sf_gate=True,
    ).to(device)
    x = torch.randn(1, 1, *spatial_size, device=device)
    with torch.no_grad():
        y_before = model(x)

    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "ckpt.pt")
        torch.save({"model_state_dict": model.state_dict()}, path)

        model2 = create_model(
            img_size=spatial_size,
            in_channels=1,
            out_channels=2,
            feature_size=feature_size,
            use_checkpoint=False,
            use_sf_gate=True,
        ).to(device)
        model2.load_state_dict(torch.load(path, map_location=device, weights_only=False)["model_state_dict"])
        with torch.no_grad():
            y_after = model2(x)

    check(
        torch.allclose(y_before, y_after, atol=1e-6),
        "reloaded checkpoint reproduces identical output",
    )


def main():
    parser = argparse.ArgumentParser(description="Zero-data pipeline smoke test")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[64, 64, 64])
    parser.add_argument("--feature_size", type=int, default=24)
    parser.add_argument("--overfit_steps", type=int, default=40)
    args = parser.parse_args()

    spatial_size = tuple(args.spatial_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    part1_model(device, spatial_size, args.feature_size)
    part2_dataloaders(spatial_size)
    part3_overfit(device, spatial_size, args.feature_size, args.overfit_steps)
    part4_checkpoint(device, spatial_size, args.feature_size)

    print("\n=== ALL SMOKE TESTS PASSED ===")


if __name__ == "__main__":
    main()
