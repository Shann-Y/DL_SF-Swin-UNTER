"""Pipeline quickstart on a small (2-3 case) subset.

Run this BEFORE full training. It validates, in order:

  1. data sanity  — image/mask pair count + the mask is binary 0/1
  2. dataloaders   — batch shapes / dtype / NaN / label range
  3. overfit       — the model can memorize ONE real patch (loss -> 0, Dice -> 1)

Parts 1-2 run fine on CPU; part 3 (overfit) is much faster on GPU (Colab).

Usage (from the proj_code directory):
    python quickstart.py                                     # default ./data/zenodo_small
    python quickstart.py --data_root ./data/zenodo_small --ssl_pretrained
    python quickstart.py --spatial_size 64 64 64 --feature_size 24 --steps 30   # lighter, for CPU
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
import nibabel as nib
from monai.losses import DiceCELoss

from data import COVID19Dataset, get_dataloaders
from models import create_model


def check(cond: bool, msg: str) -> None:
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        sys.exit(1)


def part1_data(data_root: str) -> None:
    print("\n=== 1/3: data sanity ===")
    ds = COVID19Dataset(data_root=data_root, dataset_type="zenodo", split="all")
    n = len(ds.data_dicts)
    check(n >= 1, f"found {n} image/mask pair(s)")
    print(f"  pairs: {n}")
    for d in ds.data_dicts[:3]:
        print("   ", d["image"])

    img = nib.load(ds.data_dicts[0]["image"]).get_fdata()
    lbl = nib.load(ds.data_dicts[0]["label"]).get_fdata()
    uniq = sorted(set(np.unique(lbl).tolist()))
    check(set(uniq) <= {0.0, 1.0}, f"mask is binary 0/1 (unique = {uniq})")
    print(f"  first image shape {img.shape}, HU range [{img.min():.0f}, {img.max():.0f}]")


def part2_loader(data_root: str, spatial_size, batch_size: int) -> None:
    print("\n=== 2/3: dataloader + preprocessing ===")
    train_loader, val_loader = get_dataloaders(
        data_root=data_root, dataset_type="zenodo",
        batch_size=batch_size, spatial_size=spatial_size,
        num_workers=0, val_ratio=0.3, seed=42,
    )
    check(len(train_loader) > 0, "train_loader non-empty")
    check(len(val_loader) > 0, "val_loader non-empty")

    b = next(iter(train_loader))
    x, y = b["image"], b["label"]
    # With few (2-3) cases the batch may be smaller than batch_size, so only
    # check the channel + spatial dimensions here.
    check(tuple(x.shape)[1:] == (1, *spatial_size), f"train image shape {tuple(x.shape)} (batch {x.shape[0]})")
    check(x.dtype == torch.float32, f"dtype {x.dtype}")
    check(not torch.isnan(x).any().item(), "no NaN in images")
    ul = torch.unique(y).tolist()
    check(set(ul) <= {0.0, 1.0}, f"label values in {{0,1}} (unique {ul})")

    vb = next(iter(val_loader))
    print(f"  val image shape {tuple(vb['image'].shape)} (full volume; sliding-window handles it)")


def part3_overfit(data_root: str, spatial_size, feature_size: int, steps: int, ssl_pretrained: bool) -> None:
    print(f"\n=== 3/3: overfit one real patch ({steps} steps) ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("  device:", device)

    train_loader, _ = get_dataloaders(
        data_root=data_root, dataset_type="zenodo",
        batch_size=1, spatial_size=spatial_size,
        num_workers=0, val_ratio=0.3, seed=42,
    )
    model = create_model(
        img_size=spatial_size, in_channels=1, out_channels=2,
        feature_size=feature_size, use_checkpoint=False, use_sf_gate=True,
    )
    if ssl_pretrained:
        model.load_ssl_pretrained(str(Path(__file__).parent / "model_swinvit.pt"))
    model = model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)  # high LR to overfit one patch fast
    loss_fn = DiceCELoss(to_onehot_y=True, softmax=True)

    batch = next(iter(train_loader))
    x = batch["image"].to(device)
    y = batch["label"].to(device)

    def dice(p, t):
        p = p.argmax(1)
        t = t.squeeze(1).round().long()
        inter = (p * t).sum().float()
        return (2 * inter / (p.sum() + t.sum() + 1e-6)).item()

    first = last = float("nan")
    for step in range(steps):
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        if step == 0:
            first = loss.item()
        last = loss.item()
        if step % 10 == 0:
            print(f"  step {step:3d}: loss {loss.item():.4f}")

    with torch.no_grad():
        final_d = dice(model(x), y)
    print(f"  final: loss {first:.4f} -> {last:.4f}, Dice {final_d:.4f}")

    check(last < first, f"loss decreased ({first:.3f} -> {last:.3f})")
    check(final_d > 0.9, f"overfit succeeded (Dice {final_d:.3f} > 0.9)")


def main() -> None:
    p = argparse.ArgumentParser(description="Pipeline quickstart on a small subset")
    p.add_argument("--data_root", type=str, default="./data/zenodo_small")
    p.add_argument("--spatial_size", type=int, nargs=3, default=[96, 96, 96])
    p.add_argument("--feature_size", type=int, default=48)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--steps", type=int, default=200, help="overfit steps (single patch, so 200 is plenty)")
    p.add_argument("--ssl_pretrained", action="store_true", help="load model_swinvit.pt before overfit")
    args = p.parse_args()

    spatial = tuple(args.spatial_size)
    part1_data(args.data_root)
    part2_loader(args.data_root, spatial, args.batch_size)
    part3_overfit(args.data_root, spatial, args.feature_size, args.steps, args.ssl_pretrained)

    print("\n=== QUICKSTART PASSED ===")


if __name__ == "__main__":
    main()
