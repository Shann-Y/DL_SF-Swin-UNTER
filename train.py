#!/usr/bin/env python3
"""Train SF-Swin-UNETR for COVID-19 CT infection segmentation."""

import os
import sys
import argparse
import time
import random
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

from monai.losses import DiceLoss, DiceCELoss, FocalLoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference
from monai.data import decollate_batch
from monai.transforms import AsDiscrete, Compose

from data import get_dataloaders
from models import create_model
from configs import Config


def set_seed(seed: int, deterministic: bool = False):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.benchmark = True


def get_loss_function(config: Config):
    """Return the configured loss function."""
    if config.loss_type == "dice":
        return DiceLoss(to_onehot_y=True, softmax=True)
    elif config.loss_type == "ce":
        return nn.CrossEntropyLoss()
    elif config.loss_type == "dice_ce":
        return DiceCELoss(to_onehot_y=True, softmax=True)
    elif config.loss_type == "focal":
        return FocalLoss(to_onehot_y=True)
    else:
        raise ValueError(f"Unknown loss type: {config.loss_type}")


def get_optimizer(model: nn.Module, config: Config):
    """Return the optimizer; SF-Gate params use a higher learning rate."""
    if config.use_sf_gate:
        sf_gate_params = model.get_sf_gate_params()
        backbone_params = model.get_backbone_params()
        param_groups = [
            {"params": backbone_params, "lr": config.base_lr},
            {"params": sf_gate_params, "lr": config.base_lr * config.sf_gate_lr_multiplier},
        ]
    else:
        param_groups = model.parameters()

    if config.optimizer == "adamw":
        optimizer = optim.AdamW(param_groups, lr=config.base_lr, weight_decay=config.weight_decay)
    elif config.optimizer == "sgd":
        optimizer = optim.SGD(param_groups, lr=config.base_lr, momentum=0.9, weight_decay=config.weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {config.optimizer}")

    return optimizer


def get_scheduler(optimizer, config: Config):
    """Return the configured LR scheduler."""
    if config.lr_scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.max_epochs - config.warmup_epochs, eta_min=1e-6,
        )
    elif config.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)
    elif config.lr_scheduler == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=20)
    else:
        scheduler = None
    return scheduler


def warmup_lr(optimizer, epoch: int, warmup_epochs: int):
    """Linear warm-up, scaled per param group so the SF-Gate/backbone learning-rate
    ratio (sf_gate_lr_multiplier) is preserved."""
    if epoch < warmup_epochs:
        scale = (epoch + 1) / warmup_epochs
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['initial_lr'] * scale


def train_one_epoch(model, train_loader, optimizer, loss_fn, scaler, device, epoch, config, writer=None):
    """Train for one epoch and return the average loss."""
    model.train()
    epoch_loss = 0
    step = 0

    for batch_data in train_loader:
        step += 1
        inputs = batch_data["image"].to(device)
        labels = batch_data["label"].to(device)

        optimizer.zero_grad()

        if config.amp:
            with autocast():
                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

        if step % config.log_interval == 0:
            print(f"  Step {step}/{len(train_loader)}, Loss: {loss.item():.4f}")
            if writer is not None:
                global_step = epoch * len(train_loader) + step
                writer.add_scalar("train/loss_step", loss.item(), global_step)

    return epoch_loss / step


def validate(model, val_loader, loss_fn, device, config, post_pred, post_label, dice_metric):
    """Run sliding-window inference on the validation set; return (loss, Dice)."""
    model.eval()
    val_loss = 0

    with torch.no_grad():
        for batch_data in val_loader:
            inputs = batch_data["image"].to(device)
            labels = batch_data["label"].to(device)

            outputs = sliding_window_inference(
                inputs,
                roi_size=config.spatial_size,
                sw_batch_size=4,
                predictor=model,
                overlap=0.5,
            )

            loss = loss_fn(outputs, labels)
            val_loss += loss.item()

            outputs_convert = [post_pred(i) for i in decollate_batch(outputs)]
            labels_convert = [post_label(i) for i in decollate_batch(labels)]
            dice_metric(y_pred=outputs_convert, y=labels_convert)

    avg_loss = val_loss / len(val_loader)
    dice_score = dice_metric.aggregate().item()
    dice_metric.reset()
    return avg_loss, dice_score


def main(args):
    config = Config.from_args(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"{'='*60}\n")

    set_seed(config.seed, config.deterministic)

    print("Loading data...")
    train_loader, val_loader = get_dataloaders(
        data_root=config.data_root,
        dataset_type=config.dataset_type,
        batch_size=config.batch_size,
        spatial_size=config.spatial_size,
        num_workers=config.num_workers,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")

    print("\nCreating model...")
    model = create_model(
        img_size=config.spatial_size,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        feature_size=config.feature_size,
        use_checkpoint=config.use_checkpoint,
        use_sf_gate=config.use_sf_gate,
        sf_gate_reduction=config.sf_gate_reduction,
    )

    if config.use_ssl_pretrained:
        print("Loading SSL pretrained weights...")
        try:
            ssl_weight_path = "./model_swinvit.pt"
            if not os.path.exists(ssl_weight_path):
                print(f"Warning: Local file {ssl_weight_path} not found!")
            model.load_ssl_pretrained(ssl_weight_path)
        except Exception as e:
            print(f"Warning: Failed to load SSL pretrained weights: {e}")
            print("Continuing with random initialization...")

    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6:.2f} M")
    print(f"Trainable parameters: {trainable_params / 1e6:.2f} M")

    loss_fn = get_loss_function(config)
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    scaler = GradScaler() if config.amp else None

    dice_metric = DiceMetric(include_background=False, reduction="mean")
    post_pred = Compose([AsDiscrete(argmax=True, to_onehot=config.out_channels)])
    post_label = Compose([AsDiscrete(to_onehot=config.out_channels)])

    writer = None
    if config.use_tensorboard:
        writer = SummaryWriter(log_dir=os.path.join(config.exp_dir, "logs"))

    best_dice = 0
    best_epoch = 0
    no_improve_count = 0
    train_losses, val_losses, val_dices, val_epochs = [], [], [], []

    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"{'='*60}\n")

    for epoch in range(config.max_epochs):
        epoch_start = time.time()

        if epoch < config.warmup_epochs:
            warmup_lr(optimizer, epoch, config.warmup_epochs)

        print(f"Epoch {epoch + 1}/{config.max_epochs}")
        train_loss = train_one_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device, epoch, config, writer
        )
        train_losses.append(train_loss)

        if epoch >= config.warmup_epochs and scheduler is not None:
            if config.lr_scheduler != "plateau":
                scheduler.step()

        val_loss = None
        val_dice = None
        if (epoch + 1) % config.val_interval == 0:
            val_loss, val_dice = validate(
                model, val_loader, loss_fn, device, config, post_pred, post_label, dice_metric
            )
            val_losses.append(val_loss)
            val_dices.append(val_dice)
            val_epochs.append(epoch + 1)

            if config.lr_scheduler == "plateau" and scheduler is not None:
                scheduler.step(val_dice)

            if val_dice > best_dice:
                best_dice = val_dice
                best_epoch = epoch + 1
                no_improve_count = 0
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_dice": best_dice,
                        "config": vars(config),
                    },
                    os.path.join(config.exp_dir, "best_model.pth"),
                )
                print(f"  * New best model saved! Dice: {best_dice:.4f}")
            else:
                no_improve_count += config.val_interval

        if (epoch + 1) % config.save_interval == 0:
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": vars(config),
                },
                os.path.join(config.exp_dir, "checkpoints", f"checkpoint_epoch{epoch + 1}.pth"),
            )

        epoch_time = time.time() - epoch_start
        lr = optimizer.param_groups[0]['lr']
        log_msg = f"  Loss: {train_loss:.4f}, LR: {lr:.2e}, Time: {epoch_time:.1f}s"
        if val_loss is not None:
            log_msg += f"\n  Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}, Best: {best_dice:.4f} (Epoch {best_epoch})"
        print(log_msg)

        if writer is not None:
            writer.add_scalar("train/loss", train_loss, epoch + 1)
            writer.add_scalar("train/lr", lr, epoch + 1)
            if val_loss is not None:
                writer.add_scalar("val/loss", val_loss, epoch + 1)
                writer.add_scalar("val/dice", val_dice, epoch + 1)

        if config.early_stopping and no_improve_count >= config.early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch + 1} epochs!")
            break

    torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_dice": best_dice,
            "config": vars(config),
        },
        os.path.join(config.exp_dir, "final_model.pth"),
    )

    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best Dice: {best_dice:.4f} at Epoch {best_epoch}")
    print(f"Models saved to: {config.exp_dir}")
    print(f"{'='*60}")

    if writer is not None:
        writer.close()

    try:
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(range(1, len(train_losses) + 1), train_losses, label='Train Loss')
        if val_losses:
            plt.plot(val_epochs, val_losses, label='Val Loss', marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        if val_dices:
            plt.plot(val_epochs, val_dices, label='Val Dice', marker='o', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Dice Score')
        plt.title('Validation Dice Score')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(os.path.join(config.exp_dir, "training_curves.png"))
        print(f"Training curves saved to {os.path.join(config.exp_dir, 'training_curves.png')}")
    except Exception as e:
        print(f"Failed to plot training curves: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train SF-Swin-UNETR for COVID-19 CT segmentation")

    # data
    parser.add_argument("--data_root", type=str, required=True, help="Data root directory")
    parser.add_argument("--dataset_type", type=str, default="zenodo", help="Dataset type")

    # model
    parser.add_argument("--use_sf_gate", action="store_true", default=True, help="Use SF-Gate")
    parser.add_argument("--no_sf_gate", dest="use_sf_gate", action="store_false", help="Disable SF-Gate")
    parser.add_argument("--feature_size", type=int, default=48, help="Feature size")

    # training
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=500, dest="max_epochs", help="Max epochs")
    parser.add_argument("--lr", type=float, default=2e-4, dest="base_lr", help="Learning rate")
    parser.add_argument("--spatial_size", type=int, nargs=3, default=[96, 96, 96], help="Spatial size")

    # validation
    parser.add_argument("--val_interval", type=int, default=10, help="Validation interval")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="Validation ratio")

    # output
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--exp_name", type=str, default="covid_sf_swin", help="Experiment name")

    # misc
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers")
    parser.add_argument("--no_amp", dest="amp", action="store_false", help="Disable AMP/mixed precision (required on CPU)")
    parser.add_argument("--no_ssl_pretrained", dest="use_ssl_pretrained", action="store_false",
                        help="Skip SSL pretrained backbone weights (faster for debug/smoke runs)")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if isinstance(args.spatial_size, list):
        args.spatial_size = tuple(args.spatial_size)
    main(args)
