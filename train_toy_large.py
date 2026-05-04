"""
Training script for large-scale toy datasets (10k-100k clips).

Differences from train_toy.py:
    - Higher batch size (64 vs 16) — better GPU utilization
    - Larger learning rate with warmup — typical for transformer training at scale
    - More dataloader workers — keeps GPU fed when CPU decoding is the bottleneck
    - Periodic checkpointing — protects against crashes during long runs
    - Mixed precision (fp16) on GPU — ~2x training throughput on modern GPUs
    - Better progress reporting per-epoch with samples/sec tracking

Usage:
    # On GPU (recommended for 100k clips)
    python train_toy_large.py \\
        --data_dir ./big_data \\
        --output_dir ./runs/imu_large \\
        --use_imu true \\
        --epochs 20 \\
        --batch_size 64

    # On CPU (will be slow — only for testing)
    python train_toy_large.py \\
        --data_dir ./big_data \\
        --output_dir ./runs/imu_large \\
        --use_imu true \\
        --epochs 5 \\
        --batch_size 32
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from PhaseCoder import PhaseCoder, PhaseCoderLoss
from train_toy import ToyDataset, collate_fn


def get_lr_schedule(optimizer, num_epochs: int, warmup_epochs: int = 2):
    """Linear warmup followed by cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        # Cosine decay over the remaining epochs
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.amp
    print(f"Using device: {device}")
    print(f"Mixed precision (AMP): {'ENABLED' if use_amp else 'disabled'}")

    # Load manifest
    data_dir = Path(args.data_dir)
    with open(data_dir / "manifest.json") as f:
        manifest = json.load(f)

    train_ds = ToyDataset(data_dir, manifest["train_filenames"], use_imu=args.use_imu)
    val_ds = ToyDataset(data_dir, manifest["val_filenames"], use_imu=args.use_imu)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, drop_last=True,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=max(1, args.num_workers // 2),
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    print(f"Train clips: {len(train_ds)}, Val clips: {len(val_ds)}")
    print(f"Batches per epoch: {len(train_loader)}")
    print(f"IMU mode: {'ON' if args.use_imu else 'OFF (baseline)'}")

    model = PhaseCoder().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = get_lr_schedule(optimizer, args.epochs, warmup_epochs=args.warmup_epochs)
    criterion = PhaseCoderLoss()

    # GradScaler for mixed precision
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_az_acc": [],
               "val_el_acc": [], "val_dist_acc": [], "lr": []}
    best_val_loss = float("inf")
    overall_start = time.time()

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_losses = []
        epoch_start = time.time()
        samples_seen = 0

        for batch_idx, batch in enumerate(train_loader):
            audio = batch["audio"].to(device, non_blocking=True)
            mic_coords = batch["mic_coords"].to(device, non_blocking=True)
            imu = batch["imu_quats"].to(device, non_blocking=True) if batch["imu_quats"] is not None else None
            targets = {k: v.to(device, non_blocking=True) for k, v in batch["labels"].items()}

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out = model(audio, mic_coords, imu_orientations=imu)
                    losses = criterion(out, targets)
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out = model(audio, mic_coords, imu_orientations=imu)
                losses = criterion(out, targets)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            train_losses.append(losses["loss"].item())
            samples_seen += audio.shape[0]

            # Mid-epoch progress for long epochs
            if (batch_idx + 1) % max(1, len(train_loader) // 10) == 0:
                elapsed = time.time() - epoch_start
                rate = samples_seen / max(elapsed, 0.001)
                pct = 100 * (batch_idx + 1) / len(train_loader)
                avg_loss_so_far = np.mean(train_losses[-100:])
                print(f"  epoch {epoch+1}: {pct:5.1f}% | "
                      f"batch {batch_idx+1}/{len(train_loader)} | "
                      f"loss {avg_loss_so_far:.3f} | "
                      f"{rate:.0f} samp/s")

        scheduler.step()
        epoch_time = time.time() - epoch_start
        avg_train_loss = np.mean(train_losses)

        # --- Validate ---
        model.eval()
        val_losses = []
        az_correct = el_correct = dist_correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device, non_blocking=True)
                mic_coords = batch["mic_coords"].to(device, non_blocking=True)
                imu = batch["imu_quats"].to(device, non_blocking=True) if batch["imu_quats"] is not None else None
                targets = {k: v.to(device, non_blocking=True) for k, v in batch["labels"].items()}

                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        out = model(audio, mic_coords, imu_orientations=imu)
                        losses = criterion(out, targets)
                else:
                    out = model(audio, mic_coords, imu_orientations=imu)
                    losses = criterion(out, targets)

                val_losses.append(losses["loss"].item())
                az_correct += (out["azimuth_logits"].argmax(dim=-1) == targets["azimuth"]).sum().item()
                el_correct += (out["elevation_logits"].argmax(dim=-1) == targets["elevation"]).sum().item()
                dist_correct += (out["distance_logits"].argmax(dim=-1) == targets["distance"]).sum().item()
                total += targets["azimuth"].shape[0]

        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        az_acc = az_correct / max(total, 1)
        el_acc = el_correct / max(total, 1)
        dist_acc = dist_correct / max(total, 1)

        history["train_loss"].append(float(avg_train_loss))
        history["val_loss"].append(avg_val_loss)
        history["val_az_acc"].append(az_acc)
        history["val_el_acc"].append(el_acc)
        history["val_dist_acc"].append(dist_acc)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        total_elapsed = (time.time() - overall_start) / 60
        print(f"\nEpoch {epoch+1:3d}/{args.epochs} | "
              f"train {avg_train_loss:.3f} | val {avg_val_loss:.3f} | "
              f"az {az_acc:.2%} el {el_acc:.2%} dist {dist_acc:.2%} | "
              f"epoch {epoch_time:.0f}s | total {total_elapsed:.1f}min\n")

        # Save best checkpoint
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "val_az_acc": az_acc,
                "val_el_acc": el_acc,
                "val_dist_acc": dist_acc,
                "args": vars(args),
            }, output_dir / "best.pt")

        # Always save latest for crash recovery
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "history": history,
            "args": vars(args),
        }, output_dir / "latest.pt")

        # Checkpoint history every epoch
        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs - 1,
        "args": vars(args),
    }, output_dir / "final.pt")

    print(f"\n✓ Training complete in {(time.time() - overall_start)/60:.1f} minutes")
    print(f"  Best val loss: {best_val_loss:.3f}")
    print(f"  Final val accuracies: az={az_acc:.2%} el={el_acc:.2%} dist={dist_acc:.2%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_imu", type=lambda s: s.lower() == "true", default=False)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--warmup_epochs", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--amp", type=lambda s: s.lower() == "true", default=True,
                        help="Use mixed precision on GPU (default: true)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
