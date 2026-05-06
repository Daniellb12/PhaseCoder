"""
Train PhaseCoder on toy synthetic dataset.

Trains the same PhaseCoder architecture as the paper, on a small synthetic
dataset generated with pyroomacoustics. Supports both static and IMU-conditioned
training modes.

Usage:
    # Train baseline (ignores IMU even on dynamic clips)
    python train_toy.py --data_dir ./toy_data --output_dir ./runs/baseline --use_imu false

    # Train with IMU conditioning
    python train_toy.py --data_dir ./toy_data --output_dir ./runs/imu --use_imu true
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PhaseCoder.PhaseCoder import PhaseCoder, PhaseCoderLoss


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ToyDataset(Dataset):
    def __init__(self, data_dir: Path, filenames: list[str], use_imu: bool):
        self.data_dir = data_dir
        self.filenames = filenames
        self.use_imu = use_imu

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        path = self.data_dir / self.filenames[idx]
        data = np.load(path)

        audio = torch.from_numpy(data["audio"]).float()  # (C, T)
        mic_coords = torch.from_numpy(data["mic_coords"]).float()  # (C, 3)

        labels = {
            "azimuth": int(data["azimuth_class"]),
            "elevation": int(data["elevation_class"]),
            "distance": int(data["distance_class"]),
        }

        is_dynamic = bool(data["is_dynamic"])
        if self.use_imu and is_dynamic:
            imu_quats = torch.from_numpy(data["imu_quats"]).float()  # (F, 4)
        elif self.use_imu and not is_dynamic:
            # Identity quaternions for static clips when running in IMU mode.
            # Frame count must match what PhaseCoder.STFTPatchExtractor produces.
            F = data["imu_quats"].shape[0] if "imu_quats" in data.files else 32
            imu_quats = torch.zeros(F, 4)
            imu_quats[:, 0] = 1.0  # w=1 → identity
        else:
            imu_quats = None  # signals "use static path"

        return {
            "audio": audio,
            "mic_coords": mic_coords,
            "imu_quats": imu_quats,
            "labels": labels,
            "is_dynamic": is_dynamic,
            "rotation_rate_dps": float(data["rotation_rate_dps"]),
        }


def collate_fn(batch):
    """Custom collate to handle variable mic counts via padding to max in batch."""
    max_mics = max(item["mic_coords"].shape[0] for item in batch)
    B = len(batch)
    T = batch[0]["audio"].shape[-1]

    audio = torch.zeros(B, max_mics, T)
    mic_coords = torch.zeros(B, max_mics, 3)
    imu_quats = None

    has_imu = batch[0]["imu_quats"] is not None
    if has_imu:
        F = batch[0]["imu_quats"].shape[0]
        imu_quats = torch.zeros(B, F, 4)

    for i, item in enumerate(batch):
        c = item["audio"].shape[0]
        audio[i, :c] = item["audio"]
        mic_coords[i, :c] = item["mic_coords"]
        # Pad remaining mic positions with the centroid (so they don't shift it)
        if c < max_mics:
            centroid = item["mic_coords"].mean(dim=0)
            mic_coords[i, c:] = centroid
        if has_imu:
            imu_quats[i] = item["imu_quats"]

    labels = {
        "azimuth": torch.tensor([item["labels"]["azimuth"] for item in batch], dtype=torch.long),
        "elevation": torch.tensor([item["labels"]["elevation"] for item in batch], dtype=torch.long),
        "distance": torch.tensor([item["labels"]["distance"] for item in batch], dtype=torch.long),
    }

    return {
        "audio": audio,
        "mic_coords": mic_coords,
        "imu_quats": imu_quats,
        "labels": labels,
        "is_dynamic": [item["is_dynamic"] for item in batch],
        "rotation_rate_dps": [item["rotation_rate_dps"] for item in batch],
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load manifest
    data_dir = Path(args.data_dir)
    with open(data_dir / "manifest.json") as f:
        manifest = json.load(f)

    train_ds = ToyDataset(data_dir, manifest["train_filenames"], use_imu=args.use_imu)
    val_ds = ToyDataset(data_dir, manifest["val_filenames"], use_imu=args.use_imu)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    print(f"Train clips: {len(train_ds)}, Val clips: {len(val_ds)}")
    print(f"IMU mode: {'ON' if args.use_imu else 'OFF (baseline)'}")

    # Build model
    model = PhaseCoder().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = PhaseCoderLoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "val_az_acc": [],
               "val_el_acc": [], "val_dist_acc": []}

    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_losses = []
        for batch in train_loader:
            audio = batch["audio"].to(device)
            mic_coords = batch["mic_coords"].to(device)
            imu = batch["imu_quats"].to(device) if batch["imu_quats"] is not None else None
            targets = {k: v.to(device) for k, v in batch["labels"].items()}

            optimizer.zero_grad()
            out = model(audio, mic_coords, imu_orientations=imu)
            losses = criterion(out, targets)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(losses["loss"].item())

        scheduler.step()
        avg_train_loss = np.mean(train_losses)

        # --- Validate ---
        model.eval()
        val_losses = []
        az_correct = el_correct = dist_correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device)
                mic_coords = batch["mic_coords"].to(device)
                imu = batch["imu_quats"].to(device) if batch["imu_quats"] is not None else None
                targets = {k: v.to(device) for k, v in batch["labels"].items()}

                out = model(audio, mic_coords, imu_orientations=imu)
                losses = criterion(out, targets)
                val_losses.append(losses["loss"].item())

                az_correct += (out["azimuth_logits"].argmax(dim=-1) == targets["azimuth"]).sum().item()
                el_correct += (out["elevation_logits"].argmax(dim=-1) == targets["elevation"]).sum().item()
                dist_correct += (out["distance_logits"].argmax(dim=-1) == targets["distance"]).sum().item()
                total += targets["azimuth"].shape[0]

        avg_val_loss = np.mean(val_losses) if val_losses else float("nan")
        az_acc = az_correct / max(total, 1)
        el_acc = el_correct / max(total, 1)
        dist_acc = dist_correct / max(total, 1)

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_az_acc"].append(az_acc)
        history["val_el_acc"].append(el_acc)
        history["val_dist_acc"].append(dist_acc)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train {avg_train_loss:.3f} | val {avg_val_loss:.3f} | "
              f"az {az_acc:.2%} el {el_acc:.2%} dist {dist_acc:.2%} | "
              f"{elapsed:.0f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "args": vars(args),
            }, output_dir / "best.pt")

    # Save final history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs - 1,
        "args": vars(args),
    }, output_dir / "final.pt")

    print(f"\n✓ Training complete. Best val loss: {best_val_loss:.3f}")
    print(f"  Checkpoints: {output_dir}/best.pt, {output_dir}/final.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--use_imu", type=lambda s: s.lower() == "true", default=False,
                        help="Whether to feed IMU quaternions to the model")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
