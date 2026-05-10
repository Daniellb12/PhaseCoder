"""
Training script for the time-varying RIR dataset (tv_train_data / tv_eval_data).

Trains both a baseline (static-array) and an IMU-conditioned PhaseCoder on pre-generated
time-varying RIR clips. The two dataset directories are fixed:

    Training:   ./tv_train_data   (100k clips, 50% dynamic, time-varying RIRs)
    Evaluation: ./tv_eval_data    (1k clips, 100% dynamic, held-out eval set)

Key differences from train_toy_large.py:
    - Uses TvDataset instead of ToyDataset — imu_quats is always present in every
      clip (identity quaternions for static clips), matching the tv_rir format.
    - Separates train vs. eval manifests — training uses tv_train_data's
      train_filenames split; final evaluation is always on tv_eval_data.
    - ffn_expansion=4 — matches the paper's 4x feedforward for better GPU utilization.
    - AMP fp16 on GPU — ~2x training throughput on RTX 4060 and above.
    - num_workers=0 — avoids Windows/Jupyter multiprocessing deadlocks.
    - Per-batch progress printing — shows loss, lr, and ETA within each epoch.

Usage:
    # Train baseline (no IMU)
    python train_toy_newdataset_large.py --use_imu false --output_dir ./runs/tv_baseline

    # Train IMU-conditioned model
    python train_toy_newdataset_large.py --use_imu true --output_dir ./runs/tv_imu

    # Custom dataset paths
    python train_toy_newdataset_large.py \\
        --train_dir ./tv_train_data \\
        --eval_dir  ./tv_eval_data  \\
        --use_imu true \\
        --output_dir ./runs/tv_imu \\
        --epochs 20 \\
        --batch_size 128
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from PhaseCoder import PhaseCoder, PhaseCoderLoss


# ---------------------------------------------------------------------------
# Dataset — matches the tv_rir clip format
# ---------------------------------------------------------------------------

class TvDataset(Dataset):
    """Dataset for time-varying RIR clips (tv_train_data / tv_eval_data).

    Every clip stores imu_quats unconditionally (identity quaternions for static
    clips). When use_imu=False the quaternions are not loaded, keeping the model
    on the static positional-embedding path.
    """

    def __init__(self, data_dir: Path, filenames: list[str], use_imu: bool):
        self.data_dir = data_dir
        self.filenames = filenames
        self.use_imu = use_imu

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        data = np.load(self.data_dir / self.filenames[idx])
        audio      = torch.from_numpy(data["audio"]).float()       # (C, T)
        mic_coords = torch.from_numpy(data["mic_coords"]).float()  # (C, 3)
        labels = {
            "azimuth":   int(data["azimuth_class"]),
            "elevation": int(data["elevation_class"]),
            "distance":  int(data["distance_class"]),
        }
        imu_quats = torch.from_numpy(data["imu_quats"]).float() if self.use_imu else None
        return {
            "audio":            audio,
            "mic_coords":       mic_coords,
            "imu_quats":        imu_quats,
            "labels":           labels,
            "is_dynamic":       bool(data["is_dynamic"]),
            "rotation_rate_dps": float(data["rotation_rate_dps"]),
        }


def collate_fn(batch):
    """Pad variable mic counts to the max in the batch."""
    max_mics = max(item["mic_coords"].shape[0] for item in batch)
    B = len(batch)
    T = batch[0]["audio"].shape[-1]

    audio      = torch.zeros(B, max_mics, T)
    mic_coords = torch.zeros(B, max_mics, 3)
    has_imu    = batch[0]["imu_quats"] is not None
    imu_quats  = torch.zeros(B, batch[0]["imu_quats"].shape[0], 4) if has_imu else None

    for i, item in enumerate(batch):
        c = item["audio"].shape[0]
        audio[i, :c]      = item["audio"]
        mic_coords[i, :c] = item["mic_coords"]
        if c < max_mics:
            mic_coords[i, c:] = item["mic_coords"].mean(dim=0)
        if has_imu:
            imu_quats[i] = item["imu_quats"]

    labels = {
        k: torch.tensor([item["labels"][k] for item in batch], dtype=torch.long)
        for k in ["azimuth", "elevation", "distance"]
    }
    return {
        "audio":            audio,
        "mic_coords":       mic_coords,
        "imu_quats":        imu_quats,
        "labels":           labels,
        "is_dynamic":       [item["is_dynamic"] for item in batch],
        "rotation_rate_dps": [item["rotation_rate_dps"] for item in batch],
    }


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def get_lr_schedule(optimizer, num_epochs: int, warmup_epochs: int):
    """Linear warmup → cosine decay."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, num_epochs - warmup_epochs)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------

def run_eval(model, loader, criterion, device, use_amp):
    """One evaluation pass. Returns (avg_loss, az_acc, el_acc, dist_acc)."""
    if loader is None:
        return float("nan"), 0.0, 0.0, 0.0

    model.eval()
    losses = []
    az_c = el_c = dist_c = tot = 0
    with torch.no_grad():
        for batch in loader:
            audio      = batch["audio"].to(device, non_blocking=True)
            mic_coords = batch["mic_coords"].to(device, non_blocking=True)
            imu        = batch["imu_quats"].to(device, non_blocking=True) if batch["imu_quats"] is not None else None
            targets    = {k: v.to(device, non_blocking=True) for k, v in batch["labels"].items()}

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out  = model(audio, mic_coords, imu_orientations=imu)
                    loss = criterion(out, targets)
            else:
                out  = model(audio, mic_coords, imu_orientations=imu)
                loss = criterion(out, targets)

            losses.append(loss["loss"].item())
            az_c   += (out["azimuth_logits"].argmax(-1)   == targets["azimuth"]).sum().item()
            el_c   += (out["elevation_logits"].argmax(-1)  == targets["elevation"]).sum().item()
            dist_c += (out["distance_logits"].argmax(-1)   == targets["distance"]).sum().item()
            tot    += targets["azimuth"].shape[0]

    return (
        float(np.mean(losses)),
        az_c   / max(tot, 1),
        el_c   / max(tot, 1),
        dist_c / max(tot, 1),
    )


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(args):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda" and args.amp

    print(f"Using device: {device}")
    print(f"AMP (fp16):   {'ENABLED' if use_amp else 'disabled'}")

    train_dir = Path(args.train_dir)
    eval_dir  = Path(args.eval_dir)

    # ── Load manifests ──────────────────────────────────────────────────────
    with open(train_dir / "manifest.json") as f:
        train_manifest = json.load(f)
    with open(eval_dir / "manifest.json") as f:
        eval_manifest = json.load(f)

    # Use ALL clips in tv_train_data for training — the manifest's baked-in
    # train_filenames/val_filenames split is intentionally ignored so the full
    # 100k dataset is used. tv_eval_data is the sole evaluation set.
    train_files = [c["filename"] for c in train_manifest["clips"]]
    eval_files  = [c["filename"] for c in eval_manifest["clips"]]

    # ── Datasets & loaders ─────────────────────────────────────────────────
    train_ds = TvDataset(train_dir, train_files, use_imu=args.use_imu)
    val_ds   = None  # no val split — tv_eval_data is the evaluation set
    eval_ds  = TvDataset(eval_dir,  eval_files,  use_imu=args.use_imu)

    # num_workers=0: main-thread loading — safe on Windows without deadlock risk.
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers,
        drop_last=True, pin_memory=(device.type == "cuda"),
    )
    eval_loader = DataLoader(
        eval_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ───────────────────────────────────────────────────────────────
    # ffn_expansion=4 matches the paper's 4x feedforward size and increases
    # compute per batch, reducing the GPU idle fraction during data loading.
    model = PhaseCoder(ffn_expansion=4).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = get_lr_schedule(optimizer, args.epochs, args.warmup_epochs)
    criterion = PhaseCoderLoss()
    scaler    = torch.amp.GradScaler("cuda") if use_amp else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    label = "IMU" if args.use_imu else "baseline"
    n_train_batches = len(train_loader)
    log_every       = max(1, n_train_batches // 10)

    print(f"\n{'='*60}")
    print(f"Training {label} model")
    print(f"{'='*60}")
    print(f"  Train dir:  {train_dir}  ({len(train_ds):,} clips, {n_train_batches} batches/epoch)")
    print(f"  Eval dir:   {eval_dir}  ({len(eval_ds):,} clips)")
    print(f"  IMU mode:   {'ON' if args.use_imu else 'OFF (baseline)'}")
    print(f"  Params:     {n_params:,} ({n_params/1e6:.2f}M)  |  ffn_expansion=4")
    print(f"  batch={args.batch_size}  lr={args.lr}  epochs={args.epochs}  "
          f"warmup={args.warmup_epochs}  workers={args.num_workers}")
    print(f"  Output:     {output_dir}")
    print(f"{'='*60}")
    sys.stdout.flush()

    history = {
        "train_loss": [], "eval_loss": [],
        "eval_az_acc": [], "eval_el_acc": [], "eval_dist_acc": [],
        "lr": [],
    }
    best_eval    = float("inf")
    overall_start = time.time()

    for epoch in range(args.epochs):
        model.train()
        train_losses  = []
        running_loss  = 0.0
        epoch_start   = time.time()
        samples_seen  = 0

        for batch_idx, batch in enumerate(train_loader):
            audio      = batch["audio"].to(device, non_blocking=True)
            mic_coords = batch["mic_coords"].to(device, non_blocking=True)
            imu        = batch["imu_quats"].to(device, non_blocking=True) if batch["imu_quats"] is not None else None
            targets    = {k: v.to(device, non_blocking=True) for k, v in batch["labels"].items()}

            optimizer.zero_grad(set_to_none=True)

            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    out    = model(audio, mic_coords, imu_orientations=imu)
                    losses = criterion(out, targets)
                scaler.scale(losses["loss"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                out    = model(audio, mic_coords, imu_orientations=imu)
                losses = criterion(out, targets)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            loss_val = losses["loss"].item()
            train_losses.append(loss_val)
            running_loss += loss_val
            samples_seen += audio.shape[0]

            if (batch_idx + 1) % log_every == 0 or (batch_idx + 1) == n_train_batches:
                elapsed     = time.time() - epoch_start
                batches_done = batch_idx + 1
                avg_so_far  = running_loss / batches_done
                eta_s       = elapsed / batches_done * (n_train_batches - batches_done)
                current_lr  = scheduler.get_last_lr()[0]
                clips_done  = batches_done * args.batch_size
                clips_total = n_train_batches * args.batch_size
                print(f"  Epoch {epoch+1:2d}/{args.epochs} | "
                      f"batch {batches_done:4d}/{n_train_batches} "
                      f"({clips_done:,}/{clips_total:,} clips) | "
                      f"loss {avg_so_far:.4f} | "
                      f"lr {current_lr:.2e} | "
                      f"ETA {eta_s:.0f}s")
                sys.stdout.flush()

        scheduler.step()

        avg_train                              = float(np.mean(train_losses))
        avg_eval, eval_az, eval_el, eval_dist  = run_eval(model, eval_loader, criterion, device, use_amp)

        history["train_loss"].append(avg_train)
        history["eval_loss"].append(avg_eval)
        history["eval_az_acc"].append(eval_az)
        history["eval_el_acc"].append(eval_el)
        history["eval_dist_acc"].append(eval_dist)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        epoch_time    = time.time() - epoch_start
        total_elapsed = time.time() - overall_start
        saved         = " ✓ saved" if avg_eval < best_eval else ""
        eta_remaining = (total_elapsed / (epoch + 1)) * (args.epochs - epoch - 1)

        print(f"  {'─'*56}")
        print(f"  Epoch {epoch+1:2d}/{args.epochs} COMPLETE | "
              f"train {avg_train:.4f} | eval {avg_eval:.4f}{saved}")
        print(f"  Eval acc → az {eval_az:.2%}  el {eval_el:.2%}  dist {eval_dist:.2%}")
        print(f"  Epoch {epoch_time:.0f}s | "
              f"Total {total_elapsed/60:.1f}min | "
              f"ETA {eta_remaining/60:.1f}min")
        print(f"  {'─'*56}")
        sys.stdout.flush()

        # Save best checkpoint (keyed on eval loss, not val loss)
        if avg_eval < best_eval:
            best_eval = avg_eval
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch":            epoch,
                "eval_loss":        avg_eval,
                "eval_az_acc":      eval_az,
                "eval_el_acc":      eval_el,
                "eval_dist_acc":    eval_dist,
                "args":             vars(args),
            }, output_dir / "best.pt")

        # Always save latest checkpoint for crash recovery
        torch.save({
            "model_state_dict":      model.state_dict(),
            "optimizer_state_dict":  optimizer.state_dict(),
            "scheduler_state_dict":  scheduler.state_dict(),
            "epoch":                 epoch,
            "history":               history,
            "args":                  vars(args),
        }, output_dir / "latest.pt")

        with open(output_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    # Final checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch":            args.epochs - 1,
        "args":             vars(args),
    }, output_dir / "final.pt")

    total_min = (time.time() - overall_start) / 60
    print(f"\n{'='*60}")
    print(f"✓ {label} training complete in {total_min:.1f} minutes")
    print(f"  Best eval loss:     {best_eval:.4f}")
    print(f"  Best eval az acc:   {max(history['eval_az_acc']):.2%}")
    print(f"  Best eval el acc:   {max(history['eval_el_acc']):.2%}")
    print(f"  Best eval dist acc: {max(history['eval_dist_acc']):.2%}")
    print(f"  Checkpoints saved to: {output_dir}")
    print(f"{'='*60}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train PhaseCoder on tv_train_data / tv_eval_data"
    )
    parser.add_argument("--train_dir",      type=str, default="./tv_train_data",
                        help="Directory containing training clips + manifest.json")
    parser.add_argument("--eval_dir",       type=str, default="./tv_eval_data",
                        help="Directory containing evaluation clips + manifest.json")
    parser.add_argument("--output_dir",     type=str, required=True,
                        help="Where to save checkpoints and history")
    parser.add_argument("--use_imu",        type=lambda s: s.lower() == "true",
                        default=False,
                        help="Feed per-frame IMU quaternions to the model (default: false)")
    parser.add_argument("--epochs",         type=int,   default=15)
    parser.add_argument("--batch_size",     type=int,   default=128,
                        help="Batch size (default 128, safe for RTX 4060 8 GB)")
    parser.add_argument("--lr",             type=float, default=5e-4)
    parser.add_argument("--warmup_epochs",  type=int,   default=2)
    parser.add_argument("--num_workers",    type=int,   default=0,
                        help="DataLoader workers (default 0 — avoids Windows deadlock)")
    parser.add_argument("--amp",            type=lambda s: s.lower() == "true",
                        default=True,
                        help="Use AMP fp16 on GPU (default: true)")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
