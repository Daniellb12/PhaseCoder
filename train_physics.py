"""
Train MicPositionDecoder on the toy dataset.

MicPositionDecoder sits on top of PhaseCoder's transformer and predicts
per-mic Cartesian positions (B, F, C, 3) at every STFT frame, supervised
against ground truth positions derived from the dataset's IMU quaternions.

Ground truth derivation mirrors PhaseCoder.forward exactly: each frame's
mic positions are the neutral-frame coords rotated by R_rel (the frame's
absolute rotation expressed relative to the canonical clip pose at F//2).
For static clips, R_rel = I at every frame, so ground truth = mic_coords.

The full PhaseCoder backbone can be frozen (decoder-only training) or
fine-tuned jointly via --freeze_phasecoder.

Usage:
    # Decoder only, from a pretrained PhaseCoder checkpoint
    python train_physics.py \\
        --data_dir ./toy_data \\
        --output_dir ./runs/mic_decoder \\
        --phasecoder_ckpt ./runs/imu/best.pt \\
        --freeze_phasecoder true

    # Joint fine-tuning from a checkpoint
    python train_physics.py \\
        --data_dir ./toy_data \\
        --output_dir ./runs/mic_decoder_joint \\
        --phasecoder_ckpt ./runs/imu/best.pt \\
        --freeze_phasecoder false

    # Train everything from scratch (no checkpoint)
    python train_physics.py \\
        --data_dir ./toy_data \\
        --output_dir ./runs/mic_decoder_scratch \\
        --freeze_phasecoder false
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from PhaseCoder.PhaseCoder import PhaseCoder, quaternion_to_rotation_matrix
from train_toy import ToyDataset, collate_fn


# ---------------------------------------------------------------------------
# Ground truth computation
# ---------------------------------------------------------------------------

def compute_gt_mic_positions(
    mic_coords: torch.Tensor,
    imu_quats: torch.Tensor,
    canonical_frame_idx: int | None = None,
) -> torch.Tensor:
    """Derive ground truth per-frame mic positions from neutral coords + IMU quats.

    Mirrors the rotation logic in PhaseCoder.forward so that the decoder's
    supervision target is in the same relative frame the encoder uses.

    Args:
        mic_coords:          (B, C, 3) — neutral-frame mic positions (metres).
        imu_quats:           (B, F, 4) — per-frame unit quaternions (w, x, y, z).
                             Identity quaternions are used for static clips.
        canonical_frame_idx: reference frame; defaults to F // 2.

    Returns:
        (B, F, C, 3) — ground truth mic positions at each STFT frame.
    """
    B, F, _ = imu_quats.shape
    ref = canonical_frame_idx if canonical_frame_idx is not None else F // 2

    R_abs = quaternion_to_rotation_matrix(imu_quats)          # (B, F, 3, 3)
    R_ref_T = R_abs[:, ref].transpose(-1, -2).unsqueeze(1)    # (B, 1, 3, 3)
    R_rel = torch.matmul(R_ref_T, R_abs)                       # (B, F, 3, 3)

    return torch.einsum("bfij,bcj->bfci", R_rel, mic_coords)  # (B, F, C, 3)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Freeze PhaseCoder: {args.freeze_phasecoder}")

    data_dir = Path(args.data_dir)
    with open(data_dir / "manifest.json") as f:
        manifest = json.load(f)

    # use_imu=True ensures every clip (static and dynamic) has frame-aligned
    # quaternions — static clips receive identity quats from ToyDataset.
    train_ds = ToyDataset(data_dir, manifest["train_filenames"], use_imu=True)
    val_ds = ToyDataset(data_dir, manifest["val_filenames"], use_imu=True)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=2, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=2,
    )

    print(f"Train clips: {len(train_ds)}, Val clips: {len(val_ds)}")

    # Build model (PhaseCoder includes MicPositionDecoder in its forward)
    model = PhaseCoder().to(device)

    if args.phasecoder_ckpt:
        ckpt = torch.load(args.phasecoder_ckpt, map_location=device, weights_only=False)
        # strict=False: checkpoint may predate mic_pos_decoder; missing keys get
        # random init while the backbone loads correctly.
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded PhaseCoder from {args.phasecoder_ckpt}")
        if missing:
            print(f"  Missing keys (will be randomly init'd): {missing}")
    else:
        print("No checkpoint provided — training from scratch")

    if args.freeze_phasecoder:
        for name, param in model.named_parameters():
            if not name.startswith("mic_pos_decoder"):
                param.requires_grad_(False)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_trainable:,} trainable / {n_total:,} total")

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = nn.MSELoss()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    start_time = time.time()

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        train_losses = []

        for batch in train_loader:
            audio = batch["audio"].to(device)
            mic_coords = batch["mic_coords"].to(device)
            imu = batch["imu_quats"].to(device)

            gt_positions = compute_gt_mic_positions(mic_coords, imu)  # (B, F, C, 3)

            optimizer.zero_grad()
            out = model(audio, mic_coords, imu_orientations=imu)
            loss = criterion(out["mic_positions"], gt_positions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        avg_train_loss = float(np.mean(train_losses))

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                audio = batch["audio"].to(device)
                mic_coords = batch["mic_coords"].to(device)
                imu = batch["imu_quats"].to(device)

                gt_positions = compute_gt_mic_positions(mic_coords, imu)
                out = model(audio, mic_coords, imu_orientations=imu)
                loss = criterion(out["mic_positions"], gt_positions)
                val_losses.append(loss.item())

        avg_val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train {avg_train_loss:.6f} | val {avg_val_loss:.6f} | {elapsed:.0f}s")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val_loss,
                "args": vars(args),
            }, output_dir / "best.pt")

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch,
            "history": history,
            "args": vars(args),
        }, output_dir / "latest.pt")

    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs - 1,
        "args": vars(args),
    }, output_dir / "final.pt")

    print(f"\n✓ Training complete in {(time.time() - start_time)/60:.1f} min")
    print(f"  Best val MSE loss: {best_val_loss:.6f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MicPositionDecoder")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory produced by generate_toy_dataset.py")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Where to save checkpoints and history")
    parser.add_argument("--phasecoder_ckpt", type=str, default=None,
                        help="Path to a pretrained PhaseCoder .pt checkpoint (optional)")
    parser.add_argument("--freeze_phasecoder", type=lambda s: s.lower() == "true",
                        default=True,
                        help="Freeze all PhaseCoder weights; train only MicPositionDecoder "
                             "(default: true)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
