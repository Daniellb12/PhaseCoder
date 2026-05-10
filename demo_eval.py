"""
Quick demo: load best.pt checkpoint, run forward passes on a few eval examples,
and print predicted vs ground-truth mic positions.
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from PhaseCoder import PhaseCoder
from train_physics import LOCATADataset, N_FRAMES, CLIP_SAMPLES

CKPT   = Path("outputs/best.pt")
LOCATA = Path("LOCATA")
N_SHOW = 5   # how many clips to show


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load checkpoint
    ckpt = torch.load(CKPT, map_location=device, weights_only=False)
    print(f"Checkpoint: epoch {ckpt['epoch']}, val_loss={ckpt['val_loss']:.6f}")
    print(f"Training args: {ckpt.get('args', {})}\n")

    model = PhaseCoder().to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Load eval dataset (just the first array type found)
    ds = LOCATADataset(LOCATA, "eval")
    print(f"\nEval clips available: {len(ds)}")
    print("=" * 70)

    indices = np.linspace(0, len(ds) - 1, N_SHOW, dtype=int)

    with torch.no_grad():
        for i, idx in enumerate(indices):
            item = ds[idx]
            audio      = item["audio"].unsqueeze(0).to(device)       # (1, C, T)
            mic_coords = item["mic_coords"].unsqueeze(0).to(device)  # (1, C, 3)
            gt_pos     = item["gt_positions"]                         # (F, C, 3) cpu

            out = model(audio, mic_coords, imu_orientations=None)
            pred_pos = out["mic_positions"].squeeze(0).cpu()          # (F, C, 3)

            C = mic_coords.shape[1]
            F = gt_pos.shape[0]

            # Per-mic, per-frame L2 error in cm
            err = (pred_pos - gt_pos).norm(dim=-1) * 100.0  # (F, C) in cm

            print(f"\n--- Clip {i+1} (dataset index {idx}) ---")
            print(f"  Mics: {C}, STFT frames: {F}")
            print(f"  Mean error: {err.mean():.2f} cm | "
                  f"Max error: {err.max():.2f} cm\n")

            # Show middle frame for each mic
            mid = F // 2
            print(f"  {'Mic':>4}  {'GT x':>8} {'GT y':>8} {'GT z':>8}  "
                  f"{'Pred x':>8} {'Pred y':>8} {'Pred z':>8}  {'Err(cm)':>8}")
            print(f"  {'-'*4}  {'-'*8} {'-'*8} {'-'*8}  "
                  f"{'-'*8} {'-'*8} {'-'*8}  {'-'*8}")
            for m in range(C):
                g = gt_pos[mid, m]
                p = pred_pos[mid, m]
                e = err[mid, m]
                print(f"  {m:>4}  {g[0]:8.4f} {g[1]:8.4f} {g[2]:8.4f}  "
                      f"{p[0]:8.4f} {p[1]:8.4f} {p[2]:8.4f}  {e:8.2f}")

    print("\n" + "=" * 70)
    print("Done.")


if __name__ == "__main__":
    main()
