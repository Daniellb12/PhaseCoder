"""
Train MicPositionDecoder on the LOCATA dataset (Zenodo #3630471).

LOCATA provides real-room multichannel recordings (48kHz) with optical-tracking
ground truth for both the microphone array and sound source positions (~111 Hz).

Pipeline:
  - Audio is resampled 48kHz → 16kHz for PhaseCoder
  - Per-mic positions are read directly from position_array_*.txt
  - Each STFT frame is aligned to the nearest position sample by absolute timestamp
  - MicPositionDecoder is supervised against absolute per-frame per-mic positions

Dataset layout expected under <locata_root>:
  dev/task{1..6}/recording{N}/<array>/   →  training
  eval/task{1..6}/recording{N}/<array>/  →  validation

Each <array> folder must contain:
  audio_array_<array>.wav
  audio_array_timestamps_<array>.txt
  position_array_<array>.txt

Supported arrays: benchmark2 (12 mics), dicit (15), eigenmike (32), dummy (4)

Usage:
    # Decoder only from a pretrained PhaseCoder checkpoint
    python train_physics.py \\
        --locata_root ./3630471 \\
        --output_dir ./runs/locata_decoder \\
        --phasecoder_ckpt ./runs/imu/best.pt \\
        --freeze_phasecoder true

    # Joint fine-tuning from scratch
    python train_physics.py \\
        --locata_root ./3630471 \\
        --output_dir ./runs/locata_scratch \\
        --freeze_phasecoder false

    # Only moving-array tasks, eigenmike only
    python train_physics.py \\
        --locata_root ./3630471 \\
        --output_dir ./runs/locata_moving \\
        --tasks task5 task6 \\
        --arrays eigenmike
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.io import wavfile
from scipy.signal import resample_poly
from torch.utils.data import Dataset, DataLoader

from PhaseCoder import PhaseCoder, STFTPatchExtractor


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SOURCE_SR   = 48000   # LOCATA recording sample rate
TARGET_SR   = 16000   # PhaseCoder input sample rate
RESAMPLE_UP = 1
RESAMPLE_DN = SOURCE_SR // TARGET_SR   # = 3

N_FFT      = 256
HOP_LENGTH = 128
CLIP_SAMPLES = 4000   # 250 ms at 16 kHz

MIC_COLS_START = 21   # column index where mic1_x begins in position_array_*.txt

VALID_ARRAYS = ("benchmark2", "dicit", "eigenmike", "dummy")

# Compute the number of STFT frames produced by PhaseCoder for one 250ms clip.
with torch.no_grad():
    _probe = STFTPatchExtractor(N_FFT, HOP_LENGTH, N_FFT)(torch.zeros(1, 1, CLIP_SAMPLES))
N_FRAMES = _probe.shape[2]   # typically 32


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def _row_to_seconds(row: np.ndarray) -> float:
    """Convert a [year, month, day, hour, minute, second] row to seconds-since-midnight."""
    return float(row[3]) * 3600.0 + float(row[4]) * 60.0 + float(row[5])


def _read_start_timestamp(ts_path: Path) -> float:
    """Read the first data row of an audio timestamps file to get t=0 for that recording."""
    with open(ts_path) as fh:
        fh.readline()           # header
        row = list(map(float, fh.readline().split()))
    return row[3] * 3600.0 + row[4] * 60.0 + row[5]


def _load_position_array(pos_path: Path):
    """
    Parse position_array_<array>.txt.

    Returns:
        pos_ts:       (N,)       absolute seconds-since-midnight for each row
        mic_positions:(N, M, 3)  per-mic (x, y, z) positions
    """
    data = np.loadtxt(pos_path, skiprows=1)  # (N, 21 + M*3)
    pos_ts = data[:, 3] * 3600.0 + data[:, 4] * 60.0 + data[:, 5]
    n_mic_cols = data.shape[1] - MIC_COLS_START
    n_mics = n_mic_cols // 3
    mic_positions = data[:, MIC_COLS_START:].reshape(-1, n_mics, 3).astype(np.float32)
    return pos_ts, mic_positions


def _load_resample_audio(wav_path: Path) -> np.ndarray:
    """
    Load a LOCATA WAV (48kHz float64) and resample to 16kHz float32.

    Returns:
        audio: (C, T_16k)  float32
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sr, data = wavfile.read(str(wav_path))
    assert sr == SOURCE_SR, f"Expected {SOURCE_SR} Hz, got {sr} in {wav_path}"
    if data.ndim == 1:
        data = data[:, np.newaxis]
    # resample_poly operates on axis 0 → (T_48k, C) → (T_16k, C)
    data_16k = resample_poly(data.astype(np.float32), RESAMPLE_UP, RESAMPLE_DN, axis=0)
    return data_16k.T  # (C, T_16k)


def _align_positions(frame_abs_times: np.ndarray,
                     pos_ts: np.ndarray,
                     mic_pos: np.ndarray) -> np.ndarray:
    """
    Nearest-neighbour alignment of position samples to STFT frame timestamps.

    Args:
        frame_abs_times: (F,)      absolute seconds for each STFT frame centre
        pos_ts:          (N,)      absolute seconds for each position sample
        mic_pos:         (N, M, 3) per-mic positions

    Returns:
        (F, M, 3) ground-truth mic positions aligned to each STFT frame
    """
    idx = np.searchsorted(pos_ts, frame_abs_times)
    idx = np.clip(idx, 0, len(pos_ts) - 1)
    prev = np.maximum(idx - 1, 0)
    use_prev = np.abs(pos_ts[prev] - frame_abs_times) < np.abs(pos_ts[idx] - frame_abs_times)
    idx = np.where(use_prev, prev, idx)
    return mic_pos[idx]   # (F, M, 3)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LOCATADataset(Dataset):
    """
    250ms chunks from one split (dev or eval) of the LOCATA dataset.

    Items:
        audio        : (C, CLIP_SAMPLES)   float32 at 16 kHz
        mic_coords   : (C, 3)              neutral-frame mic positions (middle STFT frame)
        gt_positions : (N_FRAMES, C, 3)    ground-truth per-frame mic positions
    """

    def __init__(
        self,
        locata_root: Path,
        split: str,                               # "dev" or "eval"
        arrays: tuple = VALID_ARRAYS,
        tasks: tuple | None = None,               # None → all tasks
    ):
        self._chunks: list[tuple] = []
        split_dir = locata_root / split
        if not split_dir.exists():
            raise FileNotFoundError(f"Split not found: {split_dir}")
        self._scan(split_dir, set(arrays), tasks)
        print(f"  [{split}] {len(self._chunks)} clips from {split_dir}")

    def _scan(self, split_dir: Path, arrays: set, tasks):
        for task_dir in sorted(split_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            if tasks is not None and task_dir.name not in tasks:
                continue
            for rec_dir in sorted(task_dir.iterdir()):
                if not rec_dir.is_dir():
                    continue
                for arr_dir in sorted(rec_dir.iterdir()):
                    if not arr_dir.is_dir() or arr_dir.name not in arrays:
                        continue
                    try:
                        self._load_recording(arr_dir)
                    except Exception as exc:
                        print(f"  Warning: skipping {arr_dir.relative_to(arr_dir.parent.parent.parent)}: {exc}")

    def _load_recording(self, arr_dir: Path):
        arr = arr_dir.name
        wav_path = arr_dir / f"audio_array_{arr}.wav"
        ts_path  = arr_dir / f"audio_array_timestamps_{arr}.txt"
        pos_path = arr_dir / f"position_array_{arr}.txt"

        if not (wav_path.exists() and ts_path.exists() and pos_path.exists()):
            return

        audio = _load_resample_audio(wav_path)    # (C, T_16k)
        C, T  = audio.shape

        pos_ts, mic_pos = _load_position_array(pos_path)   # (N,), (N, M, 3)
        t_start = _read_start_timestamp(ts_path)            # seconds-since-midnight

        # Stride through the recording in non-overlapping 250ms windows
        chunk_start = 0
        while chunk_start + CLIP_SAMPLES <= T:
            # Absolute times of each STFT frame centre for this chunk
            frame_abs = t_start + (chunk_start + np.arange(N_FRAMES) * HOP_LENGTH) / TARGET_SR

            # Skip chunks whose frames fall outside the position-data range
            if frame_abs[0] < pos_ts[0] or frame_abs[-1] > pos_ts[-1]:
                chunk_start += CLIP_SAMPLES
                continue

            gt_pos = _align_positions(frame_abs, pos_ts, mic_pos)  # (F, M, 3)

            # Neutral frame = middle STFT frame (mirrors train_physics.py convention)
            mic_coords = gt_pos[N_FRAMES // 2]   # (M, 3)

            self._chunks.append((
                torch.from_numpy(audio[:, chunk_start:chunk_start + CLIP_SAMPLES].copy()),
                torch.from_numpy(mic_coords.copy()),
                torch.from_numpy(gt_pos.copy()),
            ))
            chunk_start += CLIP_SAMPLES

    def __len__(self) -> int:
        return len(self._chunks)

    def __getitem__(self, idx: int) -> dict:
        audio, mic_coords, gt_positions = self._chunks[idx]
        return {
            "audio":        audio,        # (C, CLIP_SAMPLES)
            "mic_coords":   mic_coords,   # (C, 3)
            "gt_positions": gt_positions, # (N_FRAMES, C, 3)
        }


def collate_fn(batch: list[dict]) -> dict:
    """Pad mic dimension to the maximum count in the batch."""
    max_mics = max(item["mic_coords"].shape[0] for item in batch)
    B = len(batch)
    T = CLIP_SAMPLES
    F = N_FRAMES

    audio       = torch.zeros(B, max_mics, T)
    mic_coords  = torch.zeros(B, max_mics, 3)
    gt_positions = torch.zeros(B, F, max_mics, 3)

    for i, item in enumerate(batch):
        C = item["mic_coords"].shape[0]
        audio[i, :C]              = item["audio"]
        mic_coords[i, :C]         = item["mic_coords"]
        gt_positions[i, :, :C]    = item["gt_positions"]
        if C < max_mics:
            # Pad extra slots with the centroid so they don't bias the loss
            centroid    = item["mic_coords"].mean(dim=0)            # (3,)
            gt_centroid = item["gt_positions"].mean(dim=1)          # (F, 3)
            mic_coords[i, C:]         = centroid
            gt_positions[i, :, C:]    = gt_centroid.unsqueeze(1).expand(-1, max_mics - C, -1)

    return {"audio": audio, "mic_coords": mic_coords, "gt_positions": gt_positions}


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"STFT frames per clip: {N_FRAMES}")
    print(f"Freeze PhaseCoder backbone: {args.freeze_phasecoder}")

    locata_root = Path(args.locata_root)
    arrays = tuple(args.arrays)
    tasks  = tuple(args.tasks) if args.tasks else None
    print(f"Arrays: {arrays}")
    print(f"Tasks:  {tasks if tasks else 'all'}")

    train_ds = LOCATADataset(locata_root, "dev",  arrays=arrays, tasks=tasks)
    val_ds   = LOCATADataset(locata_root, "eval", arrays=arrays, tasks=tasks)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers,
    )

    # Build model
    model = PhaseCoder().to(device)

    if args.phasecoder_ckpt:
        ckpt = torch.load(args.phasecoder_ckpt, map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded PhaseCoder from {args.phasecoder_ckpt}")
        if missing:
            print(f"  Missing keys (randomly init'd): {missing}")
    else:
        print("No checkpoint — training from scratch")

    if args.freeze_phasecoder:
        for name, param in model.named_parameters():
            if not name.startswith("mic_pos_decoder"):
                param.requires_grad_(False)

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total     = sum(p.numel() for p in model.parameters())
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
            audio        = batch["audio"].to(device)         # (B, C, T)
            mic_coords   = batch["mic_coords"].to(device)    # (B, C, 3)
            gt_positions = batch["gt_positions"].to(device)  # (B, F, C, 3)

            optimizer.zero_grad()
            # Run PhaseCoder in static mode (no IMU); decoder predicts per-frame positions
            out  = model(audio, mic_coords, imu_orientations=None)
            loss = criterion(out["mic_positions"], gt_positions)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        avg_train = float(np.mean(train_losses))

        # --- Validate ---
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                audio        = batch["audio"].to(device)
                mic_coords   = batch["mic_coords"].to(device)
                gt_positions = batch["gt_positions"].to(device)

                out  = model(audio, mic_coords, imu_orientations=None)
                loss = criterion(out["mic_positions"], gt_positions)
                val_losses.append(loss.item())

        avg_val = float(np.mean(val_losses)) if val_losses else float("nan")
        history["train_loss"].append(avg_train)
        history["val_loss"].append(avg_val)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"train {avg_train:.6f} | val {avg_val:.6f} | {elapsed:.0f}s")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_loss": avg_val,
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

    with open(output_dir / "history.json", "w") as fh:
        json.dump(history, fh, indent=2)
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": args.epochs - 1,
        "args": vars(args),
    }, output_dir / "final.pt")

    print(f"\nTraining complete in {(time.time() - start_time)/60:.1f} min")
    print(f"Best val MSE: {best_val_loss:.6f}")
    print(f"Checkpoints: {output_dir}/best.pt, {output_dir}/final.pt")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Train MicPositionDecoder on LOCATA")
    parser.add_argument("--locata_root", type=str, default="LOCATA/",
                        help="Root directory of the LOCATA dataset (contains dev/ and eval/)")
    parser.add_argument("--output_dir", type=str, default="outputs/",
                        help="Where to save checkpoints and history")
    parser.add_argument("--phasecoder_ckpt", type=str, default=None,
                        help="Path to a pretrained PhaseCoder .pt checkpoint (optional)")
    parser.add_argument("--freeze_phasecoder", type=lambda s: s.lower() == "true",
                        default=True,
                        help="Freeze PhaseCoder backbone; train only MicPositionDecoder (default: true)")
    parser.add_argument("--arrays", nargs="+", default=list(VALID_ARRAYS),
                        choices=list(VALID_ARRAYS),
                        help="Which mic array type(s) to include (default: all)")
    parser.add_argument("--tasks", nargs="+", default=None,
                        help="Which tasks to include, e.g. task3 task4 (default: all)")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num_workers", type=int, default=2)
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
