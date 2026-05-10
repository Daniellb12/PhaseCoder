"""
Time-Varying RIR Generation for Dynamic Microphone Arrays

Generates physically-accurate audio for clips where the microphone array
rotates DURING the clip (not just before/after). This captures the
within-frame motion blur that the canonical-frame approximation misses.

Methodology:
    1. Sample N rotation snapshots along the array's trajectory through the clip
    2. Compute a separate RIR at each snapshot (each represents a different
       array orientation in the same room)
    3. Convolve the source signal with each RIR independently
    4. Crossfade between the resulting multichannel audios using overlapping
       windowed segments to produce a single time-varying output

This is standard time-varying linear system simulation: when the system is
slowly varying, you can approximate the output by convolving with a different
impulse response in each short window and overlap-adding the results.

Performance:
    - ~5-10 minutes per dynamic clip on CPU (depends on RIR snapshot count)
    - Worth it for research validation; not feasible for 100k clips at scale
    - Recommended: generate 1-5k clips this way for a "high fidelity" eval set

Usage:
    python generate_time_varying_rir.py --num_clips 200 --output_dir ./tv_rir_data
"""

import argparse
import json
import math
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np
import pyroomacoustics as pra
from scipy.signal import fftconvolve

from librispeech_source import LibriSpeechProvider

# Resolve to absolute path so multiprocessing workers find it regardless of cwd
_CACHE_DIR_ABS = str(Path("./librispeech_cache").resolve())

# Lazy-initialized per worker process for multiprocessing safety
_speech_provider = None

# ---------------------------------------------------------------------------
# Constants matching PhaseCoder paper
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
CLIP_DURATION_S = 0.25
NUM_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_S)  # 4000
N_FFT = 256
HOP_LENGTH = 128
NUM_FRAMES = 32

NUM_AZIMUTH = 38
NUM_ELEVATION = 18
NUM_DISTANCE = 13
DISTANCE_BIN_EDGES = np.array(
    [0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 7.0]
)


# ---------------------------------------------------------------------------
# Helpers (same as toy generator)
# ---------------------------------------------------------------------------

def axis_angle_to_quaternion(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = angle_rad / 2
    s = math.sin(half)
    return np.array(
        [math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s],
        dtype=np.float32,
    )


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def spherical_to_cartesian(az_deg: float, el_deg: float, distance: float) -> np.ndarray:
    az = math.radians(az_deg)
    el = math.radians(el_deg)
    return np.array(
        [
            distance * math.cos(el) * math.cos(az),
            distance * math.cos(el) * math.sin(az),
            distance * math.sin(el),
        ],
        dtype=np.float32,
    )


def discretize_labels(az_deg: float, el_deg: float, dist: float) -> dict:
    az_bin = min(int(az_deg / 360 * NUM_AZIMUTH), NUM_AZIMUTH - 1)
    el_bin = min(max(int((el_deg + 90) / 180 * NUM_ELEVATION), 0), NUM_ELEVATION - 1)
    dist_bin = min(int(np.searchsorted(DISTANCE_BIN_EDGES, dist)), NUM_DISTANCE - 1)
    return {"azimuth": az_bin, "elevation": el_bin, "distance": dist_bin}


def generate_random_array(rng: np.random.Generator) -> np.ndarray:
    num_mics = rng.integers(4, 9)
    radius = rng.uniform(0.04, 0.10)
    angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)
    angles += rng.uniform(-0.1, 0.1, size=num_mics)
    coords = np.stack(
        [
            radius * np.cos(angles),
            radius * np.sin(angles),
            rng.uniform(-0.01, 0.01, size=num_mics),
        ],
        axis=-1,
    )
    return coords.astype(np.float32)


def generate_source_signal(duration_s: float, rng: np.random.Generator) -> np.ndarray:
    """Get a real-speech source signal from LibriSpeech (with synthetic fallback)."""
    global _speech_provider
    if _speech_provider is None:
        _speech_provider = LibriSpeechProvider(
            cache_dir=_CACHE_DIR_ABS,
            subset="dev-clean",
            max_utterances=2000,
            synthetic_fraction=0.1,  # 10% synthetic for source diversity
            verbose=True,
        )
        import os
        print(f"  [PID {os.getpid()}] Provider initialized. "
              f"using_synthetic_only={_speech_provider.using_synthetic_only}, "
              f"flac_files={len(_speech_provider.flac_files)}")
    return _speech_provider.get_signal(duration_s, rng)


# ---------------------------------------------------------------------------
# The core time-varying RIR machinery
# ---------------------------------------------------------------------------

def compute_snapshot_rir(
    room_dim: np.ndarray,
    e_absorption: float,
    max_order: int,
    centroid: np.ndarray,
    mic_coords_rotated: np.ndarray,
    source_world: np.ndarray,
    src_signal: np.ndarray,
) -> np.ndarray:
    """Convolve a source signal with one fixed array snapshot.

    Returns:
        Multichannel audio of shape (C, T) at SAMPLE_RATE.
    """
    C = mic_coords_rotated.shape[0]
    room = pra.ShoeBox(
        room_dim,
        fs=SAMPLE_RATE,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )

    mic_world = (centroid[None, :] + mic_coords_rotated).T  # (3, C)
    room.add_microphone_array(pra.MicrophoneArray(mic_world, fs=SAMPLE_RATE))
    room.add_source(source_world, signal=src_signal)
    room.simulate()

    return room.mic_array.signals  # (C, T_sim)


def time_varying_rir_simulation(
    mic_coords: np.ndarray,
    source_az_deg: float,
    source_el_deg: float,
    source_distance: float,
    rotation_rate_dps: float,
    rotation_axis: np.ndarray,
    rng: np.random.Generator,
    num_snapshots: int = 8,
    crossfade_overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate audio for a clip with array rotation DURING the clip.

    Args:
        mic_coords:        (C, 3) neutral mic positions in meters.
        source_az_deg:     source azimuth at canonical frame center.
        source_el_deg:     source elevation at canonical frame center.
        source_distance:   source distance from array centroid (m).
        rotation_rate_dps: array rotation rate in deg/sec.
        rotation_axis:     (3,) unit axis of rotation.
        rng:               numpy random generator.
        num_snapshots:     how many array orientations to simulate along the
                           rotation trajectory. More = more accurate, slower.
                           8 captures most within-clip motion blur effects.
        crossfade_overlap: fraction of segment that overlaps with neighbors
                           for crossfading. 0.5 = 50% overlap (recommended).

    Returns:
        audio:    (C, NUM_SAMPLES) float32 multichannel waveform.
        imu_quats: (NUM_FRAMES, 4) float32 quaternions per STFT frame.
    """
    C = mic_coords.shape[0]

    # --- 1. Define room and source position (constant across snapshots) ---
    room_dim = np.array(
        [
            rng.uniform(4.0, 8.0),
            rng.uniform(3.0, 6.0),
            rng.uniform(2.5, 3.5),
        ]
    )
    rt60 = rng.uniform(0.15, 0.35)
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    centroid = np.array(
        [
            rng.uniform(1.0, room_dim[0] - 1.0),
            rng.uniform(1.0, room_dim[1] - 1.0),
            rng.uniform(1.2, 1.8),
        ]
    )

    source_offset = spherical_to_cartesian(source_az_deg, source_el_deg, source_distance)
    source_world = np.clip(centroid + source_offset, 0.2, room_dim - 0.2)

    # --- 2. Compute rotation snapshots ---
    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Distribute snapshots evenly across the clip duration, plus a small lead
    # so the first snapshot covers any RIR that "starts" before t=0
    snapshot_times = np.linspace(0, CLIP_DURATION_S, num_snapshots)
    snapshot_angles_rad = np.radians(rotation_rate_dps * snapshot_times)
    snapshot_quats = np.stack(
        [axis_angle_to_quaternion(rotation_axis, a) for a in snapshot_angles_rad],
        axis=0,
    )

    # Rotate mic_coords for each snapshot
    snapshot_mic_coords = []
    for q in snapshot_quats:
        R = quaternion_to_rotation_matrix(q)
        snapshot_mic_coords.append(mic_coords @ R.T)

    # --- 3. Generate source signal (2s for clean crossfade behavior) ---
    src_signal = generate_source_signal(CLIP_DURATION_S * 8.0, rng)

    # --- 4. Simulate each snapshot independently ---
    snapshot_audios = []
    for mic_rotated in snapshot_mic_coords:
        sim = compute_snapshot_rir(
            room_dim=room_dim,
            e_absorption=e_absorption,
            max_order=max_order,
            centroid=centroid,
            mic_coords_rotated=mic_rotated,
            source_world=source_world,
            src_signal=src_signal,
        )
        snapshot_audios.append(sim)

    # All snapshot audios should have the same length (same source signal,
    # same room max_order). Pad to the same length defensively.
    max_len = max(s.shape[1] for s in snapshot_audios)
    snapshot_audios = [
        np.pad(s, ((0, 0), (0, max_len - s.shape[1])), mode="constant")
        for s in snapshot_audios
    ]
    snapshot_audios = np.stack(snapshot_audios, axis=0)  # (S, C, T_sim)

    # --- 5. Crossfade between snapshots over time ---
    # Each snapshot s is "valid" centered at time snapshot_times[s].
    # We construct overlapping triangular windows over the clip duration.
    # The output sample at time t is a weighted sum of snapshot_audios[s]
    # weighted by w_s(t), with sum_s w_s(t) = 1.

    T_sim = snapshot_audios.shape[2]
    output = np.zeros((C, T_sim), dtype=np.float32)
    weights_total = np.zeros(T_sim, dtype=np.float32)
    sample_times = np.arange(T_sim) / SAMPLE_RATE

    for s in range(num_snapshots):
        # Triangular weighting centered at this snapshot's time
        if num_snapshots == 1:
            w = np.ones_like(sample_times)
        else:
            # Width is twice the spacing for 50% overlap (with crossfade_overlap=0.5)
            spacing = CLIP_DURATION_S / (num_snapshots - 1)
            half_width = spacing * (1 + crossfade_overlap)
            center = snapshot_times[s]
            distance = np.abs(sample_times - center)
            w = np.maximum(0, 1 - distance / half_width)

        output += w[None, :] * snapshot_audios[s]
        weights_total += w

    # Normalize where weights sum to non-zero
    safe_weights = np.maximum(weights_total, 1e-8)
    output = output / safe_weights[None, :]

    # --- 6. Truncate to NUM_SAMPLES, normalize ---
    # The crossfade weights are only nonzero during [0, CLIP_DURATION_S].
    # Find the actual valid range from weights_total to avoid cropping into the
    # zero region beyond the crossfade window.
    valid_mask = weights_total > 1e-6  # places where snapshots actually contributed
    valid_indices = np.where(valid_mask)[0]

    if len(valid_indices) >= NUM_SAMPLES:
        valid_start_sample = valid_indices[0]
        valid_end_sample = valid_indices[-1] - NUM_SAMPLES
        
        # Add a small settling margin
        settling = int(0.005 * SAMPLE_RATE)  # 5ms
        valid_start_sample = max(valid_start_sample + settling, 0)
        
        if valid_end_sample > valid_start_sample:
            start = int(rng.integers(valid_start_sample, valid_end_sample + 1))
        else:
            start = valid_start_sample
        
        audio = output[:, start:start + NUM_SAMPLES]
    else:
        # Fallback: not enough valid signal, use whatever we have
        audio = np.zeros((C, NUM_SAMPLES), dtype=np.float32)
        take = min(NUM_SAMPLES, T_sim)
        audio[:, :take] = output[:, :take]

        peak = np.max(np.abs(audio))
        if peak > 1e-8:
            audio = (audio / peak * 0.7).astype(np.float32)

    # --- 7. Compute IMU quaternions for STFT frames (separate from snapshots) ---
    frame_times = (np.arange(NUM_FRAMES) * HOP_LENGTH + N_FFT / 2) / SAMPLE_RATE
    frame_angles_rad = np.radians(rotation_rate_dps * frame_times)
    imu_quats = np.stack(
        [axis_angle_to_quaternion(rotation_axis, a) for a in frame_angles_rad],
        axis=0,
    )

    return audio.astype(np.float32), imu_quats.astype(np.float32)


# ---------------------------------------------------------------------------
# Worker for parallel generation
# ---------------------------------------------------------------------------

def generate_one_clip(args):
    clip_idx, output_dir_str, seed, fraction_dynamic, num_snapshots = args
    output_dir = Path(output_dir_str)
    rng = np.random.default_rng(seed + clip_idx)

    try:
        mic_coords = generate_random_array(rng)
        source_az = float(rng.uniform(0, 360))
        source_el = float(rng.uniform(-30, 30))
        source_dist = float(np.exp(rng.uniform(np.log(0.5), np.log(5.0))))

        is_dynamic = rng.random() < fraction_dynamic
        if is_dynamic:
            rotation_rate = float(
                rng.choice([100, 150, 200, 300, 450, 600, 800])
                * rng.choice([-1, 1])
            )
            axis = np.array(
                [
                    rng.uniform(-0.3, 0.3),
                    rng.uniform(-0.3, 0.3),
                    rng.choice([-1, 1]) * rng.uniform(0.7, 1.0),
                ],
                dtype=np.float32,
            )
            axis = axis / np.linalg.norm(axis)
            audio, imu_quats = time_varying_rir_simulation(
                mic_coords=mic_coords,
                source_az_deg=source_az,
                source_el_deg=source_el,
                source_distance=source_dist,
                rotation_rate_dps=abs(rotation_rate),
                rotation_axis=axis,
                rng=rng,
                num_snapshots=num_snapshots,
            )
        else:
            # Static clip: single snapshot is fine
            rotation_rate = 0.0
            audio, imu_quats = time_varying_rir_simulation(
                mic_coords=mic_coords,
                source_az_deg=source_az,
                source_el_deg=source_el,
                source_distance=source_dist,
                rotation_rate_dps=0.0,
                rotation_axis=np.array([0, 0, 1.0], dtype=np.float32),
                rng=rng,
                num_snapshots=1,
            )

        labels = discretize_labels(source_az, source_el, source_dist)
        clip_path = output_dir / f"clip_{clip_idx:06d}.npz"
        save_kwargs = {
            "audio": audio,
            "mic_coords": mic_coords,
            "azimuth_class": np.int64(labels["azimuth"]),
            "elevation_class": np.int64(labels["elevation"]),
            "distance_class": np.int64(labels["distance"]),
            "azimuth_deg": np.float32(source_az),
            "elevation_deg": np.float32(source_el),
            "distance_m": np.float32(source_dist),
            "is_dynamic": np.bool_(is_dynamic),
            "rotation_rate_dps": np.float32(rotation_rate),
            "imu_quats": imu_quats,
            "num_snapshots": np.int64(num_snapshots if is_dynamic else 1),
        }
        np.savez_compressed(clip_path, **save_kwargs)

        return {
            "filename": clip_path.name,
            "is_dynamic": bool(is_dynamic),
            "rotation_rate_dps": float(rotation_rate),
            "azimuth_deg": float(source_az),
            "elevation_deg": float(source_el),
            "distance_m": float(source_dist),
            "num_mics": int(mic_coords.shape[0]),
        }
    except Exception as e:
        print(f"  Clip {clip_idx} failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def generate_dataset(
    output_dir: Path,
    num_clips: int,
    fraction_dynamic: float,
    num_workers: int,
    num_snapshots: int = 8,
    seed: int = 0,
    val_fraction: float = 0.1,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_clips} time-varying RIR clips → {output_dir}")
    print(f"  Workers: {num_workers}")
    print(f"  Snapshots per dynamic clip: {num_snapshots}")
    print(f"  Estimated time: ~{num_clips * num_snapshots * 0.3 / num_workers / 60:.1f} min")

    work_args = [
        (i, str(output_dir), seed, fraction_dynamic, num_snapshots)
        for i in range(num_clips)
    ]

    start_time = time.time()
    successful = []
    failed = 0

    with Pool(processes=num_workers) as pool:
        for i, result in enumerate(
            pool.imap_unordered(generate_one_clip, work_args, chunksize=4)
        ):
            if result is not None:
                successful.append(result)
            else:
                failed += 1

            if (i + 1) % max(1, num_clips // 50) == 0 or (i + 1) == num_clips:
                elapsed = time.time() - start_time
                rate = (i + 1) / max(elapsed, 0.001)
                eta_min = (num_clips - i - 1) / max(rate, 0.001) / 60
                print(
                    f"  [{i+1:6d}/{num_clips}] {100*(i+1)/num_clips:5.1f}% | "
                    f"{rate:.2f} clips/sec | ETA: {eta_min:.1f} min"
                )

    total_time = time.time() - start_time
    print(f"\n✓ {len(successful)} clips ({failed} failed) in {total_time/60:.1f} min")

    successful.sort(key=lambda c: c["filename"])
    rng = np.random.default_rng(seed + 999_999)
    n_val = int(len(successful) * val_fraction)
    val_indices = set(rng.choice(len(successful), size=n_val, replace=False).tolist())

    manifest = {
        "num_clips": len(successful),
        "fraction_dynamic": fraction_dynamic,
        "num_snapshots": num_snapshots,
        "sample_rate": SAMPLE_RATE,
        "num_frames": NUM_FRAMES,
        "num_azimuth": NUM_AZIMUTH,
        "num_elevation": NUM_ELEVATION,
        "num_distance": NUM_DISTANCE,
        "clips": successful,
        "train_filenames": [
            c["filename"] for i, c in enumerate(successful) if i not in val_indices
        ],
        "val_filenames": [
            c["filename"] for i, c in enumerate(successful) if i in val_indices
        ],
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Train: {len(manifest['train_filenames'])}")
    print(f"  Val:   {len(manifest['val_filenames'])}")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _run_tests():
    """Sanity-check the time-varying RIR logic before generating real data."""
    print("Running tests...\n")
    rng = np.random.default_rng(42)

    # Test 1: static clip (1 snapshot) should produce sensible audio
    print("Test 1: Static clip (num_snapshots=1)")
    mic_coords = generate_random_array(rng)
    audio, quats = time_varying_rir_simulation(
        mic_coords=mic_coords,
        source_az_deg=90.0,
        source_el_deg=0.0,
        source_distance=1.5,
        rotation_rate_dps=0.0,
        rotation_axis=np.array([0, 0, 1.0]),
        rng=rng,
        num_snapshots=1,
    )
    assert audio.shape == (mic_coords.shape[0], NUM_SAMPLES), f"Bad audio shape: {audio.shape}"
    assert quats.shape == (NUM_FRAMES, 4), f"Bad quats shape: {quats.shape}"
    assert np.abs(audio).max() <= 1.0, f"Audio not normalized: {np.abs(audio).max()}"
    print(f"  ✓ Audio shape {audio.shape}, peak {np.abs(audio).max():.3f}")

    # Test 2: dynamic clip with multiple snapshots
    print("\nTest 2: Dynamic clip (200°/sec, num_snapshots=8)")
    mic_coords = generate_random_array(rng)
    audio_dyn, quats_dyn = time_varying_rir_simulation(
        mic_coords=mic_coords,
        source_az_deg=90.0,
        source_el_deg=0.0,
        source_distance=1.5,
        rotation_rate_dps=200.0,
        rotation_axis=np.array([0, 0, 1.0]),
        rng=rng,
        num_snapshots=8,
    )
    assert audio_dyn.shape == (mic_coords.shape[0], NUM_SAMPLES)
    print(f"  ✓ Audio shape {audio_dyn.shape}, peak {np.abs(audio_dyn).max():.3f}")
    print(f"  Quaternion drift: frame 0 → {NUM_FRAMES-1}: "
          f"first w={quats_dyn[0, 0]:.3f}, last w={quats_dyn[-1, 0]:.3f}")

    # Test 3: high rotation rate should produce within-frame phase drift
    print("\nTest 3: Very fast rotation (800°/sec) should differ from static")
    rng2 = np.random.default_rng(42)
    mic_static = generate_random_array(rng2)
    audio_static_check, _ = time_varying_rir_simulation(
        mic_coords=mic_static,
        source_az_deg=90.0,
        source_el_deg=0.0,
        source_distance=1.5,
        rotation_rate_dps=0.0,
        rotation_axis=np.array([0, 0, 1.0]),
        rng=rng2,
        num_snapshots=1,
    )
    rng3 = np.random.default_rng(42)
    mic_dyn = generate_random_array(rng3)
    audio_fast, _ = time_varying_rir_simulation(
        mic_coords=mic_dyn,
        source_az_deg=90.0,
        source_el_deg=0.0,
        source_distance=1.5,
        rotation_rate_dps=800.0,
        rotation_axis=np.array([0, 0, 1.0]),
        rng=rng3,
        num_snapshots=8,
    )
    diff = np.abs(audio_static_check - audio_fast).mean()
    print(f"  Mean abs diff between static & 800°/s clips: {diff:.4f}")
    assert diff > 0.01, "Fast rotation should produce noticeably different audio"
    print(f"  ✓ Fast-rotation audio differs from static (as expected)")

    print("\n✓ All tests passed.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate time-varying RIR dataset for PhaseCoder"
    )
    parser.add_argument("--num_clips", type=int, default=200)
    parser.add_argument("--output_dir", type=str, default="./tv_rir_data")
    parser.add_argument("--fraction_dynamic", type=float, default=0.7)
    parser.add_argument("--num_snapshots", type=int, default=8,
                        help="RIR snapshots per dynamic clip (more=more accurate)")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test", action="store_true",
                        help="Run sanity tests instead of generating data")
    args = parser.parse_args()

    if args.test:
        _run_tests()
        return

    if args.num_workers is None:
        args.num_workers = max(1, cpu_count() - 1)

    print("=" * 60)
    print("Time-Varying RIR Dataset Generator")
    print("=" * 60)
    print(f"Note: This is ~{args.num_snapshots}× slower than static-RIR generation.")
    print(f"Recommended for small high-fidelity datasets (200-5000 clips).")
    print("=" * 60)

    generate_dataset(
        output_dir=Path(args.output_dir),
        num_clips=args.num_clips,
        fraction_dynamic=args.fraction_dynamic,
        num_workers=args.num_workers,
        num_snapshots=args.num_snapshots,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )


if __name__ == "__main__":
    main()
