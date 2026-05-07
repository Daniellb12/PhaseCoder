"""
Parallelized Toy Dataset Generator for PhaseCoder

Same exact methodology as generate_toy_dataset.py, but parallelizes clip
simulation across CPU cores using multiprocessing. Designed to scale to
~100k clips on a multi-core machine.

Usage:
    # Use all available cores
    python generate_toy_dataset_parallel.py --num_clips 100000 --output_dir ./big_data

    # Limit to specific number of workers
    python generate_toy_dataset_parallel.py --num_clips 100000 --num_workers 8

Performance estimates (single-clip generation ~0.3s on modern CPU):
    100k clips:
        4 cores:  ~2 hours
        8 cores:  ~1 hour
        16 cores: ~30 minutes
        32 cores: ~15 minutes

Storage: ~90 KB per clip → 100k clips ≈ 9 GB
"""

import argparse
import json
import os
import time
from pathlib import Path
from multiprocessing import Pool, cpu_count

import numpy as np

# Reuse all helpers from the single-threaded generator
from generate_toy_dataset import (
    SAMPLE_RATE, CLIP_DURATION_S, NUM_SAMPLES, N_FFT, HOP_LENGTH, NUM_FRAMES,
    NUM_AZIMUTH, NUM_ELEVATION, NUM_DISTANCE, DISTANCE_BIN_EDGES,
    generate_random_array, simulate_clip, discretize_labels,
)


# ---------------------------------------------------------------------------
# Worker function (runs in subprocess)
# ---------------------------------------------------------------------------

def _generate_one_clip(args):
    """Generate a single clip. Designed to be called by multiprocessing.Pool.

    Args:
        args: tuple of (clip_idx, output_dir_str, seed, fraction_dynamic)

    Returns:
        manifest entry dict, or None if simulation failed.
    """
    clip_idx, output_dir_str, seed, fraction_dynamic = args
    output_dir = Path(output_dir_str)

    # Per-clip RNG seeded deterministically — guarantees reproducibility
    # and ensures different workers don't collide on random state.
    rng = np.random.default_rng(seed + clip_idx)

    try:
        # Random mic geometry per clip
        mic_coords = generate_random_array(rng)

        # Random source location
        source_az = float(rng.uniform(0, 360))
        source_el = float(rng.uniform(-30, 30))
        source_dist = float(np.exp(rng.uniform(np.log(0.5), np.log(5.0))))

        # Static vs dynamic
        is_dynamic = rng.random() < fraction_dynamic
        if is_dynamic:
            rotation_rate = float(rng.choice([30, 60, 100, 150, 200, 300, 450])
                                  * rng.choice([-1, 1]))
            axis = np.array([
                rng.uniform(-0.3, 0.3),
                rng.uniform(-0.3, 0.3),
                rng.choice([-1, 1]) * rng.uniform(0.7, 1.0),
            ], dtype=np.float32)
            axis = axis / np.linalg.norm(axis)
        else:
            rotation_rate = 0.0
            axis = None

        audio, imu_quats = simulate_clip(
            mic_coords=mic_coords,
            source_az_deg=source_az,
            source_el_deg=source_el,
            source_distance=source_dist,
            rng=rng,
            rotation_rate_dps=abs(rotation_rate) if is_dynamic else 0.0,
            rotation_axis=axis,
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
        }
        if is_dynamic:
            save_kwargs["imu_quats"] = imu_quats
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
        # Silent fail — log and skip. We'll report total skipped at end.
        return None


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def generate_dataset_parallel(
    output_dir: Path,
    num_clips: int,
    fraction_dynamic: float,
    num_workers: int,
    seed: int = 0,
    val_fraction: float = 0.1,
    chunksize: int = 16,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {num_clips} clips → {output_dir}")
    print(f"  Workers: {num_workers}")
    print(f"  Fraction dynamic: {fraction_dynamic}")
    print(f"  Estimated dataset size: ~{num_clips * 0.09:.1f} MB")

    # Build argument list for workers
    work_args = [
        (i, str(output_dir), seed, fraction_dynamic)
        for i in range(num_clips)
    ]

    start_time = time.time()
    successful_clips = []
    failed_count = 0

    # Pool with imap_unordered gives streaming progress + work distribution.
    # chunksize controls how many tasks each worker picks up at once.
    with Pool(processes=num_workers) as pool:
        for i, result in enumerate(pool.imap_unordered(_generate_one_clip, work_args,
                                                         chunksize=chunksize)):
            if result is not None:
                successful_clips.append(result)
            else:
                failed_count += 1

            # Progress reporting
            if (i + 1) % max(1, num_clips // 50) == 0 or (i + 1) == num_clips:
                elapsed = time.time() - start_time
                rate = (i + 1) / max(elapsed, 0.001)
                eta = (num_clips - i - 1) / max(rate, 0.001)
                print(f"  [{i+1:6d}/{num_clips}] "
                      f"({100*(i+1)/num_clips:5.1f}%) | "
                      f"{rate:.1f} clips/sec | "
                      f"ETA: {eta/60:.1f} min")

    total_time = time.time() - start_time
    print(f"\nGeneration complete: {len(successful_clips)} clips, "
          f"{failed_count} failed, {total_time/60:.1f} minutes total")

    # Sort by filename so manifest order is deterministic
    successful_clips.sort(key=lambda c: c["filename"])

    # Train/val split (deterministic)
    rng = np.random.default_rng(seed + 999_999)  # different seed from clip generation
    n_total = len(successful_clips)
    n_val = int(n_total * val_fraction)
    val_indices = set(rng.choice(n_total, size=n_val, replace=False).tolist())

    manifest = {
        "num_clips": n_total,
        "num_failed": failed_count,
        "fraction_dynamic": fraction_dynamic,
        "sample_rate": SAMPLE_RATE,
        "clip_duration_s": CLIP_DURATION_S,
        "num_samples": NUM_SAMPLES,
        "n_fft": N_FFT,
        "hop_length": HOP_LENGTH,
        "num_frames": NUM_FRAMES,
        "num_azimuth": NUM_AZIMUTH,
        "num_elevation": NUM_ELEVATION,
        "num_distance": NUM_DISTANCE,
        "distance_bin_edges": DISTANCE_BIN_EDGES.tolist(),
        "generation_time_seconds": total_time,
        "seed": seed,
        "clips": successful_clips,
        "train_filenames": [c["filename"] for i, c in enumerate(successful_clips)
                           if i not in val_indices],
        "val_filenames": [c["filename"] for i, c in enumerate(successful_clips)
                         if i in val_indices],
    }

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  Train: {len(manifest['train_filenames'])}")
    print(f"  Val:   {len(manifest['val_filenames'])}")
    print(f"  Manifest: {output_dir / 'manifest.json'}")

    # Quick statistics on the generated dataset
    dynamic_count = sum(1 for c in successful_clips if c["is_dynamic"])
    rate_distribution = {}
    for c in successful_clips:
        if c["is_dynamic"]:
            rate = abs(c["rotation_rate_dps"])
            bucket = f"{int(rate)}"
            rate_distribution[bucket] = rate_distribution.get(bucket, 0) + 1
    mic_count_distribution = {}
    for c in successful_clips:
        n = c["num_mics"]
        mic_count_distribution[n] = mic_count_distribution.get(n, 0) + 1

    print(f"\nDataset statistics:")
    print(f"  Static clips:  {n_total - dynamic_count}")
    print(f"  Dynamic clips: {dynamic_count}")
    print(f"  Mic count distribution: {dict(sorted(mic_count_distribution.items()))}")
    if rate_distribution:
        print(f"  Rotation rate distribution: "
              f"{dict(sorted(rate_distribution.items(), key=lambda x: int(x[0])))}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parallel toy PhaseCoder dataset generator")
    parser.add_argument("--num_clips", type=int, default=100_000,
                        help="Total clips to generate (default: 100000)")
    parser.add_argument("--output_dir", type=str, default="./big_data",
                        help="Output directory (default: ./big_data)")
    parser.add_argument("--fraction_dynamic", type=float, default=0.5,
                        help="Fraction of clips with rotating array (default: 0.5)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of worker processes (default: cpu_count - 1)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.05,
                        help="Fraction held out for validation (default: 0.05)")
    parser.add_argument("--chunksize", type=int, default=16,
                        help="Tasks per worker batch (default: 16). Larger = less "
                             "overhead, smaller = better load balancing.")
    args = parser.parse_args()

    if args.num_workers is None:
        # Leave one core free for system / I/O
        args.num_workers = max(1, cpu_count() - 1)

    print("=" * 60)
    print("PhaseCoder Toy Dataset Generator (Parallel)")
    print("=" * 60)

    generate_dataset_parallel(
        output_dir=Path(args.output_dir),
        num_clips=args.num_clips,
        fraction_dynamic=args.fraction_dynamic,
        num_workers=args.num_workers,
        seed=args.seed,
        val_fraction=args.val_fraction,
        chunksize=args.chunksize,
    )


if __name__ == "__main__":
    main()
