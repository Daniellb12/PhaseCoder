"""
Toy Synthetic Dataset Generator for PhaseCoder Demo

Uses pyroomacoustics' image-source method to generate spatial audio clips
following the same methodology as the PhaseCoder paper (Dementyev et al., 2026),
just at a tiny scale suitable for a laptop demo.

Generates two types of clips:
    - 'static':  fixed mic array, single source direction. Ground-truth (az, el, dist).
    - 'dynamic': mic array rotates during the clip. Ground-truth orientation
                 quaternion per STFT frame, plus source direction at canonical frame.

Output structure:
    output_dir/
        clip_00000.npz      # contains audio, mic_coords, labels, optional imu_quats
        clip_00001.npz
        ...
        manifest.json       # dataset metadata, train/val splits

Each .npz contains:
    audio        : (C, T) float32           — multichannel waveform
    mic_coords   : (C, 3) float32           — neutral-frame mic positions (m)
    azimuth_class: scalar int               — discretized azimuth bin
    elevation_class: scalar int             — discretized elevation bin
    distance_class: scalar int              — discretized distance bin
    azimuth_deg  : scalar float             — continuous ground truth (debug)
    elevation_deg: scalar float             — continuous ground truth (debug)
    distance_m   : scalar float             — continuous ground truth (debug)
    is_dynamic   : scalar bool              — whether this clip has IMU data
    imu_quats    : (F, 4) float32           — only present if is_dynamic; quaternions
                                              per STFT frame in (w,x,y,z) order
    rotation_rate_dps: scalar float         — peak rotation rate in deg/sec (debug)
"""

import argparse
import math
import json
from pathlib import Path

import numpy as np
import pyroomacoustics as pra


# ---------------------------------------------------------------------------
# Configuration matching PhaseCoder paper
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
CLIP_DURATION_S = 0.25
NUM_SAMPLES = int(SAMPLE_RATE * CLIP_DURATION_S)  # 4000
N_FFT = 256
HOP_LENGTH = 128
# With torch.stft(center=True) on 4000 samples: floor(4000/128) + 1 = 32 frames.
# This matches what PhaseCoder.STFTPatchExtractor produces with its default config.
NUM_FRAMES = 32

NUM_AZIMUTH = 38       # 38 bins (~9.5°/bin) + 1 no-speech = 39 classes
NUM_ELEVATION = 18     # 18 bins (10°/bin) + 1 no-speech = 19 classes
NUM_DISTANCE = 13      # 13 distance bins + 1 no-speech = 14 classes

# Distance bin edges in meters (paper uses non-uniform bins favoring near-field)
DISTANCE_BIN_EDGES = np.array([0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0, 2.5, 3.0, 3.5, 4.5, 5.5, 7.0])


# ---------------------------------------------------------------------------
# Source signal: harmonic-rich tone bursts (faster than downloading speech)
# ---------------------------------------------------------------------------

def generate_source_signal(duration_s: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a synthetic harmonic source with broadband content suitable for STFT.

    Real PhaseCoder uses speech; for a toy demo, multi-harmonic signals work fine
    and are deterministically generatable without dataset downloads.
    """
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n) / SAMPLE_RATE

    f0 = rng.uniform(120, 350)  # speech-like fundamental
    signal = np.zeros(n)
    for harmonic in range(1, 7):
        amp = rng.uniform(0.3, 1.0) / harmonic
        phase = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * f0 * harmonic * t + phase)

    # Amplitude modulation for speech-like envelope
    am = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 6) * t)
    signal *= am

    # Soft fades to prevent clicks
    fade = int(0.015 * SAMPLE_RATE)
    envelope = np.ones(n)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    signal *= envelope

    signal /= max(np.max(np.abs(signal)), 1e-8)
    return (signal * 0.8).astype(np.float32)


# ---------------------------------------------------------------------------
# Microphone array geometry
# ---------------------------------------------------------------------------

def generate_random_array(rng: np.random.Generator) -> np.ndarray:
    """Generate a random microphone array geometry. Returns (C, 3) in meters,
    centered at origin. Geometry-agnostic: random C in [4, 8], random radius."""
    num_mics = rng.integers(4, 9)
    radius = rng.uniform(0.04, 0.10)

    # Random circular array tilted at a random angle
    angles = np.linspace(0, 2 * np.pi, num_mics, endpoint=False)
    # Add small random perturbation per mic
    angles += rng.uniform(-0.1, 0.1, size=num_mics)

    coords = np.stack([
        radius * np.cos(angles),
        radius * np.sin(angles),
        rng.uniform(-0.01, 0.01, size=num_mics),  # small z-jitter
    ], axis=-1)
    return coords.astype(np.float32)


# ---------------------------------------------------------------------------
# Coordinate / quaternion helpers
# ---------------------------------------------------------------------------

def spherical_to_cartesian(azimuth_deg: float, elevation_deg: float, distance: float) -> np.ndarray:
    """Convention: azimuth 0°=+x, 90°=+y; elevation 0°=horizontal, +90°=+z."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    return np.array([
        distance * math.cos(el) * math.cos(az),
        distance * math.cos(el) * math.sin(az),
        distance * math.sin(el),
    ], dtype=np.float32)


def cartesian_to_spherical(xyz: np.ndarray) -> tuple:
    x, y, z = xyz
    d = math.sqrt(x * x + y * y + z * z)
    az = math.degrees(math.atan2(y, x)) % 360
    el = math.degrees(math.asin(z / max(d, 1e-8)))
    return az, el, d


def discretize_labels(az_deg: float, el_deg: float, dist: float) -> dict:
    az_bin = min(int(az_deg / 360 * NUM_AZIMUTH), NUM_AZIMUTH - 1)
    # elevation in [-90, 90] → bin index in [0, NUM_ELEVATION-1]
    el_bin = min(max(int((el_deg + 90) / 180 * NUM_ELEVATION), 0), NUM_ELEVATION - 1)
    # distance via bin edges
    dist_bin = int(np.searchsorted(DISTANCE_BIN_EDGES, dist))
    dist_bin = min(dist_bin, NUM_DISTANCE - 1)
    return {"azimuth": az_bin, "elevation": el_bin, "distance": dist_bin}


def axis_angle_to_quaternion(axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Convert axis-angle to (w, x, y, z) quaternion."""
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    half = angle_rad / 2
    s = math.sin(half)
    return np.array([math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=np.float32)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / (np.linalg.norm(q) + 1e-12)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float32)


# ---------------------------------------------------------------------------
# Single clip generation
# ---------------------------------------------------------------------------

def simulate_clip(
    mic_coords: np.ndarray,
    source_az_deg: float,
    source_el_deg: float,
    source_distance: float,
    rng: np.random.Generator,
    rotation_rate_dps: float = 0.0,
    rotation_axis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Simulate multichannel audio for one clip.

    Args:
        mic_coords:        (C, 3) neutral-frame mic positions in meters.
        source_az_deg:     source azimuth at canonical frame center.
        source_el_deg:     source elevation at canonical frame center.
        source_distance:   source distance from array centroid (m).
        rng:               numpy random generator.
        rotation_rate_dps: array rotation rate in deg/sec. 0 → static.
        rotation_axis:     (3,) unit axis. Default: random tilted axis.

    Returns:
        audio: (C, T_samples) float32.
        imu_quats: (NUM_FRAMES, 4) float32 if dynamic, else None.

    Implementation note:
        For dynamic clips, we approximate motion by using a single set of
        rotated mic positions corresponding to the canonical frame's orientation.
        The IMU quaternions accurately reflect the full rotation trajectory.
        This gives the model a clean training signal: "here's audio approximately
        consistent with this orientation trajectory, learn to predict source
        location at the canonical frame."
    """
    C = mic_coords.shape[0]

    # Build a small random shoebox room
    room_dim = np.array([
        rng.uniform(4.0, 8.0),
        rng.uniform(3.0, 6.0),
        rng.uniform(2.5, 3.5),
    ])
    rt60 = rng.uniform(0.15, 0.35)  # short reverb to keep simulation fast
    e_absorption, max_order = pra.inverse_sabine(rt60, room_dim)

    room = pra.ShoeBox(
        room_dim,
        fs=SAMPLE_RATE,
        materials=pra.Material(e_absorption),
        max_order=max_order,
    )

    # Place array centroid randomly inside room with margins
    centroid = np.array([
        rng.uniform(1.0, room_dim[0] - 1.0),
        rng.uniform(1.0, room_dim[1] - 1.0),
        rng.uniform(1.2, 1.8),  # head height
    ])

    # Determine canonical orientation for dynamic clips
    if rotation_rate_dps > 0:
        if rotation_axis is None:
            rotation_axis = np.array([0, 0, 1], dtype=np.float32)  # yaw by default
            rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

        # Compute IMU quaternions: rotation linearly increases from 0 to total_angle.
        total_angle_deg = rotation_rate_dps * CLIP_DURATION_S
        frame_times = (np.arange(NUM_FRAMES) * HOP_LENGTH) / SAMPLE_RATE  # frame START times
        # Center each frame at its midpoint
        frame_center_times = frame_times + (N_FFT / 2) / SAMPLE_RATE
        angles_rad = np.radians(rotation_rate_dps * frame_center_times)

        imu_quats = np.stack([
            axis_angle_to_quaternion(rotation_axis, ang) for ang in angles_rad
        ], axis=0)

        # Use the canonical frame's orientation to rotate mics for simulation
        canonical_idx = NUM_FRAMES // 2
        R_canonical = quaternion_to_rotation_matrix(imu_quats[canonical_idx])
        mic_coords_rotated = mic_coords @ R_canonical.T
    else:
        imu_quats = None
        mic_coords_rotated = mic_coords

    # Place mics in world coordinates
    mic_world = (centroid[None, :] + mic_coords_rotated).T  # (3, C)
    room.add_microphone_array(pra.MicrophoneArray(mic_world, fs=SAMPLE_RATE))

    # Place source
    source_offset = spherical_to_cartesian(source_az_deg, source_el_deg, source_distance)
    source_world = centroid + source_offset

    # Clamp source to within room
    source_world = np.clip(source_world, 0.2, room_dim - 0.2)

    # Generate source signal slightly longer than clip duration
    src_signal = generate_source_signal(CLIP_DURATION_S * 2.0, rng)
    room.add_source(source_world, signal=src_signal)

    # Simulate
    room.simulate()

    # Truncate / pad to NUM_SAMPLES
    sim_audio = room.mic_array.signals  # (C, T_sim)
    T_sim = sim_audio.shape[1]
    if T_sim >= NUM_SAMPLES:
        # Take a window starting after the direct-path delay has settled
        start = max(0, int(0.005 * SAMPLE_RATE))  # 5ms in to skip silence
        if start + NUM_SAMPLES > T_sim:
            start = T_sim - NUM_SAMPLES
        audio = sim_audio[:, start:start + NUM_SAMPLES]
    else:
        audio = np.zeros((C, NUM_SAMPLES), dtype=np.float32)
        audio[:, :T_sim] = sim_audio

    # Normalize
    peak = np.max(np.abs(audio))
    if peak > 1e-8:
        audio = audio / peak * 0.7
    audio = audio.astype(np.float32)

    return audio, imu_quats


# ---------------------------------------------------------------------------
# Dataset generation driver
# ---------------------------------------------------------------------------

def generate_dataset(
    output_dir: Path,
    num_clips: int,
    fraction_dynamic: float,
    seed: int = 0,
    val_fraction: float = 0.1,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    manifest = {
        "num_clips": num_clips,
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
        "clips": [],
    }

    print(f"Generating {num_clips} toy clips → {output_dir}")
    print(f"  Static clips: {int(num_clips * (1 - fraction_dynamic))}")
    print(f"  Dynamic clips: {int(num_clips * fraction_dynamic)}")

    for i in range(num_clips):
        # Pick a random mic geometry per clip (geometry-agnostic training)
        mic_coords = generate_random_array(rng)

        # Pick source location
        source_az = rng.uniform(0, 360)
        source_el = rng.uniform(-30, 30)  # bias toward horizontal (realistic)
        source_dist = float(np.exp(rng.uniform(np.log(0.5), np.log(5.0))))  # log-uniform

        # Decide static vs dynamic
        is_dynamic = rng.random() < fraction_dynamic
        if is_dynamic:
            # Random rotation rate; weighted toward moderate rates that would
            # actually appear in wearable use.
            rotation_rate = float(rng.choice([
                30, 60, 100, 150, 200, 300, 450,
            ]) * rng.choice([-1, 1]))
            # Random tilted axis (mostly yaw, some roll/pitch)
            axis = np.array([
                rng.uniform(-0.3, 0.3),
                rng.uniform(-0.3, 0.3),
                rng.choice([-1, 1]) * rng.uniform(0.7, 1.0),
            ], dtype=np.float32)
            axis = axis / np.linalg.norm(axis)
        else:
            rotation_rate = 0.0
            axis = None

        # Simulate
        try:
            audio, imu_quats = simulate_clip(
                mic_coords=mic_coords,
                source_az_deg=source_az,
                source_el_deg=source_el,
                source_distance=source_dist,
                rng=rng,
                rotation_rate_dps=abs(rotation_rate) if is_dynamic else 0.0,
                rotation_axis=axis,
            )
        except Exception as e:
            print(f"  clip {i}: simulation failed ({e}), skipping")
            continue

        # Discretize labels
        labels = discretize_labels(source_az, source_el, source_dist)

        # Save
        clip_path = output_dir / f"clip_{i:05d}.npz"
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

        manifest["clips"].append({
            "filename": clip_path.name,
            "is_dynamic": bool(is_dynamic),
            "rotation_rate_dps": float(rotation_rate),
            "azimuth_deg": float(source_az),
            "elevation_deg": float(source_el),
            "distance_m": float(source_dist),
            "num_mics": int(mic_coords.shape[0]),
        })

        if (i + 1) % 50 == 0:
            print(f"  generated {i + 1}/{num_clips}")

    # Train/val split (deterministic by index)
    n_val = int(len(manifest["clips"]) * val_fraction)
    val_indices = set(rng.choice(len(manifest["clips"]), size=n_val, replace=False).tolist())
    manifest["train_filenames"] = [
        c["filename"] for i, c in enumerate(manifest["clips"]) if i not in val_indices
    ]
    manifest["val_filenames"] = [
        c["filename"] for i, c in enumerate(manifest["clips"]) if i in val_indices
    ]

    with open(output_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Saved {len(manifest['clips'])} clips to {output_dir}")
    print(f"  Train: {len(manifest['train_filenames'])}")
    print(f"  Val:   {len(manifest['val_filenames'])}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate toy PhaseCoder dataset")
    parser.add_argument("--num_clips", type=int, default=500,
                        help="Total clips to generate (default: 500)")
    parser.add_argument("--output_dir", type=str, default="./toy_data",
                        help="Output directory")
    parser.add_argument("--fraction_dynamic", type=float, default=0.5,
                        help="Fraction of clips with rotating array (default: 0.5)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    args = parser.parse_args()

    generate_dataset(
        output_dir=Path(args.output_dir),
        num_clips=args.num_clips,
        fraction_dynamic=args.fraction_dynamic,
        seed=args.seed,
        val_fraction=args.val_fraction,
    )


if __name__ == "__main__":
    main()
