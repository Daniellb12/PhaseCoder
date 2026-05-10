"""
LibriSpeech Source Provider for PhaseCoder Toy Datasets

Replaces the synthetic harmonic source signal with real human speech from
LibriSpeech (https://www.openslr.org/12). Designed to integrate with both
`generate_toy_dataset_parallel.py` and `generate_time_varying_rir.py` without
changing their architectures — just swap the source generator function.

Why LibriSpeech:
    - PhaseCoder paper used LibriSpeech for source signals
    - Clean read speech with diverse speakers and content (1000+ speakers)
    - 16 kHz native sample rate matches PhaseCoder's expected input
    - Permissive CC-BY 4.0 license, scientifically standard
    - Direct download from OpenSLR (no HuggingFace API, no authentication)

Recommended subset: "dev-clean" (~337 MB, 5.4 hours, 40 speakers)
    - Small enough to download in 5 minutes on most connections
    - Diverse enough for 25k-100k clip generation since each utterance gets
      reused across clips with different random crop windows
    - "clean" means low background noise, ideal for spatial audio synthesis

Usage:

    # One-time download (reuses cache on subsequent runs)
    python librispeech_source.py --download

    # In your generator code:
    from librispeech_source import LibriSpeechProvider

    provider = LibriSpeechProvider(
        cache_dir="./librispeech_cache",
        subset="dev-clean",
        synthetic_fraction=0.1,  # 10% synthetic, 90% real speech
    )

    # In your clip generation loop:
    src_signal = provider.get_signal(duration_s=0.5, rng=rng)
"""

import argparse
import hashlib
import os
import random
import sys
import tarfile
import time
import urllib.request
from pathlib import Path
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000  # PhaseCoder standard, matches LibriSpeech native rate

# OpenSLR download URLs and approximate sizes
LIBRISPEECH_URLS = {
    "dev-clean":   ("https://www.openslr.org/resources/12/dev-clean.tar.gz",   337 * 1024 * 1024),
    "dev-other":   ("https://www.openslr.org/resources/12/dev-other.tar.gz",   314 * 1024 * 1024),
    "test-clean":  ("https://www.openslr.org/resources/12/test-clean.tar.gz",  346 * 1024 * 1024),
    "test-other":  ("https://www.openslr.org/resources/12/test-other.tar.gz",  328 * 1024 * 1024),
    "train-clean-100": ("https://www.openslr.org/resources/12/train-clean-100.tar.gz", 6300 * 1024 * 1024),
    "train-clean-360": ("https://www.openslr.org/resources/12/train-clean-360.tar.gz", 23000 * 1024 * 1024),
    "train-other-500": ("https://www.openslr.org/resources/12/train-other-500.tar.gz", 30000 * 1024 * 1024),
}


# ---------------------------------------------------------------------------
# Synthetic fallback (same signal as your existing toy generator)
# ---------------------------------------------------------------------------

def generate_synthetic_signal(duration_s: float, rng: np.random.Generator) -> np.ndarray:
    """Generate a synthetic harmonic source signal at SAMPLE_RATE.

    Same as your existing toy generator. Kept here for fallback and for
    mixing real + synthetic clips when synthetic_fraction > 0.
    """
    n = int(SAMPLE_RATE * duration_s)
    t = np.arange(n) / SAMPLE_RATE

    f0 = rng.uniform(120, 350)
    signal = np.zeros(n)
    for harmonic in range(1, 7):
        amp = rng.uniform(0.3, 1.0) / harmonic
        phase = rng.uniform(0, 2 * np.pi)
        signal += amp * np.sin(2 * np.pi * f0 * harmonic * t + phase)

    am = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(2, 6) * t)
    signal *= am

    fade = int(0.015 * SAMPLE_RATE)
    envelope = np.ones(n)
    envelope[:fade] = np.linspace(0, 1, fade)
    envelope[-fade:] = np.linspace(1, 0, fade)
    signal *= envelope

    signal /= max(np.max(np.abs(signal)), 1e-8)
    return (signal * 0.8).astype(np.float32)


# ---------------------------------------------------------------------------
# Downloader with progress
# ---------------------------------------------------------------------------

def _download_with_progress(url: str, dest_path: Path, expected_size: Optional[int] = None):
    """Download a file with a progress bar."""
    if dest_path.exists():
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"  Already downloaded: {dest_path} ({size_mb:.0f} MB)")
        return

    print(f"  Downloading {url}")
    print(f"  → {dest_path}")
    if expected_size:
        print(f"  Expected size: ~{expected_size / 1024 / 1024:.0f} MB")

    start_time = time.time()
    last_print = start_time

    def reporthook(blocks_done, block_size, total_size):
        nonlocal last_print
        downloaded = blocks_done * block_size
        now = time.time()
        if now - last_print < 0.5 and downloaded < total_size:
            return
        last_print = now

        elapsed = now - start_time
        speed_mbps = (downloaded / elapsed) / (1024 * 1024) if elapsed > 0 else 0

        if total_size > 0:
            pct = 100 * downloaded / total_size
            mb_done = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            eta_sec = (total_size - downloaded) / max(downloaded / elapsed, 1) if elapsed > 0 else 0
            sys.stdout.write(
                f"\r  [{pct:5.1f}%] {mb_done:.0f}/{mb_total:.0f} MB | "
                f"{speed_mbps:.1f} MB/s | ETA {eta_sec/60:.1f} min"
            )
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, dest_path, reporthook=reporthook)
        sys.stdout.write("\n")
        elapsed = time.time() - start_time
        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ Downloaded {size_mb:.0f} MB in {elapsed/60:.1f} min")
    except Exception as e:
        if dest_path.exists():
            dest_path.unlink()  # remove partial file
        raise RuntimeError(f"Download failed: {e}")


def _extract_tarball(tar_path: Path, extract_to: Path):
    """Extract a .tar.gz file with progress reporting."""
    if extract_to.exists() and any(extract_to.iterdir()):
        print(f"  Already extracted to {extract_to}")
        return

    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"  Extracting {tar_path.name}...")
    start_time = time.time()

    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()
        total = len(members)
        for i, member in enumerate(members):
            tar.extract(member, extract_to)
            if (i + 1) % 1000 == 0 or (i + 1) == total:
                pct = 100 * (i + 1) / total
                sys.stdout.write(f"\r  [{pct:5.1f}%] {i+1}/{total} files")
                sys.stdout.flush()

    sys.stdout.write("\n")
    print(f"  ✓ Extracted in {(time.time()-start_time)/60:.1f} min")


# ---------------------------------------------------------------------------
# Provider class
# ---------------------------------------------------------------------------

class LibriSpeechProvider:
    """Provides real speech source signals for spatial audio simulation.

    First use:
        - Downloads the requested LibriSpeech subset from OpenSLR
        - Extracts the tarball
        - Indexes all .flac files for fast random access

    Subsequent uses:
        - Loads the cached file index
        - Reads .flac files on demand during generation

    Each call to `get_signal(duration_s, rng)` returns a randomly-cropped
    audio segment from a randomly-chosen utterance, normalized to peak 0.8.

    Args:
        cache_dir:           Local directory for downloaded data.
        subset:              Which LibriSpeech subset to use. Default
                             "dev-clean" (337 MB, 5.4 hours of speech).
                             For more variety: "train-clean-100" (6.3 GB).
        max_utterances:      Cap how many utterance files to index. None = all.
                             Capping at 1000-2000 is plenty for 25k clips.
        synthetic_fraction:  Fraction of get_signal() calls that should return
                             synthetic harmonics instead of real speech.
                             0.0 = all real, 1.0 = all synthetic.
        min_duration_s:      Skip utterances shorter than this (seconds).
                             Default 1.5s, ensures we can crop a 0.5s segment
                             with margin on either side.
        verbose:             Print progress messages.

    Notes:
        - LibriSpeech audio is mono, 16 kHz, 16-bit FLAC. Native rate matches
          our SAMPLE_RATE so no resampling is needed.
        - The first instantiation downloads and indexes; subsequent calls
          are fast (~1-2 ms per signal request).
        - Thread-safe for read access; safe to use across multiprocessing
          workers as long as the cache exists before workers start.
    """

    def __init__(
        self,
        cache_dir: str | Path = "./librispeech_cache",
        subset: str = "dev-clean",
        max_utterances: Optional[int] = 2000,
        synthetic_fraction: float = 0.0,
        min_duration_s: float = 1.5,
        verbose: bool = True,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.subset = subset
        self.max_utterances = max_utterances
        self.synthetic_fraction = synthetic_fraction
        self.min_duration_s = min_duration_s
        self.verbose = verbose

        if subset not in LIBRISPEECH_URLS:
            raise ValueError(
                f"Unknown subset '{subset}'. Choices: {list(LIBRISPEECH_URLS.keys())}"
            )

        self.flac_files: list[Path] = []
        self.using_synthetic_only = False

        try:
            self._ensure_cached()
            self._build_file_index()
            if not self.flac_files:
                raise RuntimeError("No FLAC files found in cache")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  LibriSpeech setup failed: {e}")
                print(f"  Falling back to synthetic-only mode.")
            self.using_synthetic_only = True

        if verbose and not self.using_synthetic_only:
            print(f"  ✓ Indexed {len(self.flac_files)} utterances "
                  f"from {self.subset}")
            print(f"  Synthetic fraction: {self.synthetic_fraction:.0%}")

    def _ensure_cached(self):
        """Download and extract the LibriSpeech subset if not present."""
        url, expected_size = LIBRISPEECH_URLS[self.subset]
        tar_path = self.cache_dir / f"{self.subset}.tar.gz"
        extract_path = self.cache_dir / "LibriSpeech" / self.subset

        if not extract_path.exists() or not any(extract_path.rglob("*.flac")):
            _download_with_progress(url, tar_path, expected_size)
            _extract_tarball(tar_path, self.cache_dir)

    def _build_file_index(self):
        """Find all FLAC files in the cached subset."""
        index_file = self.cache_dir / f"index_{self.subset}.npy"

        if index_file.exists():
            try:
                cached_paths = np.load(index_file, allow_pickle=True)
                self.flac_files = [Path(str(p)) for p in cached_paths]
                self.flac_files = [f for f in self.flac_files if f.exists()]
                if self.flac_files:
                    if self.max_utterances:
                        # Deterministic subsample for reproducibility
                        rng = np.random.default_rng(0)
                        if len(self.flac_files) > self.max_utterances:
                            indices = rng.choice(
                                len(self.flac_files),
                                size=self.max_utterances,
                                replace=False,
                            )
                            self.flac_files = [self.flac_files[i] for i in sorted(indices)]
                    return
            except Exception:
                pass  # rebuild index on any failure

        # Build fresh index
        extract_path = self.cache_dir / "LibriSpeech" / self.subset
        if self.verbose:
            print(f"  Indexing FLAC files in {extract_path}...")

        all_flacs = sorted(extract_path.rglob("*.flac"))
        if self.verbose:
            print(f"  Found {len(all_flacs)} FLAC files")

        if self.max_utterances and len(all_flacs) > self.max_utterances:
            rng = np.random.default_rng(0)
            indices = rng.choice(len(all_flacs), size=self.max_utterances, replace=False)
            all_flacs = [all_flacs[i] for i in sorted(indices)]

        self.flac_files = all_flacs
        np.save(index_file, np.array([str(p) for p in self.flac_files]))

    def get_signal(self, duration_s: float, rng: np.random.Generator) -> np.ndarray:
        """Get a source signal of the specified duration.

        With probability `synthetic_fraction`, returns a synthetic harmonic.
        Otherwise returns a randomly-cropped segment of real speech.

        Args:
            duration_s: target duration in seconds.
            rng:        numpy random generator for deterministic sampling.

        Returns:
            (n_samples,) float32 audio at SAMPLE_RATE, peak-normalized to 0.8.
        """
        if self.using_synthetic_only or rng.random() < self.synthetic_fraction:
            return generate_synthetic_signal(duration_s, rng)

        # Pick a random utterance and load it
        flac_path = rng.choice(self.flac_files)

        try:
            import soundfile as sf
            audio, sr = sf.read(str(flac_path), dtype="float32")
        except Exception as e:
            # File-level failure: fall back to synthetic for this clip
            return generate_synthetic_signal(duration_s, rng)

        # LibriSpeech is 16 kHz mono, but check defensively
        if sr != SAMPLE_RATE:
            # Should never happen with LibriSpeech, but handle gracefully
            return generate_synthetic_signal(duration_s, rng)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # mono-fold if somehow stereo

        target_samples = int(duration_s * SAMPLE_RATE)

        # Skip utterances shorter than our minimum duration
        if len(audio) < int(self.min_duration_s * SAMPLE_RATE):
            return generate_synthetic_signal(duration_s, rng)
       
        # Random crop window with voice activity check
        # Keep trying random crops until we find one with enough signal energy.
        # This prevents training on near-silent slices (gaps between words, breaths).
        MIN_RMS = 0.02  # minimum RMS energy to consider "voiced"
        MAX_RETRIES = 20

        if len(audio) >= target_samples:
            max_start = len(audio) - target_samples
            cropped = None
            
            for _ in range(MAX_RETRIES):
                if max_start > 0:
                    start = int(rng.integers(0, max_start + 1))
                else:
                    start = 0
                candidate = audio[start : start + target_samples]
                
                # Check if this crop has enough voice content
                candidate_rms = np.sqrt(np.mean(candidate ** 2))
                if candidate_rms >= MIN_RMS:
                    cropped = candidate
                    break
            
            # If we never found a good crop, take the loudest segment we can find
            if cropped is None:
                # Slide a window across the audio to find the loudest target_samples region
                best_start = 0
                best_rms = 0
                # Step in 10ms increments for efficiency
                step = int(0.01 * SAMPLE_RATE)
                for start in range(0, max_start + 1, step):
                    window = audio[start : start + target_samples]
                    window_rms = np.sqrt(np.mean(window ** 2))
                    if window_rms > best_rms:
                        best_rms = window_rms
                        best_start = start
                cropped = audio[best_start : best_start + target_samples]
        else:
            # Utterance shorter than target: pad with zeros
            cropped = np.zeros(target_samples, dtype=np.float32)
            cropped[: len(audio)] = audio

        cropped = cropped.astype(np.float32)

        # Apply soft fade in/out to prevent clicks at boundaries
        fade = int(0.015 * SAMPLE_RATE)
        if len(cropped) > 2 * fade:
            envelope = np.ones(len(cropped), dtype=np.float32)
            envelope[:fade] = np.linspace(0, 1, fade)
            envelope[-fade:] = np.linspace(1, 0, fade)
            cropped = cropped * envelope

        # Peak-normalize to 0.8
        peak = np.max(np.abs(cropped))
        if peak > 1e-8:
            cropped = cropped / peak * 0.8

        # Skip clips that are essentially silent (e.g. all-zero pad regions)
        if np.max(np.abs(cropped)) < 0.05:
            return generate_synthetic_signal(duration_s, rng)

        return cropped


# ---------------------------------------------------------------------------
# CLI for pre-caching the dataset
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Download and cache LibriSpeech")
    parser.add_argument("--cache_dir", type=str, default="./librispeech_cache")
    parser.add_argument("--subset", type=str, default="dev-clean",
                        choices=list(LIBRISPEECH_URLS.keys()))
    parser.add_argument("--max_utterances", type=int, default=2000,
                        help="Index at most N utterances (default 2000)")
    parser.add_argument("--test", action="store_true",
                        help="After caching, generate a test signal to verify")
    args = parser.parse_args()

    print("=" * 60)
    print("LibriSpeech Source Provider Setup")
    print("=" * 60)

    provider = LibriSpeechProvider(
        cache_dir=args.cache_dir,
        subset=args.subset,
        max_utterances=args.max_utterances,
        synthetic_fraction=0.0,
        verbose=True,
    )

    if provider.using_synthetic_only:
        print("\n⚠️  Setup did not complete successfully.")
        print("    Provider is in synthetic-only mode.")
        sys.exit(1)

    print(f"\n✓ Setup complete.")
    print(f"  Subset: {args.subset}")
    print(f"  Indexed utterances: {len(provider.flac_files)}")
    print(f"  Cache size: {sum(f.stat().st_size for f in provider.flac_files) / (1024**3):.2f} GB")

    if args.test:
        print(f"\nGenerating 5 test signals...")
        rng = np.random.default_rng(42)
        for i in range(5):
            sig = provider.get_signal(duration_s=0.5, rng=rng)
            print(f"  Signal {i+1}: shape={sig.shape}, "
                  f"peak={np.max(np.abs(sig)):.3f}, "
                  f"rms={np.sqrt(np.mean(sig**2)):.3f}")
        print("✓ Test signals generated successfully.")


if __name__ == "__main__":
    main()
