# PhaseCoder

PyTorch implementation of **PhaseCoder**, a microphone geometry–agnostic spatial audio encoder for multimodal LLMs, as described in *PhaseCoder: Microphone Geometry-Agnostic Spatial Audio Understanding for Multimodal LLMs* (Dementyev et al., 2026).

This repository is a minimal reference: the model, loss, and a small forward/loss sanity check live in [`PhaseCoder.py`](PhaseCoder.py).

## Requirements

- Python 3.x
- [PyTorch](https://pytorch.org/) (with CUDA optional; the script falls back to CPU)

Install PyTorch following the instructions for your platform, for example:

```bash
pip install torch
```

## Model overview

| Component | Role |
|-----------|------|
| `STFTPatchExtractor` | 16 kHz multichannel audio → STFT magnitude + phase patches per mic/frame (default 33 frames × 258 features). |
| `MicrophonePositionalEmbedding` | Cartesian mic positions → embedding (GI-DOAEnet-style spherical fusion). |
| `PhaseCoder` | Linear patch projection, summed sequential + frame + mic positional embeddings, learnable `[CLS]`, 5-layer ViT-style `TransformerEncoder`, spatial MLP, and three classification heads (azimuth, elevation, distance). |
| `PhaseCoderLoss` | Weighted sum of cross-entropy losses over the three heads (default λ_dist = 0.5). |

Rough parameter count: ~6M (see `__main__` block for a live count).

## Inputs and outputs

**Forward** (`PhaseCoder.forward`):

- `audio`: `(B, C, T)` — raw multichannel waveform at **16 kHz**. Example: **250 ms** → `T = 4000`.
- `mic_coords`: `(B, C, 3)` — Cartesian `(x, y, z)` per microphone in **meters**, in the *device-neutral* (manufacturer) frame. These are the fixed, physical mic positions on the device; the model rotates them per frame internally when IMU data is provided.
- `imu_orientations` *(optional)*: `(B, F, 4)` — unit quaternions in `(w, x, y, z)` order, one per STFT frame (`F = 33` for the default 250 ms / 16 kHz / hop-128 configuration). Pass `None` (default) to reproduce the original static-geometry behaviour.

#### IMU preprocessing (caller's responsibility)

Raw IMU data typically arrives at 100–1000 Hz. Before calling the model, **SLERP-interpolate** the quaternion stream to produce exactly `F` uniformly-spaced samples aligned to the STFT frame centres. This decouples IMU hardware from model architecture — the model never sees sample rates.

#### Reference frame convention (Option A)

When `imu_orientations` is provided, the model expresses all per-frame mic positions *relative* to a single canonical STFT frame (default: `F // 2 = 16`, the midpoint of the clip). Concretely:

```
R_rel_f = R_ref^T @ R_f
p_i,f   = R_rel_f · p_i,neutral
```

At `f = ref`, `R_rel = I` and `p_i,ref = p_i,neutral`. The CLS-token output is therefore unambiguously in the **device-instantaneous frame at the canonical frame index** — a single, well-defined coordinate system per clip. To change the reference frame, pass `canonical_frame_idx=<idx>` to the `PhaseCoder` constructor.

If a downstream application needs *world-relative* azimuth/elevation, compose PhaseCoder's device-frame prediction with the absolute IMU orientation at the canonical frame (available from the same SLERP output).

**Returns** a `dict`:

- `spatial_embedding`: `(B, D)` with `D = 256` by default — soft token for downstream LLM or other modules.
- `azimuth_logits`: `(B, 39)` — 38 azimuth bins + no-speech (paper-style discretization).
- `elevation_logits`: `(B, 19)` — 18 elevation bins + no-speech.
- `distance_logits`: `(B, 14)` — 13 distance bins + no-speech.

**Loss** (`PhaseCoderLoss`): pass the model output dict and a `targets` dict with integer labels `(B,)` for keys `azimuth`, `elevation`, and `distance`.

## Run the built-in sanity check

From this directory:

```bash
python PhaseCoder.py
```

This instantiates the model, runs a random batch, prints tensor shapes and parameter count, and computes a sample total loss.

## Citation

If you use this architecture in published work, cite the PhaseCoder paper (Dementyev et al., 2026) as given in the docstring of [`PhaseCoder.py`](PhaseCoder.py).
