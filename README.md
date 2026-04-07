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
- `mic_coords`: `(B, C, 3)` — Cartesian `(x, y, z)` per microphone in **meters** (relative geometry is used via centroid-centered spherical features).

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
