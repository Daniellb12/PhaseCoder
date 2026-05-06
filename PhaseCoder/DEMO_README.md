# PhaseCoder Toy Demo

A small synthetic dataset and training pipeline for demonstrating PhaseCoder
with the IMU-conditioned dynamic-array extension. Designed to run end-to-end
on a laptop in **~30 minutes** with a GPU, or **~2 hours on CPU**.

This is a **proof-of-concept**, not a benchmark result. It shows that:
1. The PhaseCoder architecture trains and converges on synthetic data.
2. The IMU-conditioning extension functions correctly end-to-end.
3. On dynamic-array clips, the IMU-conditioned model can outperform a baseline
   that ignores array motion.

## Files

- `PhaseCoder.py` — model and loss (your existing implementation).
- `imu_preprocessing.py` — IMU → frame-quaternion utility.
- `generate_toy_dataset.py` — synthesizes a toy dataset using pyroomacoustics.
- `train_toy.py` — trains baseline and IMU-conditioned variants.
- `evaluate_toy.py` — produces the headline comparison plot.

## Setup

```bash
pip install torch numpy pyroomacoustics matplotlib
```

That's it. No CUDA required (works on CPU, just slower).

## Quick run (the full demo, ~30 min on GPU / ~2 hours on CPU)

```bash
# 1. Generate 600 synthetic clips (~3 minutes)
python generate_toy_dataset.py \
    --num_clips 600 \
    --output_dir ./toy_data \
    --fraction_dynamic 0.6

# 2a. Train baseline (no IMU)  ~10 min on GPU
python train_toy.py \
    --data_dir ./toy_data \
    --output_dir ./runs/baseline \
    --use_imu false \
    --epochs 30 \
    --batch_size 16

# 2b. Train IMU-conditioned model  ~10 min on GPU
python train_toy.py \
    --data_dir ./toy_data \
    --output_dir ./runs/imu \
    --use_imu true \
    --epochs 30 \
    --batch_size 16

# 3. Evaluate and produce comparison plot
python evaluate_toy.py \
    --data_dir ./toy_data \
    --baseline_ckpt ./runs/baseline/best.pt \
    --imu_ckpt ./runs/imu/best.pt \
    --output_dir ./eval_results
```

The headline output is `eval_results/comparison.png`: a bar chart of azimuth
classification accuracy bucketed by array rotation rate, comparing the
baseline against the IMU-conditioned model.

## How it follows the PhaseCoder paper

The methodology mirrors the paper exactly, just at a tiny scale:

| Paper                                   | This demo                          |
|-----------------------------------------|------------------------------------|
| 4M synthetic clips                      | 500-2000 clips                     |
| Speech sources (LibriSpeech)            | Synthetic harmonic signals         |
| Synthetic RIRs (custom)                 | pyroomacoustics image-source       |
| Hundreds of random mic geometries       | Random circular arrays per clip    |
| 250ms clips, 16kHz                      | Same                               |
| STFT (256 win, 128 hop)                 | Same                               |
| 6M-param transformer                    | Same architecture (~2.2M with default config) |
| Az/el/dist classification heads         | Same                               |

The *training methodology* is the same; only the *scale and source variety* are reduced.

## Pitching this in a presentation

Key talking points that frame this honestly and well:

1. **"This is a proof-of-concept on synthetic data, not a benchmark result."**
   Reviewers respect this framing.
2. **"The full PhaseCoder methodology is preserved."** RIR-based synthesis,
   geometry-agnostic embeddings, magnitude+phase STFT features.
3. **"This validates the IMU-conditioning extension end-to-end."** The
   architecture supports per-frame mic position updates from quaternions,
   training converges, and there's a measurable contribution on dynamic clips.
4. **"Scale is the next step."** Once we have proof-of-concept, we scale to
   LOCATA + AEA + custom synthetic data with full diversity.

## Notes

- Frame count: with `torch.stft(center=True)` on 4000 samples, you get 32
  frames, not 33. Both the dataset generator and the model agree on this.
- CPU training: ~70 sec/epoch with 540 train clips, 32 frames, 8 mics.
  GPU should be ~10x faster.
- The baseline in evaluate_toy.py uses `use_imu=False`, meaning it ignores
  IMU data entirely (treating dynamic clips as if static). This is the
  honest comparison: both models see the same audio, but only one knows
  about array motion.

## Scaling up later

For real results worth publishing, you'd want:

- 50k-100k clips minimum (still tiny vs. paper's 4M, but enough for meaningful
  generalization tests)
- Real speech (LibriSpeech subset)
- More room diversity (varying RT60, dimensions, source types)
- Evaluation on LOCATA dynamic tasks (Tasks 4-6)

The pipeline here scales to those directly: just bump `--num_clips`, swap the
`generate_source_signal` function for LibriSpeech sampling, and you're set.
