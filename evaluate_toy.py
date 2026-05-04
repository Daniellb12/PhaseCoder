"""
Evaluate trained PhaseCoder models and produce comparison plot.

Compares baseline (no IMU) vs IMU-conditioned model performance, broken down by
rotation rate. The headline figure for the demo: a plot showing baseline accuracy
degrading as rotation rate increases, while the IMU model maintains performance.

Usage:
    python evaluate_toy.py --data_dir ./toy_data \\
        --baseline_ckpt ./runs/baseline/best.pt \\
        --imu_ckpt ./runs/imu/best.pt \\
        --output_dir ./eval_results
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt

from PhaseCoder import PhaseCoder
from train_toy import ToyDataset, collate_fn
from torch.utils.data import DataLoader


def load_model(ckpt_path: Path, device) -> PhaseCoder:
    model = PhaseCoder().to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def evaluate_model(model, loader, device, use_imu: bool):
    """Run model on loader. Returns per-clip records."""
    records = []
    with torch.no_grad():
        for batch in loader:
            audio = batch["audio"].to(device)
            mic_coords = batch["mic_coords"].to(device)
            imu = batch["imu_quats"].to(device) if (use_imu and batch["imu_quats"] is not None) else None
            targets = batch["labels"]

            out = model(audio, mic_coords, imu_orientations=imu)
            az_pred = out["azimuth_logits"].argmax(dim=-1).cpu()
            el_pred = out["elevation_logits"].argmax(dim=-1).cpu()
            dist_pred = out["distance_logits"].argmax(dim=-1).cpu()

            for i in range(audio.shape[0]):
                records.append({
                    "is_dynamic": batch["is_dynamic"][i],
                    "rotation_rate": abs(batch["rotation_rate_dps"][i]),
                    "az_correct": int(az_pred[i] == targets["azimuth"][i]),
                    "el_correct": int(el_pred[i] == targets["elevation"][i]),
                    "dist_correct": int(dist_pred[i] == targets["distance"][i]),
                    "az_pred": int(az_pred[i]),
                    "az_true": int(targets["azimuth"][i]),
                    "el_pred": int(el_pred[i]),
                    "el_true": int(targets["elevation"][i]),
                })
    return records


def bin_by_rotation_rate(records, bins=(0, 50, 150, 250, 400, 1000)):
    """Group records by rotation rate buckets."""
    buckets = {}
    for r in records:
        rate = r["rotation_rate"]
        for i in range(len(bins) - 1):
            if bins[i] <= rate < bins[i + 1]:
                key = f"{bins[i]}-{bins[i+1]}"
                buckets.setdefault(key, []).append(r)
                break
    return buckets


def compute_accuracy(records):
    if not records:
        return {"az": 0, "el": 0, "dist": 0, "n": 0}
    n = len(records)
    return {
        "az": sum(r["az_correct"] for r in records) / n,
        "el": sum(r["el_correct"] for r in records) / n,
        "dist": sum(r["dist_correct"] for r in records) / n,
        "n": n,
    }


def make_plot(baseline_buckets, imu_buckets, output_path: Path):
    """Headline plot: accuracy vs rotation rate, baseline vs IMU."""
    bucket_keys = sorted(set(baseline_buckets.keys()) | set(imu_buckets.keys()),
                         key=lambda s: int(s.split("-")[0]))

    baseline_az = [compute_accuracy(baseline_buckets.get(k, []))["az"] for k in bucket_keys]
    imu_az = [compute_accuracy(imu_buckets.get(k, []))["az"] for k in bucket_keys]
    counts = [compute_accuracy(baseline_buckets.get(k, []))["n"] for k in bucket_keys]

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(bucket_keys))
    width = 0.35

    ax.bar(x - width/2, baseline_az, width, label="Baseline (no IMU)", color="#d62728", alpha=0.85)
    ax.bar(x + width/2, imu_az, width, label="IMU-conditioned", color="#2ca02c", alpha=0.85)

    ax.set_xlabel("Rotation rate (deg/sec)", fontsize=12)
    ax.set_ylabel("Azimuth classification accuracy", fontsize=12)
    ax.set_title("PhaseCoder Performance vs. Array Rotation Rate\n(Toy Synthetic Dataset)",
                 fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n(n={n})" for k, n in zip(bucket_keys, counts)])
    ax.legend(fontsize=11, loc="upper right")
    ax.set_ylim(0, max(max(baseline_az + imu_az) * 1.15, 0.1))
    ax.grid(axis="y", alpha=0.3)

    # Annotate bars with values
    for i, (b, m) in enumerate(zip(baseline_az, imu_az)):
        ax.text(i - width/2, b + 0.01, f"{b:.0%}", ha="center", fontsize=9)
        ax.text(i + width/2, m + 0.01, f"{m:.0%}", ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved plot: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--baseline_ckpt", type=str, required=True)
    parser.add_argument("--imu_ckpt", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./eval_results")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    with open(data_dir / "manifest.json") as f:
        manifest = json.load(f)

    # Use full validation set; for richer results, also run on train (with note about it)
    eval_filenames = manifest["val_filenames"]
    print(f"Evaluating on {len(eval_filenames)} clips")

    # --- Baseline model: ignores IMU even on dynamic clips ---
    print("\n[1/2] Evaluating baseline model (no IMU)...")
    baseline_ds = ToyDataset(data_dir, eval_filenames, use_imu=False)
    baseline_loader = DataLoader(baseline_ds, batch_size=args.batch_size,
                                  collate_fn=collate_fn, num_workers=0)
    baseline_model = load_model(Path(args.baseline_ckpt), device)
    baseline_records = evaluate_model(baseline_model, baseline_loader, device, use_imu=False)

    # --- IMU model: uses IMU conditioning ---
    print("[2/2] Evaluating IMU-conditioned model...")
    imu_ds = ToyDataset(data_dir, eval_filenames, use_imu=True)
    imu_loader = DataLoader(imu_ds, batch_size=args.batch_size,
                             collate_fn=collate_fn, num_workers=0)
    imu_model = load_model(Path(args.imu_ckpt), device)
    imu_records = evaluate_model(imu_model, imu_loader, device, use_imu=True)

    # Bin and compare
    baseline_buckets = bin_by_rotation_rate(baseline_records)
    imu_buckets = bin_by_rotation_rate(imu_records)

    # --- Print summary ---
    print("\n" + "=" * 70)
    print("RESULTS BY ROTATION RATE")
    print("=" * 70)
    print(f"{'Bucket (deg/sec)':<20} {'n':<6} {'Baseline Az':<14} {'IMU Az':<14} {'Δ':<10}")
    print("-" * 70)
    for k in sorted(baseline_buckets.keys(), key=lambda s: int(s.split("-")[0])):
        b = compute_accuracy(baseline_buckets[k])
        m = compute_accuracy(imu_buckets.get(k, []))
        delta = m["az"] - b["az"]
        print(f"{k:<20} {b['n']:<6} {b['az']:<14.2%} {m['az']:<14.2%} {delta:+.2%}")

    print("\nOverall:")
    b_all = compute_accuracy(baseline_records)
    m_all = compute_accuracy(imu_records)
    print(f"  Baseline:  az={b_all['az']:.2%}  el={b_all['el']:.2%}  dist={b_all['dist']:.2%}")
    print(f"  IMU model: az={m_all['az']:.2%}  el={m_all['el']:.2%}  dist={m_all['dist']:.2%}")

    # --- Save plot and JSON ---
    make_plot(baseline_buckets, imu_buckets, output_dir / "comparison.png")

    with open(output_dir / "results.json", "w") as f:
        json.dump({
            "baseline": {k: compute_accuracy(v) for k, v in baseline_buckets.items()},
            "imu": {k: compute_accuracy(v) for k, v in imu_buckets.items()},
            "overall_baseline": b_all,
            "overall_imu": m_all,
        }, f, indent=2)

    print(f"\n✓ Results saved to {output_dir}/")


if __name__ == "__main__":
    main()
