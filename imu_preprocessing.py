"""
IMU Preprocessing for PhaseCoder Dynamic Geometry Path

Converts raw high-rate IMU quaternion samples into STFT-frame-aligned
quaternions suitable for PhaseCoder's `imu_orientations` argument.

Pipeline:
    Raw IMU quaternions (T_imu, 4) at e.g. 200-1000 Hz
        → sign-flip resolution (avoid SLERP "long way around")
        → SLERP interpolation onto STFT frame timestamps
        → Frame-aligned quaternions (F, 4) ready for PhaseCoder

Key design decisions:
    - Quaternion convention: (w, x, y, z), matching scipy/most IMU drivers.
    - Sign-flip handling: applied as a preprocessing pass before SLERP, so
      consecutive samples take the shortest geodesic path on SO(3).
    - SLERP fallback: degenerate cases (q_a · q_b ≈ 1) fall back to linear
      interpolation for numerical stability, then renormalize.
    - Frame timestamp model: STFT center=False → frames at hop_length intervals
      starting at 0. STFT center=True → frames offset by n_fft/2. Both are
      supported via the `center` argument.
"""

from __future__ import annotations
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core quaternion utilities
# ---------------------------------------------------------------------------

def normalize_quaternions(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize quaternions to unit length along the last dim.

    Args:
        q:   (..., 4) quaternions in (w, x, y, z) order.
        eps: floor on norm to avoid division by zero.

    Returns:
        Unit quaternions of the same shape.
    """
    return q / q.norm(dim=-1, keepdim=True).clamp(min=eps)


def resolve_sign_flips(q: torch.Tensor) -> torch.Tensor:
    """Ensure consecutive quaternions take the shortest geodesic path.

    A quaternion q and -q represent the same rotation, but SLERP between
    q_prev and q_curr will take the long way around the 4-sphere if their
    dot product is negative. This pass walks along the time axis and flips
    the sign of any quaternion whose dot product with its predecessor is
    negative, so that subsequent SLERP always takes the short path.

    Args:
        q: (..., T, 4) quaternions in (w, x, y, z) order. The second-to-last
           dimension is treated as time.

    Returns:
        (..., T, 4) sign-corrected quaternions. The first quaternion along
        the time axis is left unchanged; subsequent quaternions are flipped
        as needed to maintain consistent hemisphere with their predecessor.
    """
    if q.shape[-2] < 2:
        return q.clone()

    # Iterative pass: each q_t depends on the (possibly already-flipped) q_{t-1}.
    # We can't fully vectorize this because flip decisions cascade.
    out = q.clone()
    for t in range(1, out.shape[-2]):
        prev = out[..., t - 1, :]   # (..., 4)
        curr = out[..., t, :]       # (..., 4)
        dot = (prev * curr).sum(dim=-1, keepdim=True)  # (..., 1)
        # Where dot < 0, flip curr.
        out[..., t, :] = torch.where(dot < 0, -curr, curr)
    return out


def slerp(
    q_a: torch.Tensor,
    q_b: torch.Tensor,
    t: torch.Tensor,
    dot_threshold: float = 0.9995,
) -> torch.Tensor:
    """Spherical linear interpolation between two unit quaternions.

    Falls back to normalized linear interpolation (NLERP) when the angle
    between q_a and q_b is small enough that SLERP becomes numerically
    unstable.

    Args:
        q_a:           (..., 4) start quaternion (unit length).
        q_b:           (..., 4) end quaternion (unit length, same hemisphere as q_a).
        t:             (...,) interpolation parameter in [0, 1]. t=0 → q_a, t=1 → q_b.
        dot_threshold: above this dot product, fall back to NLERP for stability.

    Returns:
        (..., 4) interpolated unit quaternions.

    Note:
        Assumes q_a and q_b are already in the same hemisphere (dot >= 0).
        Run `resolve_sign_flips` on the input series first.
    """
    dot = (q_a * q_b).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)  # (..., 1)
    t = t.unsqueeze(-1)  # (..., 1)

    # NLERP fallback path: lerp then normalize.
    nlerp_result = normalize_quaternions((1.0 - t) * q_a + t * q_b)

    # SLERP path: only meaningful when dot < threshold.
    # Compute it generically; we'll select per-element below.
    omega = torch.acos(dot.clamp(-1.0, 1.0))  # (..., 1) angle between q_a, q_b
    sin_omega = torch.sin(omega).clamp(min=1e-8)
    coef_a = torch.sin((1.0 - t) * omega) / sin_omega
    coef_b = torch.sin(t * omega) / sin_omega
    slerp_result = coef_a * q_a + coef_b * q_b

    # Per-element selection: where dot is high, use NLERP; otherwise SLERP.
    use_nlerp = dot > dot_threshold  # (..., 1)
    return torch.where(use_nlerp, nlerp_result, slerp_result)


# ---------------------------------------------------------------------------
# STFT frame timestamp helpers
# ---------------------------------------------------------------------------

def stft_frame_timestamps(
    num_frames: int,
    hop_length: int,
    sample_rate: int,
    n_fft: int = 256,
    center: bool = True,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Compute the timestamp (in seconds) of each STFT frame center.

    For torch.stft with center=True, frame f is centered at sample
    (f * hop_length); the reflection padding makes frame 0 a "virtual"
    centered window at t=0. For center=False, frame f starts at sample
    (f * hop_length) and is centered at sample (f * hop_length + n_fft/2).

    Args:
        num_frames:  F, number of STFT frames produced.
        hop_length:  STFT hop in samples.
        sample_rate: audio sample rate in Hz.
        n_fft:       STFT window size in samples (only used when center=False).
        center:      whether torch.stft was called with center=True.
        device:      device for the output tensor.
        dtype:       dtype for the output tensor (default float64 for timing precision).

    Returns:
        (num_frames,) timestamps in seconds, monotonically increasing.
    """
    f = torch.arange(num_frames, device=device, dtype=dtype)
    if center:
        # Frame f is centered at sample (f * hop_length).
        sample_idx = f * hop_length
    else:
        # Frame f covers samples [f*hop, f*hop + n_fft); center at f*hop + n_fft/2.
        sample_idx = f * hop_length + n_fft / 2
    return sample_idx / sample_rate


# ---------------------------------------------------------------------------
# Main preprocessing function
# ---------------------------------------------------------------------------

def imu_to_frame_quaternions(
    imu_quats: torch.Tensor,
    imu_timestamps: torch.Tensor,
    frame_timestamps: torch.Tensor,
    *,
    extrapolation: str = "clamp",
    dot_threshold: float = 0.9995,
) -> torch.Tensor:
    """Convert raw high-rate IMU quaternions to STFT-frame-aligned quaternions.

    Pipeline:
        1. Normalize raw quaternions to unit length.
        2. Resolve sign flips along the time axis (shortest path).
        3. For each frame timestamp, find bracketing IMU samples.
        4. SLERP between the brackets onto the frame timestamp.

    Args:
        imu_quats:        (B, T_imu, 4) raw quaternions in (w, x, y, z) order.
        imu_timestamps:   (B, T_imu) or (T_imu,) timestamps in seconds, monotonic.
                          If 1D, the same timeline is broadcast across the batch.
        frame_timestamps: (F,) or (B, F) STFT frame timestamps in seconds.
                          From `stft_frame_timestamps()`.
        extrapolation:    'clamp' → frames outside IMU range use endpoint quat;
                          'error' → raise if any frame is outside IMU range.
        dot_threshold:    SLERP/NLERP switching threshold.

    Returns:
        (B, F, 4) unit quaternions ready for PhaseCoder's `imu_orientations`.

    Raises:
        ValueError on shape mismatches, non-monotonic timestamps, or
        out-of-range frames when extrapolation='error'.
    """
    if imu_quats.dim() != 3 or imu_quats.shape[-1] != 4:
        raise ValueError(
            f"imu_quats must have shape (B, T_imu, 4); got {tuple(imu_quats.shape)}"
        )
    B, T_imu, _ = imu_quats.shape

    # Broadcast 1D imu_timestamps to (B, T_imu) for uniform handling.
    if imu_timestamps.dim() == 1:
        if imu_timestamps.shape[0] != T_imu:
            raise ValueError(
                f"imu_timestamps length {imu_timestamps.shape[0]} != T_imu {T_imu}"
            )
        imu_timestamps = imu_timestamps.unsqueeze(0).expand(B, -1)
    elif imu_timestamps.shape != (B, T_imu):
        raise ValueError(
            f"imu_timestamps must be (T_imu,) or (B, T_imu); "
            f"got {tuple(imu_timestamps.shape)}"
        )

    # Verify monotonicity (cheap O(T_imu) check, helps catch upstream bugs early).
    if (imu_timestamps[:, 1:] < imu_timestamps[:, :-1]).any():
        raise ValueError("imu_timestamps must be monotonically non-decreasing")

    # Broadcast frame_timestamps to (B, F).
    if frame_timestamps.dim() == 1:
        F_frames = frame_timestamps.shape[0]
        frame_timestamps = frame_timestamps.unsqueeze(0).expand(B, -1)
    elif frame_timestamps.dim() == 2:
        if frame_timestamps.shape[0] != B:
            raise ValueError(
                f"frame_timestamps batch dim {frame_timestamps.shape[0]} != B {B}"
            )
        F_frames = frame_timestamps.shape[1]
    else:
        raise ValueError(
            f"frame_timestamps must be (F,) or (B, F); "
            f"got {tuple(frame_timestamps.shape)}"
        )

    # Range check / clamp.
    imu_min = imu_timestamps[:, 0:1]    # (B, 1)
    imu_max = imu_timestamps[:, -1:]    # (B, 1)
    out_of_range = (frame_timestamps < imu_min) | (frame_timestamps > imu_max)
    if out_of_range.any():
        if extrapolation == "error":
            raise ValueError(
                "frame_timestamps contain values outside the IMU timestamp range; "
                "either provide IMU samples covering the full clip or set "
                "extrapolation='clamp'."
            )
        # 'clamp' path: clamp targets into range, SLERP will then resolve to endpoints.
        frame_timestamps = torch.maximum(frame_timestamps, imu_min)
        frame_timestamps = torch.minimum(frame_timestamps, imu_max)

    # 1. Normalize.
    q_norm = normalize_quaternions(imu_quats)

    # 2. Resolve sign flips along the time axis.
    q_consistent = resolve_sign_flips(q_norm)

    # 3 & 4. For each (b, f), find bracketing IMU indices and SLERP.
    # We use torch.searchsorted for vectorized bracket lookup. Note searchsorted
    # operates per-row when given matching batch dims, and requires contiguous
    # inputs for performance (otherwise PyTorch issues a copy warning).
    #
    # right_idx[b, f] = index of first imu_t > frame_t[b, f], in [1, T_imu].
    imu_timestamps_c = imu_timestamps.contiguous()
    frame_timestamps_c = frame_timestamps.contiguous()
    right_idx = torch.searchsorted(imu_timestamps_c, frame_timestamps_c, right=True)
    right_idx = right_idx.clamp(min=1, max=T_imu - 1)  # ensure valid bracket
    left_idx = right_idx - 1

    # Gather quaternions at bracket indices. q_consistent: (B, T_imu, 4).
    # We need to gather along dim=1 with indices of shape (B, F).
    # Expand index to (B, F, 4) for gather.
    left_idx_exp = left_idx.unsqueeze(-1).expand(-1, -1, 4)   # (B, F, 4)
    right_idx_exp = right_idx.unsqueeze(-1).expand(-1, -1, 4) # (B, F, 4)
    q_left = torch.gather(q_consistent, dim=1, index=left_idx_exp)   # (B, F, 4)
    q_right = torch.gather(q_consistent, dim=1, index=right_idx_exp) # (B, F, 4)

    # Gather corresponding timestamps.
    t_left = torch.gather(imu_timestamps, dim=1, index=left_idx)    # (B, F)
    t_right = torch.gather(imu_timestamps, dim=1, index=right_idx)  # (B, F)

    # Compute interpolation parameter, guarding against zero-duration brackets.
    duration = (t_right - t_left).clamp(min=1e-12)
    alpha = ((frame_timestamps - t_left) / duration).clamp(0.0, 1.0)  # (B, F)

    # Cast alpha to the quaternion dtype for SLERP.
    alpha = alpha.to(q_left.dtype)

    # 4. SLERP.
    q_frames = slerp(q_left, q_right, alpha, dot_threshold=dot_threshold)

    # Final renormalization (SLERP results are unit-norm in theory, but float drift
    # over many calls can compound; one renorm here is cheap insurance).
    q_frames = normalize_quaternions(q_frames)

    return q_frames


# ---------------------------------------------------------------------------
# Convenience: end-to-end from raw IMU + audio config
# ---------------------------------------------------------------------------

def prepare_imu_for_phasecoder(
    imu_quats: torch.Tensor,
    imu_sample_rate: float,
    num_frames: int,
    *,
    audio_sample_rate: int = 16000,
    n_fft: int = 256,
    hop_length: int = 128,
    center: bool = True,
    imu_start_offset: float = 0.0,
    extrapolation: str = "clamp",
) -> torch.Tensor:
    """One-shot convenience wrapper for typical PhaseCoder use.

    Assumes the IMU samples are uniformly spaced at imu_sample_rate, starting
    at time `imu_start_offset` seconds relative to the audio clip start (t=0).

    Args:
        imu_quats:         (B, T_imu, 4) raw quaternions.
        imu_sample_rate:   IMU sampling rate in Hz (e.g. 200, 500, 1000).
        num_frames:        F, the STFT frame count from your audio extractor.
        audio_sample_rate: audio sample rate (default 16000).
        n_fft:             STFT window (default 256).
        hop_length:        STFT hop (default 128).
        center:            torch.stft center mode (default True).
        imu_start_offset:  IMU sample 0's timestamp in seconds (default 0).
                           Use this to align an IMU stream that started before
                           or after the audio clip.
        extrapolation:     'clamp' or 'error', see imu_to_frame_quaternions.

    Returns:
        (B, F, 4) unit quaternions ready for PhaseCoder.
    """
    B, T_imu, _ = imu_quats.shape
    device = imu_quats.device
    dtype = torch.float64  # use double for timing math

    imu_timestamps = (
        torch.arange(T_imu, device=device, dtype=dtype) / imu_sample_rate
        + imu_start_offset
    )
    frame_timestamps = stft_frame_timestamps(
        num_frames=num_frames,
        hop_length=hop_length,
        sample_rate=audio_sample_rate,
        n_fft=n_fft,
        center=center,
        device=device,
        dtype=dtype,
    )

    return imu_to_frame_quaternions(
        imu_quats=imu_quats,
        imu_timestamps=imu_timestamps,
        frame_timestamps=frame_timestamps,
        extrapolation=extrapolation,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _test_normalize():
    q = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 1.0]])
    qn = normalize_quaternions(q)
    norms = qn.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-6), \
        f"Normalization failed: norms={norms}"
    print("  ✓ normalize_quaternions: unit norm preserved")


def _test_sign_flip_resolution():
    # Construct a sequence where every other sample is sign-flipped.
    base = torch.tensor([1.0, 0.0, 0.0, 0.0])  # identity
    seq = torch.stack([base, -base, base, -base, base])  # (5, 4)
    seq = seq.unsqueeze(0)  # (1, 5, 4)

    fixed = resolve_sign_flips(seq)
    # After fix, all should match the first quaternion (identity).
    expected = base.unsqueeze(0).unsqueeze(0).expand(1, 5, 4)
    assert torch.allclose(fixed, expected), \
        f"Sign flip resolution failed: got {fixed}"
    print("  ✓ resolve_sign_flips: alternating sign sequence corrected")

    # Sanity: an already-consistent sequence should be unchanged.
    consistent = base.unsqueeze(0).unsqueeze(0).expand(1, 5, 4).contiguous()
    out = resolve_sign_flips(consistent)
    assert torch.allclose(out, consistent), "Consistent sequence was modified"
    print("  ✓ resolve_sign_flips: already-consistent sequence unchanged")


def _test_slerp_endpoints():
    q_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])   # identity
    # 90° rotation about z: cos(45°), 0, 0, sin(45°)
    angle = torch.tensor(torch.pi / 4)
    q_b = torch.tensor([[torch.cos(angle), 0.0, 0.0, torch.sin(angle)]])

    t0 = torch.tensor([0.0])
    t1 = torch.tensor([1.0])
    t_half = torch.tensor([0.5])

    out0 = slerp(q_a, q_b, t0)
    out1 = slerp(q_a, q_b, t1)
    out_half = slerp(q_a, q_b, t_half)

    assert torch.allclose(out0, q_a, atol=1e-6), f"SLERP t=0 failed: {out0} vs {q_a}"
    assert torch.allclose(out1, q_b, atol=1e-6), f"SLERP t=1 failed: {out1} vs {q_b}"
    # t=0.5 should be the geodesic midpoint: 45° rotation about z.
    half_angle = torch.tensor(torch.pi / 8)
    expected_half = torch.tensor(
        [[torch.cos(half_angle), 0.0, 0.0, torch.sin(half_angle)]]
    )
    assert torch.allclose(out_half, expected_half, atol=1e-5), \
        f"SLERP t=0.5 failed: {out_half} vs {expected_half}"
    print("  ✓ slerp: endpoints and midpoint correct for 90° rotation")


def _test_slerp_close_quaternions():
    # When q_a ≈ q_b, SLERP should fall back to NLERP without numerical issues.
    q_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    eps = 1e-4
    q_b = torch.tensor([[1.0, eps, 0.0, 0.0]])
    q_b = normalize_quaternions(q_b)

    t = torch.tensor([0.5])
    out = slerp(q_a, q_b, t)
    norm = out.norm(dim=-1)
    assert torch.allclose(norm, torch.ones_like(norm), atol=1e-5), \
        f"SLERP close-quaternion result not unit norm: {norm}"
    assert not torch.isnan(out).any(), "SLERP produced NaN for close quaternions"
    print("  ✓ slerp: NLERP fallback handles near-identical quaternions")


def _test_frame_timestamps():
    # 33 frames, 128-sample hop, 16kHz, center=True.
    ts = stft_frame_timestamps(33, hop_length=128, sample_rate=16000, center=True)
    expected_first = 0.0
    expected_last = 32 * 128 / 16000  # 0.256s
    assert abs(ts[0].item() - expected_first) < 1e-9, f"First timestamp: {ts[0]}"
    assert abs(ts[-1].item() - expected_last) < 1e-9, f"Last timestamp: {ts[-1]}"
    # Monotonic increase.
    assert (ts[1:] > ts[:-1]).all(), "Timestamps not monotonically increasing"
    print(f"  ✓ stft_frame_timestamps: 33 frames span [0, {expected_last:.4f}]s")


def _test_end_to_end():
    """Full pipeline: synthetic IMU stream → frame quaternions."""
    torch.manual_seed(0)

    B = 2
    imu_rate = 500.0  # Hz
    duration = 0.260  # seconds, slightly longer than 250ms clip
    T_imu = int(imu_rate * duration)

    # Build a synthetic IMU stream: rotation about z increasing linearly with time.
    # At t=0: identity. At t=duration: 60° rotation.
    timestamps = torch.arange(T_imu, dtype=torch.float64) / imu_rate
    angles = (timestamps / duration) * (torch.pi / 3)  # 0 → 60°
    q_w = torch.cos(angles / 2).to(torch.float32)
    q_z = torch.sin(angles / 2).to(torch.float32)
    q_x = torch.zeros_like(q_w)
    q_y = torch.zeros_like(q_w)
    imu_quats = torch.stack([q_w, q_x, q_y, q_z], dim=-1)  # (T_imu, 4)
    imu_quats = imu_quats.unsqueeze(0).expand(B, -1, -1).contiguous()  # (B, T_imu, 4)

    # Inject a sign flip at sample 50 to test resolution.
    imu_quats[:, 50:75] = -imu_quats[:, 50:75]

    frame_quats = prepare_imu_for_phasecoder(
        imu_quats=imu_quats,
        imu_sample_rate=imu_rate,
        num_frames=33,
        audio_sample_rate=16000,
        n_fft=256,
        hop_length=128,
        center=True,
    )

    assert frame_quats.shape == (B, 33, 4), f"Output shape: {frame_quats.shape}"
    norms = frame_quats.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"Frame quaternions not unit norm: max dev {(norms - 1).abs().max()}"

    # Frame 0 (t=0) should be ~identity.
    assert torch.allclose(frame_quats[0, 0, 0].abs(), torch.tensor(1.0), atol=1e-3), \
        f"Frame 0 not identity: {frame_quats[0, 0]}"

    # Last frame (t≈0.256s) should be rotation by ~ (0.256/0.260)*60° ≈ 59.08°.
    # Expected w = cos(59.08°/2) ≈ cos(29.54°) ≈ 0.8703
    expected_w = torch.cos(torch.tensor(0.256 / 0.260 * torch.pi / 3 / 2))
    actual_w = frame_quats[0, -1, 0].abs()
    assert torch.allclose(actual_w, expected_w, atol=1e-3), \
        f"Last frame w: expected {expected_w}, got {actual_w}"

    print(f"  ✓ end-to-end: 60°/clip rotation correctly interpolated to 33 frames")
    print(f"    Frame 0 quat:  {frame_quats[0, 0].tolist()}")
    print(f"    Frame 16 quat: {frame_quats[0, 16].tolist()}")
    print(f"    Frame 32 quat: {frame_quats[0, -1].tolist()}")


def _test_extrapolation_error():
    imu_quats = torch.tensor([[[1.0, 0.0, 0.0, 0.0]] * 5]).float()  # (1, 5, 4)
    imu_ts = torch.linspace(0.0, 0.01, 5)  # 0 to 10ms
    frame_ts = torch.tensor([0.005, 0.020])  # second frame is past IMU end

    try:
        imu_to_frame_quaternions(imu_quats, imu_ts, frame_ts, extrapolation="error")
        assert False, "Expected ValueError for out-of-range frame"
    except ValueError as e:
        assert "outside the IMU timestamp range" in str(e)
    print("  ✓ extrapolation='error': out-of-range frames raise ValueError")

    # Clamp mode should produce the endpoint quaternion for out-of-range frames.
    out = imu_to_frame_quaternions(imu_quats, imu_ts, frame_ts, extrapolation="clamp")
    assert torch.allclose(out[0, 1], imu_quats[0, -1], atol=1e-5), \
        "Clamp mode did not produce endpoint quaternion"
    print("  ✓ extrapolation='clamp': out-of-range frames use endpoint")


def _test_phasecoder_integration():
    """Full round-trip: raw IMU → frame quaternions → PhaseCoder.forward."""
    from PhaseCoder import PhaseCoder

    torch.manual_seed(42)
    B = 4
    C = 8
    sr = 16000
    dur = 0.25  # 250ms
    T_audio = int(sr * dur)
    imu_rate = 500.0
    T_imu = int(imu_rate * dur) + 10  # 135 samples — covers the clip with headroom

    # Random unit quaternions across the batch.
    raw = torch.randn(B, T_imu, 4)
    raw = normalize_quaternions(raw)

    # Compute the STFT frame count analytically for center=True:
    # PyTorch pads by n_fft//2 on each side → F = 1 + floor(T_audio / hop_length).
    n_fft, hop_length = 256, 128
    F = 1 + T_audio // hop_length  # = 32 for T_audio=4000

    frame_quats = prepare_imu_for_phasecoder(
        imu_quats=raw,
        imu_sample_rate=imu_rate,
        num_frames=F,
        audio_sample_rate=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        center=True,
    )

    # Verify preprocessing output independently.
    assert frame_quats.shape == (B, F, 4), f"Unexpected shape: {frame_quats.shape}"
    assert frame_quats.dtype in (torch.float32, torch.float64)
    norms = frame_quats.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5), \
        f"Frame quaternions not unit norm: max dev {(norms - 1).abs().max()}"
    print(f"  ✓ preprocessing: shape ({B}, {F}, 4), unit norm")

    # Full PhaseCoder forward pass.
    model = PhaseCoder()
    model.eval()
    audio = torch.randn(B, C, T_audio)
    mic_coords = torch.randn(B, C, 3) * 0.05  # ~5cm array

    with torch.no_grad():
        out = model(audio, mic_coords, imu_orientations=frame_quats)

    assert out["spatial_embedding"].shape == (B, 256), \
        f"spatial_embedding shape: {out['spatial_embedding'].shape}"
    assert out["azimuth_logits"].shape == (B, 39), \
        f"azimuth_logits shape: {out['azimuth_logits'].shape}"
    assert out["elevation_logits"].shape == (B, 19), \
        f"elevation_logits shape: {out['elevation_logits'].shape}"
    assert out["distance_logits"].shape == (B, 14), \
        f"distance_logits shape: {out['distance_logits'].shape}"
    assert not any(v.isnan().any() for v in out.values()), "NaN in model output"

    print(f"  ✓ PhaseCoder.forward: all output shapes correct, no NaNs")


if __name__ == "__main__":
    print("Running IMU preprocessing tests...\n")
    print("Quaternion utilities:")
    _test_normalize()
    _test_sign_flip_resolution()
    _test_slerp_endpoints()
    _test_slerp_close_quaternions()

    print("\nFrame timestamp helpers:")
    _test_frame_timestamps()

    print("\nEnd-to-end pipeline:")
    _test_end_to_end()
    _test_extrapolation_error()
    _test_phasecoder_integration()

    print("\n✓ All tests passed.")
