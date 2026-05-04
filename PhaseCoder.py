"""
PhaseCoder: Microphone Geometry-Agnostic Spatial Audio Encoder
 
Based on: "PhaseCoder: Microphone Geometry-Agnostic Spatial Audio
Understanding for Multimodal LLMs" (Dementyev et al., 2026)
 
Architecture (Section 3.1):
- STFT-based patch extraction (magnitude + phase, 258-dim per mic/frame)
- Linear projection to D=256
- Three summed positional embeddings: sequential, frame-level, mic-coordinate
- Learnable [CLS] token
- 5-layer ViT-style transformer encoder (4-head, FFN 1x expansion)
- 2-layer MLP for spatial embedding output
- 3 classification heads: azimuth (38), elevation (18), distance (13)
- ~6M parameters
"""
 
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
 
 
class STFTPatchExtractor(nn.Module):
    """
    Extract STFT magnitude+phase patches from raw multichannel audio.
 
    Input:  (B, C, T_samples)  — C channels of 16kHz audio, 250ms = 4000 samples
    Output: (B, C, F, D_stft)  — F=33 frames, D_stft=258 (129 mag + 129 phase)
 
    STFT config: 256-sample Hann window, 128-sample hop → 129 freq bins, 33 frames
    """
 
    def __init__(self, n_fft: int = 256, hop_length: int = 128, win_length: int = 256):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.register_buffer("window", torch.hann_window(win_length))
 
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T_samples)
        Returns:
            patches: (B, C, F, 258)  where 258 = 129 mag + 129 phase
        """
        B, C, T = x.shape
        x_flat = x.reshape(B * C, T)
 
        # STFT → (B*C, n_freq=129, n_frames=33) complex
        spec = torch.stft(
            x_flat,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
            center=True,
            pad_mode="reflect",
        )
        # spec: (B*C, 129, F)
 
        mag = spec.abs()       # (B*C, 129, F)
        phase = spec.angle()   # (B*C, 129, F)
 
        # Concatenate mag and phase along freq dim → (B*C, 258, F)
        feat = torch.cat([mag, phase], dim=1)
 
        # Rearrange to (B*C, F, 258) then reshape to (B, C, F, 258)
        feat = feat.permute(0, 2, 1)
        n_freq = self.n_fft // 2 + 1
        F_frames = feat.shape[1]
        feat = feat.reshape(B, C, F_frames, 2 * n_freq)
        return feat
 
 
class MicrophonePositionalEmbedding(nn.Module):
    """
    Microphone geometry positional embedding adapted from GI-DOAEnet.
 
    Converts Cartesian mic coordinates to spherical (relative to centroid),
    then produces a D-dimensional embedding via parameterized trigonometric fusion:
 
        P_i = alpha * r_i * [cos(2*pi*beta*v + theta_i),
                              sin(2*pi*beta*v + theta_i),
                              cos(2*pi*beta*v + phi_i),
                              sin(2*pi*beta*v + phi_i)]
 
    where v = (4/D) * [0, 1, ..., D/4-1], alpha=7.0, beta=4.0
    """
 
    def __init__(self, embed_dim: int = 256, alpha: float = 7.0, beta: float = 4.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.alpha = alpha
        self.beta = beta
 
        # Base frequency vector: v = (4/D) * [0, 1, ..., D/4-1]
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        v = (4.0 / embed_dim) * torch.arange(embed_dim // 4, dtype=torch.float32)
        self.register_buffer("v", v)
 
    def forward(self, mic_coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            mic_coords: (B, C, 3)    — Cartesian (x, y, z) for each mic, or
                        (B, F, C, 3) — per-frame coords for IMU-driven mode.

        Returns:
            embedding: (B, C, D) or (B, F, C, D) matching the input rank.
        """
        # Centroid over the mic axis (second-to-last spatial dim) — works for both 3D and 4D.
        centroid = mic_coords.mean(dim=-2, keepdim=True)  # (..., 1, 3)
        rel = mic_coords - centroid                        # (..., C, 3)

        x_rel = rel[..., 0]  # (..., C)
        y_rel = rel[..., 1]
        z_rel = rel[..., 2]

        # Spherical coordinates
        r = torch.sqrt(x_rel**2 + y_rel**2 + z_rel**2).clamp(min=1e-8)  # (..., C)
        theta = torch.acos((z_rel / r).clamp(-1.0, 1.0))                 # elevation angle
        phi = torch.atan2(y_rel, x_rel)                                   # azimuth angle

        # v shape (D/4,) — build angle_base that broadcasts over all leading dims.
        # angle_base: (D/4,) broadcast with (..., C, 1) → (..., C, D/4)
        angle_base = 2.0 * math.pi * self.beta * self.v  # (D/4,)

        theta_exp = theta.unsqueeze(-1)  # (..., C, 1)
        phi_exp = phi.unsqueeze(-1)      # (..., C, 1)
        r_exp = r.unsqueeze(-1)          # (..., C, 1)

        cos_theta = torch.cos(angle_base + theta_exp)  # (..., C, D/4)
        sin_theta = torch.sin(angle_base + theta_exp)
        cos_phi = torch.cos(angle_base + phi_exp)
        sin_phi = torch.sin(angle_base + phi_exp)

        # Concatenate to form D-dim vector, scale by alpha * r
        emb = torch.cat([cos_theta, sin_theta, cos_phi, sin_phi], dim=-1)  # (..., C, D)
        emb = self.alpha * r_exp * emb

        return emb
 
 
def quaternion_to_rotation_matrix(q: torch.Tensor) -> torch.Tensor:
    """Convert unit quaternions to rotation matrices.

    Accepts any leading batch dimensions; the last dim must be 4 in (w, x, y, z) order.

    Args:
        q: (..., 4) — quaternions, need not be pre-normalized.

    Returns:
        R: (..., 3, 3) — corresponding SO(3) rotation matrices.
    """
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]

    R = torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    )  # (..., 9)
    return R.reshape(q.shape[:-1] + (3, 3))


def sinusoidal_embedding(length: int, dim: int, device: torch.device) -> torch.Tensor:
    """Standard sinusoidal positional embedding (non-learned).
 
    Returns: (length, dim)
    """
    pos = torch.arange(length, dtype=torch.float32, device=device).unsqueeze(1)
    dim_idx = torch.arange(dim, dtype=torch.float32, device=device).unsqueeze(0)
    angles = pos / (10000.0 ** (2 * (dim_idx // 2) / dim))
    emb = torch.zeros(length, dim, device=device)
    emb[:, 0::2] = torch.sin(angles[:, 0::2])
    emb[:, 1::2] = torch.cos(angles[:, 1::2])
    return emb
 
 
class PhaseCoder(nn.Module):
    """
    PhaseCoder: Transformer-only spatial audio encoder.

    Args:
        embed_dim:           Embedding dimension D (default 256)
        num_heads:           Attention heads (default 4)
        num_layers:          Transformer blocks (default 5)
        ffn_expansion:       FFN inner dim multiplier (default 1, so inner=256)
        n_fft:               STFT window size (default 256)
        hop_length:          STFT hop (default 128)
        num_azimuth:         Azimuth classes (default 38, + 1 no-speech = 39)
        num_elevation:       Elevation classes (default 18, + 1 = 19)
        num_distance:        Distance classes (default 13, + 1 = 14)
        alpha:               Mic embedding scale (default 7.0)
        beta:                Mic embedding freq scale (default 4.0)
        canonical_frame_idx: STFT frame index used as the reference pose when IMU
                             orientations are supplied (Option A). None → F // 2 at
                             forward time (frame 16 for the default 33-frame clip).
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 5,
        ffn_expansion: int = 1,
        n_fft: int = 256,
        hop_length: int = 128,
        num_azimuth: int = 39,
        num_elevation: int = 19,
        num_distance: int = 14,
        alpha: float = 7.0,
        beta: float = 4.0,
        canonical_frame_idx: int | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.canonical_frame_idx = canonical_frame_idx
 
        # --- Input feature extraction ---
        self.stft_extractor = STFTPatchExtractor(n_fft=n_fft, hop_length=hop_length, win_length=n_fft)
        stft_feat_dim = 2 * (n_fft // 2 + 1)  # 258 for n_fft=256
        self.patch_proj = nn.Linear(stft_feat_dim, embed_dim)
 
        # --- Positional embeddings ---
        self.mic_pos_embed = MicrophonePositionalEmbedding(embed_dim, alpha=alpha, beta=beta)
        # Sequential and frame embeddings are computed on-the-fly (non-learned, sinusoidal)
 
        # --- [CLS] token ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
 
        # --- Transformer encoder ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * ffn_expansion,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)
 
        # --- Output: 2-layer MLP for spatial embedding ---
        self.spatial_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
 
        # --- Classification heads (1-layer MLP each) ---
        self.azimuth_head = nn.Linear(embed_dim, num_azimuth)
        self.elevation_head = nn.Linear(embed_dim, num_elevation)
        self.distance_head = nn.Linear(embed_dim, num_distance)
 
        self._init_weights()
 
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
 
    def forward(
        self,
        audio: torch.Tensor,
        mic_coords: torch.Tensor,
        imu_orientations: torch.Tensor | None = None,
    ) -> dict:
        """
        Args:
            audio:            (B, C, T_samples) — raw multichannel audio at 16kHz.
                              For 250ms input: T_samples = 4000.
            mic_coords:       (B, C, 3) — Cartesian (x,y,z) mic positions in the
                              device-neutral (manufacturer) frame, in metres.
            imu_orientations: (B, F, 4) — unit quaternions (w, x, y, z) per STFT
                              frame, SLERP-aligned to the F time steps by the data
                              pipeline. Pass None (default) to reproduce the original
                              static-geometry behaviour.

        Returns:
            dict with keys:
                'spatial_embedding': (B, D)  — spatial soft token for LLM integration.
                'azimuth_logits':    (B, num_azimuth)
                'elevation_logits':  (B, num_elevation)
                'distance_logits':   (B, num_distance)

            When imu_orientations is supplied, predictions are expressed in the
            device-instantaneous frame at the canonical STFT frame index
            (canonical_frame_idx, default F // 2).
        """
        B, C, T = audio.shape
        device = audio.device

        # 1. STFT patch extraction → (B, C, F, 258)
        patches = self.stft_extractor(audio)
        F_frames = patches.shape[2]

        # 2. Linear projection → (B, C, F, D)
        patches = self.patch_proj(patches)

        # 3. Positional embeddings
        # 3a. Mic coordinate embedding
        if imu_orientations is None:
            # Static path: single embedding per mic, broadcast across frames.
            mic_emb = self.mic_pos_embed(mic_coords)  # (B, C, D)
            mic_emb = mic_emb.unsqueeze(2).expand(-1, -1, F_frames, -1)  # (B, C, F, D)
        else:
            # Dynamic path: per-frame mic positions in device-instantaneous frame,
            # expressed relative to the canonical clip pose (Option A).
            if imu_orientations.shape != (B, F_frames, 4):
                raise ValueError(
                    f"imu_orientations must have shape ({B}, {F_frames}, 4); "
                    f"got {tuple(imu_orientations.shape)}"
                )
            ref = self.canonical_frame_idx if self.canonical_frame_idx is not None else F_frames // 2
            if not (0 <= ref < F_frames):
                raise ValueError(
                    f"canonical_frame_idx {ref} is out of range [0, {F_frames})"
                )

            # Absolute rotation matrices for every frame: (B, F, 3, 3)
            R_abs = quaternion_to_rotation_matrix(imu_orientations)

            # Express all frames relative to the canonical pose so that at f=ref
            # the rotation is identity and the CLS prediction is in that device pose.
            R_ref_T = R_abs[:, ref].transpose(-1, -2).unsqueeze(1)  # (B, 1, 3, 3)
            R_rel = torch.matmul(R_ref_T, R_abs)                     # (B, F, 3, 3)

            # Rotate the static neutral mic coords per frame: (B, F, C, 3)
            mic_inst = torch.einsum("bfij,bcj->bfci", R_rel, mic_coords)

            # Per-frame embedding: (B, F, C, D) → permute → (B, C, F, D)
            mic_emb = self.mic_pos_embed(mic_inst).permute(0, 2, 1, 3)
 
        # 3b. Frame embedding (sinusoidal, same for all mics within a frame)
        frame_emb = sinusoidal_embedding(F_frames, self.embed_dim, device)  # (F, D)
        frame_emb = frame_emb.unsqueeze(0).unsqueeze(0)  # (1, 1, F, D)
 
        # 3c. Flatten channel and frame dims → sequence
        # patches: (B, C, F, D) → (B, C*F, D)
        patches = patches.reshape(B, C * F_frames, self.embed_dim)
        mic_emb = mic_emb.reshape(B, C * F_frames, self.embed_dim)
        frame_emb_expanded = frame_emb.expand(B, C, F_frames, self.embed_dim).reshape(
            B, C * F_frames, self.embed_dim
        )
 
        # 3d. Sequential embedding (based on position 1..L in flattened sequence)
        L = C * F_frames
        seq_emb = sinusoidal_embedding(L, self.embed_dim, device)  # (L, D)
        seq_emb = seq_emb.unsqueeze(0)  # (1, L, D)
 
        # 3e. Sum all three positional embeddings into patches
        patches = patches + seq_emb + frame_emb_expanded + mic_emb
 
        # 4. Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([cls_tokens, patches], dim=1)  # (B, 1 + L, D)
 
        # 5. Transformer encoder
        x = self.transformer(x)
        x = self.norm(x)
 
        # 6. Extract [CLS] representation
        cls_out = x[:, 0]  # (B, D)
 
        # 7. Spatial embedding via 2-layer MLP
        spatial_emb = self.spatial_mlp(cls_out)  # (B, D)
 
        # 8. Classification heads
        az_logits = self.azimuth_head(spatial_emb)    # (B, 39)
        el_logits = self.elevation_head(spatial_emb)  # (B, 19)
        dist_logits = self.distance_head(spatial_emb) # (B, 14)
 
        return {
            "spatial_embedding": spatial_emb,
            "azimuth_logits": az_logits,
            "elevation_logits": el_logits,
            "distance_logits": dist_logits,
        }
 
 
class PhaseCoderLoss(nn.Module):
    """
    Weighted cross-entropy loss for the three classification heads.
 
    L = lambda_az * CE_azimuth + lambda_el * CE_elevation + lambda_dist * CE_distance
    """
 
    def __init__(
        self,
        lambda_az: float = 1.0,
        lambda_el: float = 1.0,
        lambda_dist: float = 0.5,
    ):
        super().__init__()
        self.lambda_az = lambda_az
        self.lambda_el = lambda_el
        self.lambda_dist = lambda_dist
        self.ce = nn.CrossEntropyLoss()
 
    def forward(self, outputs: dict, targets: dict) -> dict:
        """
        Args:
            outputs: dict from PhaseCoder.forward()
            targets: dict with keys 'azimuth', 'elevation', 'distance' — each (B,) int
 
        Returns:
            dict with 'loss' (scalar) and per-head losses
        """
        loss_az = self.ce(outputs["azimuth_logits"], targets["azimuth"])
        loss_el = self.ce(outputs["elevation_logits"], targets["elevation"])
        loss_dist = self.ce(outputs["distance_logits"], targets["distance"])
 
        total = self.lambda_az * loss_az + self.lambda_el * loss_el + self.lambda_dist * loss_dist
 
        return {
            "loss": total,
            "loss_azimuth": loss_az,
            "loss_elevation": loss_el,
            "loss_distance": loss_dist,
        }
 
 
# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    B = 4       # batch size
    C = 8       # 8 microphones
    sr = 16000  # 16kHz
    dur = 0.25  # 250ms
    T = int(sr * dur)  # 4000 samples
 
    model = PhaseCoder().to(device)
 
    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"PhaseCoder parameters: {n_params:,} (~{n_params/1e6:.1f}M)")
 
    # Dummy input
    audio = torch.randn(B, C, T, device=device)
    mic_coords = torch.randn(B, C, 3, device=device) * 0.1  # ~10cm array
 
    out = model(audio, mic_coords)
    print(f"Spatial embedding shape: {out['spatial_embedding'].shape}")
    print(f"Azimuth logits shape:    {out['azimuth_logits'].shape}")
    print(f"Elevation logits shape:  {out['elevation_logits'].shape}")
    print(f"Distance logits shape:   {out['distance_logits'].shape}")
 
    # Test loss
    targets = {
        "azimuth": torch.randint(0, 39, (B,), device=device),
        "elevation": torch.randint(0, 19, (B,), device=device),
        "distance": torch.randint(0, 14, (B,), device=device),
    }
    criterion = PhaseCoderLoss()
    losses = criterion(out, targets)
    print(f"Total loss: {losses['loss'].item():.4f}")

    # --- IMU-driven dynamic geometry tests (via imu_preprocessing) ---
    from imu_preprocessing import prepare_imu_for_phasecoder

    # Derive the actual frame count from the STFT extractor to avoid hardcoding.
    with torch.no_grad():
        _probe = model.stft_extractor(audio[:1, :1])
    F_frames = _probe.shape[2]

    # Simulate a 500 Hz IMU stream for the 250ms clip, with ~5 samples of headroom
    # on each side so the frame timestamps always fall within the IMU range.
    imu_rate = 500.0
    T_imu = int(imu_rate * dur) + 10  # 135 samples
    raw_imu = torch.randn(B, T_imu, 4)
    raw_imu = raw_imu / raw_imu.norm(dim=-1, keepdim=True)

    # (a) All-identity quaternions — R_rel = I at every frame, equivalent to the
    #     static path for the mic embedding at the canonical frame.
    ident_raw = torch.zeros(B, T_imu, 4)
    ident_raw[..., 0] = 1.0
    ident_quats = prepare_imu_for_phasecoder(
        imu_quats=ident_raw,
        imu_sample_rate=imu_rate,
        num_frames=F_frames,
        audio_sample_rate=sr,
        n_fft=256,
        hop_length=128,
    ).to(device)
    out_ident = model(audio, mic_coords, imu_orientations=ident_quats)
    print(f"\n[IMU id]  Spatial embedding shape: {out_ident['spatial_embedding'].shape}")
    print(f"[IMU id]  Azimuth logits shape:    {out_ident['azimuth_logits'].shape}")

    # (b) SLERP-interpolated random-rotation stream — exercises the full per-frame
    #     rotation path through prepare_imu_for_phasecoder → PhaseCoder.forward.
    rand_quats = prepare_imu_for_phasecoder(
        imu_quats=raw_imu,
        imu_sample_rate=imu_rate,
        num_frames=F_frames,
        audio_sample_rate=sr,
        n_fft=256,
        hop_length=128,
    ).to(device)
    out_dyn = model(audio, mic_coords, imu_orientations=rand_quats)
    print(f"\n[IMU dyn] Spatial embedding shape: {out_dyn['spatial_embedding'].shape}")
    print(f"[IMU dyn] Azimuth logits shape:    {out_dyn['azimuth_logits'].shape}")

    losses_dyn = criterion(out_dyn, targets)
    print(f"[IMU dyn] Total loss: {losses_dyn['loss'].item():.4f}")