"""
This file handles all physics of mapping where the microphones move
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class JerkToMotion:
    """
    Pure ODE integrator: jerk -> acceleration -> velocity -> position.
    No air resistance, no drag. Uses RK4 for accuracy.

    Inputs:
        jerk:  (B, T, 3)  -- jerk in x/y/z at each timestep
        dt:    float      -- time delta between steps (seconds)
        x0:    (B, 3)     -- initial position
        v0:    (B, 3)     -- initial velocity
        a0:    (B, 3)     -- initial acceleration

    Output:
        positions: (B, T, 3)
    """

    @staticmethod
    def _euler_step(j_t, a, v, x, dt):
        a_next = a + j_t * dt
        v_next = v + a_next * dt
        x_next = x + v_next * dt
        return a_next, v_next, x_next

    @staticmethod
    def _rk4_step(j_t, j_t_half, a, v, x, dt):
        """
        RK4 treating acceleration as the primary state variable driven by jerk.
        j_t_half is jerk interpolated at the midpoint (t + dt/2); use the
        average of j_t and j_{t+1} if you don't have it.
        """
        half = dt / 2.0

        # k1
        da1 = j_t
        dv1 = a
        dx1 = v

        # k2  (midpoint using half-step)
        a2 = a + da1 * half
        v2 = v + dv1 * half
        x2 = x + dx1 * half
        da2 = j_t_half
        dv2 = a2
        dx2 = v2

        # k3  (midpoint, second estimate)
        a3 = a + da2 * half
        v3 = v + dv2 * half
        x3 = x + dx2 * half
        da3 = j_t_half
        dv3 = a3
        dx3 = v3

        # k4  (full step)
        a4 = a + da3 * dt
        v4 = v + dv3 * dt
        x4 = x + dx3 * dt
        da4 = j_t          # reuse end-of-step jerk (caller can pass j_{t+1} if available)
        dv4 = a4
        dx4 = v4

        a_next = a + (dt / 6.0) * (da1 + 2*da2 + 2*da3 + da4)
        v_next = v + (dt / 6.0) * (dv1 + 2*dv2 + 2*dv3 + dv4)
        x_next = x + (dt / 6.0) * (dx1 + 2*dx2 + 2*dx3 + dx4)
        return a_next, v_next, x_next

    @classmethod
    def integrate(cls, jerk, dt, x0, v0, a0, method: str = "rk4"):
        """
        Args:
            method: "euler" or "rk4"
        Returns:
            positions (B, T, 3), velocities (B, T, 3), accelerations (B, T, 3)
        """
        _, T, _ = jerk.shape
        positions, velocities, accelerations = [], [], []

        a, v, x = a0.clone(), v0.clone(), x0.clone()

        for t in range(T):
            j_t = jerk[:, t, :]

            if method == "rk4":
                j_mid = (j_t + jerk[:, min(t + 1, T - 1), :]) / 2.0
                a, v, x = cls._rk4_step(j_t, j_mid, a, v, x, dt)
            else:
                a, v, x = cls._euler_step(j_t, a, v, x, dt)

            positions.append(x.unsqueeze(1))
            velocities.append(v.unsqueeze(1))
            accelerations.append(a.unsqueeze(1))

        return (
            torch.cat(positions, dim=1),
            torch.cat(velocities, dim=1),
            torch.cat(accelerations, dim=1),
        )


class MicPositionDecoder(nn.Module):
    """
    Decodes transformer hidden embeddings into per-mic position trajectories.

    Takes a sequence of hidden states from an upstream transformer (e.g. PhaseCoder)
    and uses cross-attention — one query per (frame, mic) pair — to produce a
    (B, T, M, 3) array of microphone positions throughout the audio clip.

    Inputs:
        hidden_emb: (B, T, D) -- hidden states from upstream transformer
        mic_pos:    (B, M, 3) -- neutral-frame mic positions (metres)

    Output:
        positions:  (B, T, M, 3) -- predicted (x, y, z) per mic per frame
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 4,
        num_layers: int = 2,
        max_frames: int = 64,
    ):
        super().__init__()

        # Project neutral mic coords into embedding space to form per-mic queries
        self.mic_proj = nn.Linear(3, embed_dim)

        # Learnable per-frame time embedding added to each mic query
        self.time_embed = nn.Embedding(max_frames, embed_dim)

        # Transformer decoder: (T*M) queries cross-attend over hidden_emb sequence
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        # Output head: decoded embedding → (x, y, z) position
        self.pos_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 3),
        )

    def forward(
        self,
        hidden_emb: torch.Tensor,
        mic_pos: torch.Tensor,
        num_frames: int,
    ) -> torch.Tensor:
        """
        Args:
            hidden_emb: (B, seq_len, D) -- full transformer output sequence,
                        e.g. (B, 1 + C*F, D) including the CLS token from PhaseCoder.
            mic_pos:    (B, M, 3)       -- neutral-frame mic positions (metres).
            num_frames: int             -- number of STFT frames F to decode.
                        Distinct from seq_len; queries are built over F, not seq_len.

        Returns:
            positions: (B, F, M, 3) -- predicted (x, y, z) per mic per STFT frame.
        """
        B, _, D = hidden_emb.shape
        M = mic_pos.shape[1]

        # Build one query per (frame, mic) pair by summing time and mic embeddings.
        mic_emb = self.mic_proj(mic_pos)                                        # (B, M, D)
        t_idx = torch.arange(num_frames, device=hidden_emb.device)
        time_emb = self.time_embed(t_idx)                                       # (F, D)

        # mic_emb:  (B,  1, M, D) — one embedding per mic, shared across frames
        # time_emb: (1,  F, 1, D) — one embedding per frame, shared across mics
        queries = mic_emb.unsqueeze(1) + time_emb.unsqueeze(0).unsqueeze(2)    # (B, F, M, D)
        queries = queries.reshape(B, num_frames * M, D)                         # (B, F*M, D)

        # Each (frame, mic) query cross-attends over the full hidden sequence
        decoded = self.decoder(queries, hidden_emb)                             # (B, F*M, D)

        positions = self.pos_head(decoded)                                      # (B, F*M, 3)
        return positions.reshape(B, num_frames, M, 3)                           # (B, F, M, 3)