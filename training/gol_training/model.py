"""Lightweight 1D-CNN temporal model and calibrated logistic fusion."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from .features import TEMPORAL_FEATURE_DIM, TEMPORAL_SEQ_LEN


class TemporalTrackerNet(nn.Module):
    """
    Compact Conv1D + GRU temporal tracker over fixed-rate resampled hiding-phase
    feature sequences [B, T=64, F=24].
    Tracks sequential hand-to-hand transfers (0, 1, 2+) and predicts raw logit for P(gol in LEFT).
    """

    def __init__(self, feature_dim: int = TEMPORAL_FEATURE_DIM, hidden_dim: int = 24) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stem = nn.Sequential(
            nn.Conv1d(feature_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )
        self.attn_proj = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(p=0.15)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 24),
            nn.ReLU(),
            nn.Linear(24, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        # Mask valid non-padded timesteps (features other than padding are non-zero)
        valid_mask = (x.abs().sum(dim=-1, keepdim=True) > 1e-6).to(x.dtype)  # [B, T, 1]
        h_conv = self.stem(x.transpose(1, 2)).transpose(1, 2)  # [B, T, C]
        h0 = torch.zeros(1, x.shape[0], self.hidden_dim, dtype=x.dtype, device=x.device)
        gru_out, _ = self.gru(h_conv, h0)  # [B, T, C]

        # Weighted pooling over valid steps (heavily favors final valid state via learned attention + masked mean)
        attn_logits = self.attn_proj(gru_out) + (1.0 - valid_mask) * (-1e4)
        attn_weights = torch.softmax(attn_logits, dim=1)
        attn_pool = torch.sum(gru_out * attn_weights, dim=1)  # [B, C]

        # Extract last non-padded step via valid-step index weight or masked average of tail
        # Using cumulative valid mask difference gives exact last valid timestep without dynamic indexing
        next_mask = torch.cat(
            [valid_mask[:, 1:, :], torch.zeros_like(valid_mask[:, :1, :])], dim=1
        )
        last_step_mask = torch.clamp(valid_mask - next_mask, min=0.0)
        last_hidden = torch.sum(gru_out * last_step_mask, dim=1)  # [B, C]

        feats = torch.cat([last_hidden, attn_pool], dim=-1)
        feats = self.dropout(feats)
        return self.head(feats)  # [B, 1]


def safe_logit(p: np.ndarray | float) -> np.ndarray | float:
    eps = 1e-4
    if isinstance(p, (int, float)):
        cp = max(eps, min(1.0 - eps, float(p)))
        return math.log(cp / (1.0 - cp))
    cp = np.clip(p, eps, 1.0 - eps)
    return np.log(cp / (1.0 - cp))


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    x_clip = np.clip(x, -25.0, 25.0)
    return 1.0 / (1.0 + np.exp(-x_clip))


@dataclass(frozen=True)
class CalibratedFusionWeights:
    hmm_weight: float
    temporal_weight: float
    bias: float

    def predict_proba(self, hmm_p_left: np.ndarray, temporal_p_left: np.ndarray) -> np.ndarray:
        hmm_logits = np.asarray(safe_logit(hmm_p_left), dtype=np.float64)
        temp_logits = np.asarray(safe_logit(temporal_p_left), dtype=np.float64)
        logits = self.hmm_weight * hmm_logits + self.temporal_weight * temp_logits + self.bias
        return np.clip(sigmoid_np(logits), 1e-6, 1.0 - 1e-6)


def fit_calibrated_logistic_fusion(
    hmm_p_left: np.ndarray,
    temporal_p_left: np.ndarray,
    labels_left: np.ndarray,
    l2_reg: float = 0.05,
    steps: int = 400,
) -> CalibratedFusionWeights:
    """
    Fits calibrated logistic fusion weights `[w_hmm, w_temporal, bias]` on validation logits
    using L2-regularized Adam + L-BFGS-like fine-tuning in PyTorch.
    """
    hmm_t = torch.tensor(np.asarray(safe_logit(hmm_p_left), dtype=np.float32))
    tmp_t = torch.tensor(np.asarray(safe_logit(temporal_p_left), dtype=np.float32))
    y_t = torch.tensor(np.asarray(labels_left, dtype=np.float32))

    w_hmm = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))
    w_tmp = nn.Parameter(torch.tensor(0.8, dtype=torch.float32))
    bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float32))

    opt = torch.optim.Adam([w_hmm, w_tmp, bias], lr=0.04)
    bce = nn.BCEWithLogitsLoss()

    for _ in range(steps):
        opt.zero_grad()
        logits = w_hmm * hmm_t + w_tmp * tmp_t + bias
        reg = l2_reg * ((w_hmm - 0.5) ** 2 + (w_tmp - 0.5) ** 2 + bias**2)
        loss = bce(logits, y_t) + reg
        loss.backward()
        opt.step()
        with torch.no_grad():
            w_hmm.clamp_(0.05, 4.0)
            w_tmp.clamp_(0.05, 4.0)

    return CalibratedFusionWeights(
        hmm_weight=round(float(w_hmm.item()), 6),
        temporal_weight=round(float(w_tmp.item()), 6),
        bias=round(float(bias.item()), 6),
    )


def export_model_to_onnx(
    model: TemporalTrackerNet,
    onnx_path: str,
    seq_len: int = TEMPORAL_SEQ_LEN,
    feature_dim: int = TEMPORAL_FEATURE_DIM,
) -> None:
    model.eval()
    dummy = torch.zeros((1, seq_len, feature_dim), dtype=torch.float32)
    torch.onnx.export(
        model,
        (dummy,),
        onnx_path,
        input_names=["features"],
        output_names=["logit_left"],
        dynamic_axes={"features": {0: "batch"}, "logit_left": {0: "batch"}},
        opset_version=18,
        dynamo=False,
    )
