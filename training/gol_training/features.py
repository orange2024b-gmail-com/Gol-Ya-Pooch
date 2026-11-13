"""Canonical feature schema and fixed-rate resampling aligned with src/tracking/temporalFeatures.ts."""

from __future__ import annotations

from typing import Sequence

import numpy as np

TEMPORAL_SEQ_LEN = 64
TEMPORAL_SAMPLE_RATE_HZ = 10
TEMPORAL_FEATURE_NAMES: tuple[str, ...] = (
    "initialSideSign",
    "leftDetected",
    "rightDetected",
    "leftCurl",
    "rightCurl",
    "leftCompactness",
    "rightCompactness",
    "leftCompactnessDelta",
    "rightCompactnessDelta",
    "leftFingertipDistNorm",
    "rightFingertipDistNorm",
    "leftWristSpeed",
    "rightWristSpeed",
    "leftJitter",
    "rightJitter",
    "leftPartialOpen",
    "rightPartialOpen",
    "leftGolSighting",
    "rightGolSighting",
    "inContact",
    "interHandDistNorm",
    "contactDurationNorm",
    "lowQuality",
    "progressNorm",
)
TEMPORAL_FEATURE_DIM = len(TEMPORAL_FEATURE_NAMES)
MAX_EVENT_INDICES = frozenset({15, 16, 17, 18, 19, 22})


def resample_raw_feature_entries(
    raw_entries: Sequence[tuple[float, Sequence[float]]],
    seq_len: int = TEMPORAL_SEQ_LEN,
    sample_rate_hz: float = TEMPORAL_SAMPLE_RATE_HZ,
) -> np.ndarray:
    """
    Resample raw `(timestamp_ms, feature_vec[24])` entries to fixed `sample_rate_hz`,
    zero-padding up to `seq_len` or truncating to the most recent `seq_len` steps.
    Matches `resampleHidingFeatures` in `src/tracking/temporalFeatures.ts`.
    """
    out = np.zeros((seq_len, TEMPORAL_FEATURE_DIM), dtype=np.float32)
    if not raw_entries:
        return out

    sorted_entries = sorted(raw_entries, key=lambda item: float(item[0]))
    t_start = float(sorted_entries[0][0])
    t_end = float(sorted_entries[-1][0])
    duration_ms = max(0.0, t_end - t_start)
    step_ms = 1000.0 / float(sample_rate_hz)

    total_bins = max(1, int(duration_ms // step_ms) + 1)
    bin_offset = total_bins - seq_len if total_bins > seq_len else 0
    out_steps = min(seq_len, total_bins)
    effective_start_ms = t_start + bin_offset * step_ms

    vecs = np.asarray([list(item[1]) for item in sorted_entries], dtype=np.float32)
    if vecs.shape[1] != TEMPORAL_FEATURE_DIM:
        raise ValueError(
            f"Expected {TEMPORAL_FEATURE_DIM} features per frame, got {vecs.shape[1]}"
        )
    timestamps = np.asarray([float(item[0]) for item in sorted_entries], dtype=np.float64)

    for b in range(out_steps):
        bin_start = effective_start_ms + b * step_ms
        bin_end = bin_start + step_ms
        bin_mid = bin_start + step_ms * 0.5

        if b == out_steps - 1:
            mask = (timestamps >= bin_start - 1e-6) & (timestamps <= bin_end + 1e-6)
        else:
            mask = (timestamps >= bin_start - 1e-6) & (timestamps < bin_end - 1e-6)

        indices = np.flatnonzero(mask)
        if len(indices) == 0:
            nearest = int(np.argmin(np.abs(timestamps - bin_mid)))
            indices = np.asarray([nearest], dtype=np.int64)

        bin_vecs = vecs[indices]
        bin_progress = 0.0 if out_steps <= 1 else float(b) / float(out_steps - 1)

        for f_idx in range(TEMPORAL_FEATURE_DIM):
            if f_idx == TEMPORAL_FEATURE_DIM - 1:
                out[b, f_idx] = bin_progress
            elif f_idx in MAX_EVENT_INDICES:
                out[b, f_idx] = float(np.max(bin_vecs[:, f_idx]))
            else:
                out[b, f_idx] = float(np.mean(bin_vecs[:, f_idx]))

    return out
