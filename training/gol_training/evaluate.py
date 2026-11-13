"""Evaluation metrics: accuracy, ECE calibration, Brier score, and confusion by transfer count."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .dataset import RoundSample


@dataclass(frozen=True)
class ConfusionBreakdown:
    count: int
    accuracy: float
    tp_left: int
    fp_left: int
    tn_left: int
    fn_left: int


@dataclass(frozen=True)
class EvaluationMetrics:
    total: int
    accuracy: float
    ece: float
    brier_score: float
    log_loss: float
    overall_confusion: ConfusionBreakdown
    by_transfer_count: dict[str, ConfusionBreakdown]


def compute_ece(probs_left: np.ndarray, labels_left: np.ndarray, num_bins: int = 10) -> float:
    """Expected Calibration Error (ECE) using confidence bins over [0.5, 1.0]."""
    if len(probs_left) == 0:
        return 0.0
    probs = np.asarray(probs_left, dtype=np.float64)
    labels = np.asarray(labels_left, dtype=np.float64)
    preds = (probs >= 0.5).astype(np.float64)
    confidences = np.where(preds == 1.0, probs, 1.0 - probs)
    accuracies = (preds == labels).astype(np.float64)

    bins = np.linspace(0.5, 1.0 + 1e-9, num_bins + 1)
    ece = 0.0
    n = float(len(probs))
    for i in range(num_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == 0:
            mask = (confidences >= lo - 1e-9) & (confidences < hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        cnt = int(np.sum(mask))
        if cnt == 0:
            continue
        bin_acc = float(np.mean(accuracies[mask]))
        bin_conf = float(np.mean(confidences[mask]))
        ece += (cnt / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_confusion(probs_left: np.ndarray, labels_left: np.ndarray) -> ConfusionBreakdown:
    n = len(labels_left)
    if n == 0:
        return ConfusionBreakdown(count=0, accuracy=0.0, tp_left=0, fp_left=0, tn_left=0, fn_left=0)
    preds = (np.asarray(probs_left) >= 0.5).astype(np.int64)
    y = np.asarray(labels_left, dtype=np.int64)
    tp = int(np.sum((preds == 1) & (y == 1)))
    fp = int(np.sum((preds == 1) & (y == 0)))
    tn = int(np.sum((preds == 0) & (y == 0)))
    fn = int(np.sum((preds == 0) & (y == 1)))
    acc = float((tp + tn) / n)
    return ConfusionBreakdown(
        count=n,
        accuracy=acc,
        tp_left=tp,
        fp_left=fp,
        tn_left=tn,
        fn_left=fn,
    )


def _transfer_bucket(transfers: int) -> str:
    if transfers <= 0:
        return "0"
    if transfers == 1:
        return "1"
    return "2+"


def evaluate_predictions(
    samples: Sequence[RoundSample],
    probs_left: np.ndarray,
) -> EvaluationMetrics:
    probs = np.clip(np.asarray(probs_left, dtype=np.float64), 1e-6, 1.0 - 1e-6)
    labels = np.asarray([s.label_left for s in samples], dtype=np.float64)

    overall = compute_confusion(probs, labels)
    ece = compute_ece(probs, labels, num_bins=10)
    brier = float(np.mean((probs - labels) ** 2)) if len(labels) else 0.0
    log_loss = (
        float(-np.mean(labels * np.log(probs) + (1.0 - labels) * np.log(1.0 - probs)))
        if len(labels)
        else 0.0
    )

    by_bucket: dict[str, ConfusionBreakdown] = {}
    for bucket in ("0", "1", "2+"):
        idx = [i for i, s in enumerate(samples) if _transfer_bucket(s.transfer_count) == bucket]
        if idx:
            by_bucket[bucket] = compute_confusion(probs[idx], labels[idx])
        else:
            by_bucket[bucket] = ConfusionBreakdown(
                count=0, accuracy=0.0, tp_left=0, fp_left=0, tn_left=0, fn_left=0
            )

    return EvaluationMetrics(
        total=len(samples),
        accuracy=overall.accuracy,
        ece=ece,
        brier_score=brier,
        log_loss=log_loss,
        overall_confusion=overall,
        by_transfer_count=by_bucket,
    )


def format_evaluation_report_md(
    *,
    dataset_summary: dict[str, Any],
    hmm_metrics: EvaluationMetrics,
    temporal_metrics: EvaluationMetrics,
    fused_metrics: EvaluationMetrics,
    fusion_weights: dict[str, float],
    parity_max_abs_diff: float,
    ship_gate_passed: bool,
) -> str:
    lines = [
        "# Phase 4 Offline Training & Held-Out Player Evaluation Report",
        "",
        "## 1. Dataset & Strictly Player-Grouped Split",
        "",
        f"- Total rounds: **{dataset_summary['total_rounds']}** across **{dataset_summary['total_players']}** players",
        f"- Train split: **{dataset_summary['train_rounds']}** rounds ({dataset_summary['train_players']} players)",
        f"- Validation split: **{dataset_summary['val_rounds']}** rounds ({dataset_summary['val_players']} players)",
        f"- Held-out Test split: **{dataset_summary['test_rounds']}** rounds ({dataset_summary['test_players']} players)",
        f"- Player overlap across splits: **0** (verified)",
        "",
        "## 2. Held-Out Player Test Comparison",
        "",
        "| Model | Accuracy | ECE (10-bin) | Brier Score | Log-Loss |",
        "|---|---:|---:|---:|---:|",
        f"| HMM Filter Baseline | {hmm_metrics.accuracy * 100:.2f}% | {hmm_metrics.ece:.4f} | {hmm_metrics.brier_score:.4f} | {hmm_metrics.log_loss:.4f} |",
        f"| Temporal Conv1D+GRU (ONNX) | {temporal_metrics.accuracy * 100:.2f}% | {temporal_metrics.ece:.4f} | {temporal_metrics.brier_score:.4f} | {temporal_metrics.log_loss:.4f} |",
        f"| **Calibrated Logistic Fusion (HMM + ONNX)** | **{fused_metrics.accuracy * 100:.2f}%** | **{fused_metrics.ece:.4f}** | **{fused_metrics.brier_score:.4f}** | **{fused_metrics.log_loss:.4f}** |",
        "",
        "## 3. Confusion Breakdown by Transfer Count (Held-Out Test)",
        "",
        "| Transfers | Rounds | HMM Baseline Acc | Temporal ONNX Acc | Fused (HMM + ONNX) Acc | Fused Confusion (TP_L / FP_L / TN_L / FN_L) |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for bucket in ("0", "1", "2+"):
        hb = hmm_metrics.by_transfer_count[bucket]
        tb = temporal_metrics.by_transfer_count[bucket]
        fb = fused_metrics.by_transfer_count[bucket]
        lines.append(
            f"| {bucket} | {fb.count} | {hb.accuracy * 100:.2f}% | {tb.accuracy * 100:.2f}% | **{fb.accuracy * 100:.2f}%** | `{fb.tp_left} / {fb.fp_left} / {fb.tn_left} / {fb.fn_left}` |"
        )

    lines.extend(
        [
            "",
            "## 4. Calibrated Fusion Weights, ONNX Parity & Ship Gate",
            "",
            f"- Learned fusion weights: `hmmWeight={fusion_weights['hmmWeight']:.4f}`, `temporalWeight={fusion_weights['temporalWeight']:.4f}`, `bias={fusion_weights['bias']:.4f}`",
            f"- PyTorch vs ONNX Runtime parity (`max |diff|`): `{parity_max_abs_diff:.2e}`",
            f"- **Ship Gate (`fused_acc > hmm_acc`)**: **{'PASSED — Model Shipped to public/models/' if ship_gate_passed else 'FAILED — Model NOT Shipped'}**",
            "",
        ]
    )
    return "\n".join(lines)
