"""Tests for ONNX export parity, ECE calibration, transfer-count evaluation, and ship gate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import torch

from gol_training.dataset import generate_synthetic_dataset
from gol_training.evaluate import compute_ece, evaluate_predictions, format_evaluation_report_md
from gol_training.features import TEMPORAL_FEATURE_DIM, TEMPORAL_SEQ_LEN
from gol_training.model import TemporalTrackerNet, export_model_to_onnx
from gol_training.train import run_training_pipeline, verify_pytorch_onnx_parity


def test_onnx_export_and_pytorch_ort_parity(tmp_path: Path) -> None:
    torch.manual_seed(123)
    model = TemporalTrackerNet(feature_dim=TEMPORAL_FEATURE_DIM, hidden_dim=24)
    onnx_file = tmp_path / "model.onnx"
    export_model_to_onnx(model, str(onnx_file))

    proto = onnx.load(str(onnx_file))
    onnx.checker.check_model(proto)

    rng = np.random.default_rng(123)
    x = rng.normal(0.0, 0.5, size=(8, TEMPORAL_SEQ_LEN, TEMPORAL_FEATURE_DIM)).astype(np.float32)
    max_diff, _, _ = verify_pytorch_onnx_parity(model, onnx_file, x, atol=5e-5)
    assert max_diff <= 5e-5


def test_ece_and_transfer_count_evaluation_report() -> None:
    samples = generate_synthetic_dataset(seed=99, num_players=4, rounds_per_player=18)
    probs = np.asarray([s.hmm_p_left for s in samples], dtype=np.float64)
    metrics = evaluate_predictions(samples, probs)

    assert metrics.total == len(samples)
    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.ece <= 1.0
    assert set(metrics.by_transfer_count.keys()) == {"0", "1", "2+"}
    assert sum(b.count for b in metrics.by_transfer_count.values()) == len(samples)

    # Perfect calibration should yield near-zero ECE
    perfect_probs = np.asarray([1.0 if s.label_left == 1 else 0.0 for s in samples], dtype=np.float64)
    perfect_labels = np.asarray([s.label_left for s in samples], dtype=np.float64)
    assert compute_ece(perfect_probs, perfect_labels) < 1e-6

    md = format_evaluation_report_md(
        dataset_summary={
            "total_rounds": len(samples),
            "total_players": 4,
            "train_rounds": 36,
            "train_players": 2,
            "val_rounds": 18,
            "val_players": 1,
            "test_rounds": 18,
            "test_players": 1,
        },
        hmm_metrics=metrics,
        temporal_metrics=metrics,
        fused_metrics=metrics,
        fusion_weights={"hmmWeight": 1.0, "temporalWeight": 1.0, "bias": 0.0},
        parity_max_abs_diff=1e-6,
        ship_gate_passed=False,
    )
    assert "ECE (10-bin)" in md
    assert "0" in md and "1" in md and "2+" in md


def test_end_to_end_training_pipeline_and_ship_gate(tmp_path: Path) -> None:
    out_dir = tmp_path / "models"
    report_path = tmp_path / "report.md"
    parity_path = tmp_path / "parity.json"

    res = run_training_pipeline(
        output_dir=out_dir,
        report_path=report_path,
        parity_fixture_path=parity_path,
        seed=42,
        epochs=30,
    )

    assert report_path.exists()
    assert (out_dir / "fusion_config.json").exists()
    assert float(res["parityMaxAbsDiff"]) <= 5e-5  # type: ignore[arg-type]
    if res["shipGatePassed"]:
        assert (out_dir / "temporal_tracker.onnx").exists()
        assert parity_path.exists()
    else:
        assert not (out_dir / "temporal_tracker.onnx").exists()
