"""End-to-end training, player-grouped evaluation, ONNX export, parity verification, and ship gate."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .dataset import (
    RoundSample,
    generate_synthetic_dataset,
    load_exported_logs,
    split_by_player,
)
from .evaluate import evaluate_predictions, format_evaluation_report_md
from .features import (
    TEMPORAL_FEATURE_DIM,
    TEMPORAL_FEATURE_NAMES,
    TEMPORAL_SAMPLE_RATE_HZ,
    TEMPORAL_SEQ_LEN,
)
from .model import (
    TemporalTrackerNet,
    export_model_to_onnx,
    fit_calibrated_logistic_fusion,
    sigmoid_np,
)


def _predict_pytorch_probs(
    model: TemporalTrackerNet, samples: list[RoundSample]
) -> np.ndarray:
    model.eval()
    x = torch.tensor(np.stack([s.sequence for s in samples]), dtype=torch.float32)
    with torch.no_grad():
        logits = model(x).squeeze(-1).cpu().numpy()
    return sigmoid_np(logits)


def verify_pytorch_onnx_parity(
    model: TemporalTrackerNet,
    onnx_path: str | Path,
    sequences: np.ndarray,
    atol: float = 5e-5,
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        pt_logits = model(torch.tensor(sequences, dtype=torch.float32)).cpu().numpy()

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ort_logits = session.run(["logit_left"], {"features": sequences.astype(np.float32)})[0]
    max_diff = float(np.max(np.abs(pt_logits - ort_logits)))
    if max_diff > atol:
        raise AssertionError(
            f"PyTorch vs ONNX parity failed: max_abs_diff={max_diff:.3e} > atol={atol:.3e}"
        )
    return max_diff, pt_logits, ort_logits


def run_training_pipeline(
    *,
    logs_path: str | None = None,
    output_dir: str | Path = "../public/models",
    report_path: str | Path = "../docs/phase4-training-report.md",
    parity_fixture_path: str | Path | None = "../tests/unit/fixtures/temporal_parity.json",
    seed: int = 42,
    epochs: int = 45,
) -> dict[str, object]:
    torch.manual_seed(seed)
    np.random.seed(seed)

    if logs_path:
        samples = load_exported_logs(logs_path)
    else:
        samples = generate_synthetic_dataset(seed=seed, num_players=15, rounds_per_player=54)

    train_samples, val_samples, test_samples = split_by_player(samples, seed=seed)

    x_train = torch.tensor(np.stack([s.sequence for s in train_samples]), dtype=torch.float32)
    y_train = torch.tensor([s.label_left for s in train_samples], dtype=torch.float32).unsqueeze(-1)
    x_val = torch.tensor(np.stack([s.sequence for s in val_samples]), dtype=torch.float32)
    y_val = torch.tensor([s.label_left for s in val_samples], dtype=torch.float32).unsqueeze(-1)

    train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=32, shuffle=True)

    model = TemporalTrackerNet(feature_dim=TEMPORAL_FEATURE_DIM, hidden_dim=24)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for _epoch in range(epochs):
        model.train()
        for bx, by in train_loader:
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_logits = model(x_val)
            val_loss = float(criterion(val_logits, y_val).item())
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    model.eval()

    # Fit calibrated logistic fusion on validation split ONLY
    val_hmm_probs = np.asarray([s.hmm_p_left for s in val_samples], dtype=np.float64)
    val_temp_probs = _predict_pytorch_probs(model, val_samples)
    val_labels = np.asarray([s.label_left for s in val_samples], dtype=np.float64)
    fusion = fit_calibrated_logistic_fusion(val_hmm_probs, val_temp_probs, val_labels)

    # Export ONNX to temporary path first for parity & ship-gate validation
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_onnx = Path(tmpdir) / "temporal_tracker.onnx"
        export_model_to_onnx(model, str(tmp_onnx))
        onnx_proto = onnx.load(str(tmp_onnx))
        onnx.checker.check_model(onnx_proto)

        test_x = np.stack([s.sequence for s in test_samples]).astype(np.float32)
        parity_max_diff, _pt_logits, ort_logits = verify_pytorch_onnx_parity(
            model, tmp_onnx, test_x, atol=5e-5
        )
        test_temp_probs = sigmoid_np(ort_logits.squeeze(-1))
        test_hmm_probs = np.asarray([s.hmm_p_left for s in test_samples], dtype=np.float64)
        test_fused_probs = fusion.predict_proba(test_hmm_probs, test_temp_probs)

        hmm_metrics = evaluate_predictions(test_samples, test_hmm_probs)
        temporal_metrics = evaluate_predictions(test_samples, test_temp_probs)
        fused_metrics = evaluate_predictions(test_samples, test_fused_probs)

        ship_gate_passed = bool(fused_metrics.accuracy > hmm_metrics.accuracy)

        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        model_dest = out_dir / "temporal_tracker.onnx"
        config_dest = out_dir / "fusion_config.json"

        fusion_dict = {
            "hmmWeight": fusion.hmm_weight,
            "temporalWeight": fusion.temporal_weight,
            "bias": fusion.bias,
        }

        if ship_gate_passed:
            model_dest.write_bytes(tmp_onnx.read_bytes())

            # Optional parity fixture for TypeScript ONNX Runtime Web verification
            if parity_fixture_path:
                pf_path = Path(parity_fixture_path)
                pf_path.parent.mkdir(parents=True, exist_ok=True)
                n_fix = min(6, len(test_samples))
                fix_samples = test_samples[:n_fix]
                fix_x = np.stack([s.sequence for s in fix_samples]).astype(np.float32)
                _d, _pt, fix_ort_logits = verify_pytorch_onnx_parity(
                    model, model_dest, fix_x, atol=5e-5
                )
                fix_probs = sigmoid_np(fix_ort_logits.squeeze(-1))
                pf_payload = {
                    "modelVersion": "phase4-cnn-v1",
                    "seqLen": TEMPORAL_SEQ_LEN,
                    "featureDim": TEMPORAL_FEATURE_DIM,
                    "cases": [
                        {
                            "roundId": fix_samples[i].round_id,
                            "initialSide": fix_samples[i].initial_side,
                            "expectedLogitLeft": round(float(fix_ort_logits[i, 0]), 7),
                            "expectedProbLeft": round(float(fix_probs[i]), 7),
                            "features": fix_x[i].tolist(),
                        }
                        for i in range(n_fix)
                    ],
                }
                pf_path.write_text(json.dumps(pf_payload, indent=2), encoding="utf-8")

        config_payload = {
            "modelVersion": "phase4-cnn-v1",
            "shipGatePassed": ship_gate_passed,
            "seqLen": TEMPORAL_SEQ_LEN,
            "featureDim": TEMPORAL_FEATURE_DIM,
            "sampleRateHz": TEMPORAL_SAMPLE_RATE_HZ,
            "featureNames": list(TEMPORAL_FEATURE_NAMES),
            "fusionWeights": fusion_dict,
            "heldOutEvaluation": {
                "testRounds": fused_metrics.total,
                "hmmAccuracy": round(hmm_metrics.accuracy, 6),
                "hmmEce": round(hmm_metrics.ece, 6),
                "temporalAccuracy": round(temporal_metrics.accuracy, 6),
                "temporalEce": round(temporal_metrics.ece, 6),
                "fusedAccuracy": round(fused_metrics.accuracy, 6),
                "fusedEce": round(fused_metrics.ece, 6),
                "parityMaxAbsDiff": parity_max_diff,
            },
        }
        config_dest.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")

    dataset_summary = {
        "total_rounds": len(samples),
        "total_players": len({s.player_id for s in samples}),
        "train_rounds": len(train_samples),
        "train_players": len({s.player_id for s in train_samples}),
        "val_rounds": len(val_samples),
        "val_players": len({s.player_id for s in val_samples}),
        "test_rounds": len(test_samples),
        "test_players": len({s.player_id for s in test_samples}),
    }
    report_md = format_evaluation_report_md(
        dataset_summary=dataset_summary,
        hmm_metrics=hmm_metrics,
        temporal_metrics=temporal_metrics,
        fused_metrics=fused_metrics,
        fusion_weights=fusion_dict,
        parity_max_abs_diff=parity_max_diff,
        ship_gate_passed=ship_gate_passed,
    )
    report_dest = Path(report_path)
    report_dest.parent.mkdir(parents=True, exist_ok=True)
    report_dest.write_text(report_md, encoding="utf-8")

    return {
        "shipGatePassed": ship_gate_passed,
        "hmmAccuracy": hmm_metrics.accuracy,
        "hmmEce": hmm_metrics.ece,
        "temporalAccuracy": temporal_metrics.accuracy,
        "temporalEce": temporal_metrics.ece,
        "fusedAccuracy": fused_metrics.accuracy,
        "fusedEce": fused_metrics.ece,
        "parityMaxAbsDiff": parity_max_diff,
        "fusionWeights": fusion_dict,
        "byTransferCount": {
            k: {
                "count": v.count,
                "hmmAcc": hmm_metrics.by_transfer_count[k].accuracy,
                "temporalAcc": temporal_metrics.by_transfer_count[k].accuracy,
                "fusedAcc": v.accuracy,
            }
            for k, v in fused_metrics.by_transfer_count.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Gol ya Pooch Phase 4 Training Pipeline")
    parser.add_argument("--logs", type=str, default=None, help="Optional exported JSON round logs path")
    parser.add_argument("--output-dir", type=str, default="../public/models")
    parser.add_argument("--report-path", type=str, default="../docs/phase4-training-report.md")
    parser.add_argument(
        "--parity-fixture-path",
        type=str,
        default="../tests/unit/fixtures/temporal_parity.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=75)
    args = parser.parse_args()

    result = run_training_pipeline(
        logs_path=args.logs,
        output_dir=args.output_dir,
        report_path=args.report_path,
        parity_fixture_path=args.parity_fixture_path,
        seed=args.seed,
        epochs=args.epochs,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
