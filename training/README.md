# Gol ya Pooch — Offline Temporal Training Pipeline (`/training`)

Python 3.11+ offline training, player-grouped evaluation, calibrated logistic fusion, and ONNX export pipeline for Phase 4.

## Setup

```bash
cd training
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Tests

```bash
cd training
pytest -v
```

## Train, Evaluate, and Run Ship Gate

Run with the deterministic multi-player synthetic dataset (or pass `--logs path/to/exported-logs.json`):

```bash
cd training
python -m gol_training.train \
  --output-dir ../public/models \
  --report-path ../docs/phase4-training-report.md \
  --parity-fixture-path ../tests/unit/fixtures/temporal_parity.json
```

### Ship Gate Policy

The pipeline evaluates three predictors on the **strictly player-held-out test split** (zero player overlap with train or validation splits):
1. **HMM Filter Baseline** (`BeliefFilter` forward pass)
2. **Temporal 1D-CNN Model** (exported to ONNX)
3. **Calibrated Logistic Fusion** (HMM + Temporal ONNX logits fitted on validation split)

`temporal_tracker.onnx` and `fusion_config.json` are shipped to `public/models/` **only if** the fused model strictly beats the HMM filter-only baseline on the held-out player test accuracy.
