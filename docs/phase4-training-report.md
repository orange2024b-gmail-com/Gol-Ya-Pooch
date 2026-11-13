# Phase 4 Offline Training & Held-Out Player Evaluation Report

## 1. Dataset & Strictly Player-Grouped Split

- Total rounds: **810** across **15** players
- Train split: **486** rounds (9 players)
- Validation split: **162** rounds (3 players)
- Held-out Test split: **162** rounds (3 players)
- Player overlap across splits: **0** (verified)

## 2. Held-Out Player Test Comparison

| Model | Accuracy | ECE (10-bin) | Brier Score | Log-Loss |
|---|---:|---:|---:|---:|
| HMM Filter Baseline | 75.93% | 0.2022 | 0.2154 | 1.0637 |
| Temporal Conv1D+GRU (ONNX) | 89.51% | 0.0770 | 0.0806 | 0.2941 |
| **Calibrated Logistic Fusion (HMM + ONNX)** | **90.12%** | **0.0599** | **0.0772** | **0.2555** |

## 3. Confusion Breakdown by Transfer Count (Held-Out Test)

| Transfers | Rounds | HMM Baseline Acc | Temporal ONNX Acc | Fused (HMM + ONNX) Acc | Fused Confusion (TP_L / FP_L / TN_L / FN_L) |
|---|---:|---:|---:|---:|---|
| 0 | 54 | 79.63% | 98.15% | **98.15%** | `27 / 1 / 26 / 0` |
| 1 | 54 | 74.07% | 77.78% | **79.63%** | `24 / 8 / 19 / 3` |
| 2+ | 54 | 74.07% | 92.59% | **92.59%** | `23 / 4 / 27 / 0` |

## 4. Calibrated Fusion Weights, ONNX Parity & Ship Gate

- Learned fusion weights: `hmmWeight=0.0500`, `temporalWeight=0.7290`, `bias=-0.0610`
- PyTorch vs ONNX Runtime parity (`max |diff|`): `6.08e-06`
- **Ship Gate (`fused_acc > hmm_acc`)**: **PASSED — Model Shipped to public/models/**
