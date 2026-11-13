# Phase 4 Verification Report

## 1. Automated Verification Summary

| Check | Command | Measured Result |
|---|---|---|
| Linter | `npm run lint` | Passed (0 errors, 0 warnings) |
| Typecheck | `npm run typecheck` | Passed (`tsc --noEmit`, strict mode) |
| TypeScript Unit, Property, Vision & Sim Tests | `npm run test:coverage` | **20 / 20 files passed**, **76 / 76 tests passed** |
| Python Training & Parity Tests | `cd training && ./.venv/bin/pytest -v` | **6 / 6 tests passed** |
| Production Build | `npm run build` | Passed (`vite build`, 788 modules, 604 ms, 0 errors) |
| End-to-End Tests (Chromium) | `npm run test:e2e` | **7 / 7 Playwright specs passed**, **0 external runtime requests**, **0 page errors** |

---

## 2. Offline Training, Player-Grouped Evaluation & Ship Gate

Executed via `python -m gol_training.train --output-dir ../public/models --report-path ../docs/phase4-training-report.md --parity-fixture-path ../tests/unit/fixtures/temporal_parity.json`:

- **Dataset:** 810 rounds across 15 synthetic multi-style players (54 rounds/player spanning 0, 1, 2, and 3 transfers, fake contacts, stealthy dips, and brief occlusions).
- **Strictly Player-Grouped Split:** 9 train players (486 rounds), 3 validation players (162 rounds), 3 held-out test players (162 rounds) — **0 player overlap** verified in `training/tests/test_split_and_logs.py`.
- **Model:** Compact Conv1D + GRU (`TemporalTrackerNet`, exported to `public/models/temporal_tracker.onnx`, **31 KB**) taking fixed-rate 10 Hz resampled/padded/truncated sequences (`[1, 64, 24]`).
- **Calibrated Logistic Fusion (fitted on validation split):** `hmmWeight = 0.0500`, `temporalWeight = 0.7290`, `bias = -0.0610`.

| Model (Held-Out Test Players, n = 162) | Overall Acc | 0 Transfers | 1 Transfer | 2+ Transfers | ECE (10-bin) | Log-Loss |
|---|---:|---:|---:|---:|---:|---:|
| HMM Filter-Only Baseline | 75.93% | 79.63% | 74.07% | 74.07% | 0.2022 | 0.9277 |
| Temporal Conv1D+GRU (ONNX) | 89.51% | 98.15% | 77.78% | 92.59% | 0.0770 | 0.2528 |
| **Calibrated Fusion (HMM + ONNX)** | **90.12%** | **98.15%** | **79.63%** | **92.59%** | **0.0599** | **0.2454** |

- **PyTorch vs ONNX Runtime Parity:** `max |diff| = 6.08e-6` (`<= 5e-5`).
- **Python ORT vs JS `onnxruntime-web` Parity (`tests/unit/temporalModel.test.ts`):** `max |diff| <= 1e-4` across all 6 test fixture sequences.
- **Ship Gate (`fused_acc > hmm_acc`):** **PASSED** (+14.20% accuracy improvement, ECE reduced from 0.2022 to 0.0599) — `public/models/temporal_tracker.onnx` and `public/models/fusion_config.json` shipped.

---

## 3. Inference Latency & Self-Hosting Verification

- **Self-Hosted Runtime Assets:** `onnxruntime-web@1.31.1` WASM files (`ort-wasm-simd-threaded.wasm`, `ort-wasm-simd-threaded.mjs`) live under `public/ort/` and are resolved locally via `document.baseURI` with `numThreads = 1`. Zero external runtime requests verified in Playwright (`tests/e2e/phase4.spec.ts`).
- **Measured Browser Latency (Desktop Chromium, `onnxruntime-web` WASM backend, 20 runs):**
  - `min = 0.30 ms`, **`mean = 0.35 ms`**, `max = 0.80 ms` (well below the `< 10 ms` budget).
- **Measured Node ORT Latency (`tests/unit/temporalModel.test.ts`, 6 fixture runs):** `0.32–0.57 ms` warm.
- **Mobile Phone Latency Status:** Not directly benchmarked on a physical Android/iOS device in this cloud VM environment. Because the exported Conv1D+GRU ONNX artifact is only **31 KB** (`hidden_dim = 24`, single-threaded WASM SIMD) and runs in `0.35 ms` mean on desktop Chromium, it provides ~28x latency headroom below the `< 10 ms` target; final on-device phone verification should be recorded during Phase 5 hardware testing.
- **Fallback Resilience:** Verified in both Vitest (`tests/unit/temporalModel.test.ts`) and Playwright (`tests/e2e/phase4.spec.ts` HTTP 404 route intercept) that missing, HTML-redirected, corrupt, or ship-gate-failed models fall back seamlessly to the HMM belief filter + behavioral ensemble with 0 crashes or page errors.

---

## 4. Consent-Gated Logging & Numeric-Only Privacy

- `IndexedDbRoundLogStore` (`src/storage/roundLogStore.ts`) defaults to `consentEnabled: false` and refuses persistence until explicitly opted in.
- `assertNumericOnlyPayload` enforces at runtime (during logging, JSON export, and JSON import) that no image, video, canvas, pixel array, `Blob`, or `data:image/` / `data:video/` string can ever be stored or exported.
