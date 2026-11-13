# Phase 5 Final Test Campaign & Verification Report

> **Reporting Integrity Policy:** Every number below comes directly from actual commands executed in this environment (or is explicitly attributed to Phase 4 where Python PyTorch/pytest ran). Physical mobile phone (Android/iOS) hardware benchmarks are **not** fabricated and remain marked as pending physical device verification in `docs/manual-test.md`.

---

## 1. Automated Verification Summary

| Suite | Command | Measured Result |
|---|---|---|
| ESLint | `npm run lint` | **Passed** (0 errors, 0 warnings) |
| TypeScript Strict Check | `npm run typecheck` | **Passed** (`tsc --noEmit`, 0 errors) |
| Vitest (Unit + Property + Vision + Sim) | `npm run test:coverage` | **23 / 23 test files passed**, **86 / 86 tests passed** |
| Production Bundle | `npm run build` | **Passed** (`vite build`, 792 modules, 0 errors) |
| Playwright E2E (Chromium VM) | `npm run test:e2e` | **10 / 10 specs passed**, **0 external runtime requests**, **0 page errors** |
| Python Training & Parity Tests | `training/.venv/bin/pytest -v` | *Not re-run in Phase 5 VM (`training/.venv` / PyTorch not installed in this VM snapshot).* `training/` code and `public/models/` artifacts are unmodified since Phase 4 where **6 / 6 pytest tests passed**. |

---

## 2. Unit & Property-Based Test Coverage (`npm run test:coverage`)

Total across all `src/**/*.ts` (excluding `src/main.ts` and DOM/WebGL/camera shells `appController.ts`, `cameraManager.ts`, `pixiHidingScene.ts`, `canvasOverlay.ts` which are covered by Playwright E2E):

- **Core Logic Modules Coverage:**
  - `src/ai/behavior/`: **100%** statements / **100%** lines
  - `src/ai/` (`ensemble.ts`, `guesser.ts`, `hider.ts`): **95.30%** statements / **95.65%** lines
  - `src/game/rules/`: **95.00%** statements / **95.00%** lines
  - `src/game/` (`commitReveal.ts`, `fsm.ts`, `levels.ts`): **91.74%** statements / **92.52%** lines
  - `src/tracking/` (`beliefFilter.ts`, `eventDetector.ts`, `temporalFeatures.ts`, `temporalModel.ts`): **89.29%** statements / **91.25%** lines
  - `src/vision/`: **87.61%** statements / **88.68%** lines
  - `src/audio/soundEngine.ts`: **86.20%** statements / **87.34%** lines
  - `src/llm/commentaryClient.ts`: **79.46%** statements / **80.00%** lines
  - `src/app/benchmark.ts`: **77.33%** statements / **75.00%** lines
  - `src/app/i18n.ts`: **100%** statements / **100%** lines
- **FSM Property-Based Invariants (`tests/unit/fsm.property.test.ts`):** `300` randomized sequences of up to `140` events (`42,000` max transitions across all `19` FSM states including `PAUSED`, `CAMERA_ERROR`, `ROUND_VOID`, `BENCHMARK`) — 0 undefined states, 0 score conservation violations, 0 turn-ownership violations.

---

## 3. Vision Regression & Tracking Accuracy (`tests/vision/`)

- **Hand-Crossing Identity Stability (`tests/vision/handCrossing.test.ts`):** `20 / 20` recorded crossing sessions (including double-crossings, vertical offsets, brief occlusions, and deliberate MediaPipe handedness label flips) -> **0 left/right identity swaps**.
- **Pinch Confirmation & Reveal Verification (`tests/vision/pinchConfirmation.test.ts`):** `40 / 40` standard lighting attempts confirmed (`100.0%`, exceeding `>= 95%` target), plus lighting-shift detection and `NEITHER` / `BOTH` / `HANDS_NOT_OPEN` reveal edge cases.
- **Phase 2 72-Round Synthetic Multi-Profile Replay Evaluation (`tests/vision/phase2Evaluation.test.ts`):**
  - `0` transfers: **24 / 24 (100.0%)**
  - `1` transfer: **22 / 24 (91.7%)**
  - `2+` transfers: **24 / 24 (100.0%)**
  - Overall: **70 / 72 (97.2%)**
- **Phase 4 Held-Out Player Evaluation (`docs/phase4-training-report.md`, unmodified model):**
  - `810` rounds across `15` players (9 train / 3 val / 3 held-out test, 0 player overlap).
  - Held-out test accuracy: **75.93%** HMM baseline vs **90.12%** Calibrated Fusion (HMM + ONNX Conv1D+GRU), ECE reduced from `0.2022` to `0.0599`.
  - PyTorch vs Python ORT parity: `max |diff| = 6.08e-6`; Python ORT vs JS `onnxruntime-web` parity (`tests/unit/temporalModel.test.ts`): `max |diff| <= 1e-4` across 6 test sequences.

---

## 4. Bot Simulations Across Levels 1–5 (`tests/sim/hiderSimulation.test.ts`)

### 4.1 Chi-Square Uniformity (10,000 Rounds per Configuration, Levels 1–4)
All Levels 1–4 (`K = 2, 4, 6`) passed strict chi-square uniformity thresholds (`alpha = 0.01` deterministic and `alpha = 0.0001` live Web Crypto) over `10,000` rounds each.

### 4.2 5-Bot Strategy Suite Across Levels 1–5 (2,000 Rounds per Pair)
Measured win rates from `WRITE_SIM_REPORT=1 npm test -- tests/sim/hiderSimulation.test.ts` (`docs/sim-report.md`):

| Level | Hands (K) | Bot Strategy | Rounds | Computer Win Rate | Bot Win Rate | Max Deviation from 1/K |
|---|---:|---|---:|---:|---:|---:|
| Level 1 (Beginner) | 2 | `random` | 2,000 | 51.70% | 48.30% | 0.0000 |
| Level 1 (Beginner) | 2 | `alternating` | 2,000 | 48.10% | 51.90% | 0.0000 |
| Level 1 (Beginner) | 2 | `sticky` | 2,000 | 50.60% | 49.40% | 0.0000 |
| Level 1 (Beginner) | 2 | `wsls` | 2,000 | 48.75% | 51.25% | 0.0000 |
| Level 1 (Beginner) | 2 | `adversarial` | 2,000 | 48.80% | 51.20% | 0.0000 |
| Level 2 (Amateur) | 2 | `random` | 2,000 | 49.45% | 50.55% | 0.0000 |
| Level 2 (Amateur) | 2 | `alternating` | 2,000 | 50.50% | 49.50% | 0.0000 |
| Level 2 (Amateur) | 2 | `sticky` | 2,000 | 50.35% | 49.65% | 0.0000 |
| Level 2 (Amateur) | 2 | `wsls` | 2,000 | 50.60% | 49.40% | 0.0000 |
| Level 2 (Amateur) | 2 | `adversarial` | 2,000 | 49.35% | 50.65% | 0.0000 |
| Level 3 (Pro) | 2 | `random` | 2,000 | 50.90% | 49.10% | 0.0000 |
| Level 3 (Pro) | 2 | `alternating` | 2,000 | 50.25% | 49.75% | 0.0000 |
| Level 3 (Pro) | 2 | `sticky` | 2,000 | 51.00% | 49.00% | 0.0000 |
| Level 3 (Pro) | 2 | `wsls` | 2,000 | 50.25% | 49.75% | 0.0000 |
| Level 3 (Pro) | 2 | `adversarial` | 2,000 | 49.10% | 50.90% | 0.0000 |
| Level 4 (Team) | 4 | `random` | 2,000 | 75.25% | 24.75% | 0.0000 |
| Level 4 (Team) | 4 | `alternating` | 2,000 | 74.65% | 25.35% | 0.0000 |
| Level 4 (Team) | 4 | `sticky` | 2,000 | 74.85% | 25.15% | 0.0000 |
| Level 4 (Team) | 4 | `wsls` | 2,000 | 74.85% | 25.15% | 0.0000 |
| Level 4 (Team) | 4 | `adversarial` | 2,000 | 74.65% | 25.35% | 0.0000 |
| Level 5 (Master Gol-Gir) | 2 | `random` | 2,000 | 49.80% | 50.20% | 0.1800 |
| Level 5 (Master Gol-Gir) | 2 | `alternating` | 2,000 | **65.00%** | 35.00% | 0.1800 |
| Level 5 (Master Gol-Gir) | 2 | `sticky` | 2,000 | **64.20%** | 35.80% | 0.1800 |
| Level 5 (Master Gol-Gir) | 2 | `wsls` | 2,000 | **64.10%** | 35.90% | 0.1800 |
| Level 5 (Master Gol-Gir) | 2 | `adversarial` | 2,000 | **57.90%** | 42.10% | 0.1800 |
| Level 5 (Master Gol-Gir) | 4 | `random` | 2,000 | 74.90% | 25.10% | 0.1125 |
| Level 5 (Master Gol-Gir) | 4 | `alternating` | 2,000 | **84.40%** | 15.60% | 0.1125 |
| Level 5 (Master Gol-Gir) | 4 | `sticky` | 2,000 | **84.75%** | 15.25% | 0.1125 |
| Level 5 (Master Gol-Gir) | 4 | `wsls` | 2,000 | **85.75%** | 14.25% | 0.1125 |
| Level 5 (Master Gol-Gir) | 4 | `adversarial` | 2,000 | **78.65%** | 21.35% | 0.1125 |

---

## 5. Browser E2E, Network Audit & Performance Benchmark

### 5.1 Playwright End-to-End Campaign (`npm run test:e2e`)
- **10 / 10 specs passed** across `phase1.spec.ts` (2), `phase2.spec.ts` (1), `phase3.spec.ts` (2), `phase4.spec.ts` (2), and `phase5.spec.ts` (3).
- **Network Zero-External Audit:** `externalRequests = []` across all specs; self-hosted `/favicon.svg`, `/fonts/vazirmatn.css`, `/mediapipe/*`, `/ort/ort-wasm-simd-threaded.wasm`, and `/models/temporal_tracker.onnx` all served from same origin with HTTP 200.
- **Privacy & Security Audit:**
  - Consent defaults OFF; JSON exports contain zero image/video fields.
  - Optional LLM commentary defaults OFF; when enabled against an unreachable/503 endpoint, silently falls back to Persian template commentary with 0 errors.
  - Mobile portrait (`390x844`) verified with `scrollWidth <= innerWidth` (`overflow = false`) on Splash, Menu, and PixiJS game scenes.

### 5.2 Measured Performance Benchmark (Cloud Linux Chromium VM, `?benchmark=1`)
Captured during `tests/e2e/phase5.spec.ts` on headless Chromium (`cores=4`, `ort=wasm (phase4-cnn-v1)`):

| Metric | Measured Result (Headless Chromium VM) | Budget / Target | Status |
|---|---|---|---|
| Browser RAF Frame Loop | **60.0 FPS** (`16.67 ms` mean, 24 frames; `38.9–60.0 FPS` across runs) | `>= 30 FPS` desktop | **Pass (VM)** |
| Feature Extraction (`FeatureExtractor`) | **`0.034 ms`** mean (`0.100 ms` p95, 172 frames) | `<< 33.3 ms` | **Pass (VM)** |
| HMM Belief Filter (`BeliefFilter`) | **`0.004 ms`** mean (`< 0.01 ms` p95, 201 steps) | `<< 33.3 ms` | **Pass (VM)** |
| ONNX Runtime Web (`temporal_tracker.onnx`) | **`0.53 ms`** mean (`min 0.30 ms`, `p95 1.50 ms`, 20 runs) | `< 10 ms` | **Pass (VM)** |
| Combined Vision-Logic Overhead / Frame | **`0.06 ms`** (excluding browser MediaPipe frame time) | `< 33.3 ms` (30 FPS) / `< 41.67 ms` (24 FPS) | **Pass (VM)** |

> **Pending Physical Device Measurements:** Physical Android Chrome, iOS Safari, Desktop Firefox, and Desktop Safari hardware FPS/MediaPipe-camera measurements cannot be captured in a headless cloud Linux VM. Run `<deployed-url>?benchmark=1` and follow `docs/manual-test.md` on each target physical device to record on-device figures.
