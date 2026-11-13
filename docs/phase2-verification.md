# Phase 2 Verification & Evaluation Report

## 1. Synthetic Multi-Profile Recorded Dataset Evaluation

> **Scope & Honesty Note:** No live physical 3-person webcam sessions were recorded in this headless Linux CI environment. All evaluation numbers below were measured by replaying **72 synthetic/simulated multi-profile recorded rounds** (`tests/vision/phase2Evaluation.test.ts`) through the complete Phase 2 pipeline (`HandIdentityTracker` -> `FeatureExtractor` -> `EventDetector` -> `BeliefFilter` -> `HedgeEnsemble` -> `makeFinalGuess`).

### Dataset Design (72 Recorded Rounds)
- **3 simulated player profiles** (`profile-a`, `profile-b`, `profile-c`, 24 rounds each) with differing hand-choice priors, fake-contact rates, and brief mid-hiding occlusions.
- **Transfer distribution:**
  - `0` transfers (no-contact and fake-contact rounds): **24 rounds**
  - `1` transfer (contact + brief partial opening + post-contact tightness shift): **24 rounds**
  - `2+` transfers (multi-contact sequential handovers): **24 rounds**

### Measured Accuracy (from `npm test -- tests/vision/phase2Evaluation.test.ts`)
| Transfer Count | Correct / Total | Accuracy |
|---|---:|---:|
| `0` transfers | 24 / 24 | **100.0%** |
| `1` transfer | 22 / 24 | **91.7%** |
| `2+` transfers | 24 / 24 | **100.0%** |
| **Overall** | **70 / 72** | **97.2%** |

## 2. How to Record and Evaluate Live 3-Person Sessions (Owner Guide)

1. Open the app on desktop or mobile Chrome/Safari and complete the **Calibration Wizard** (`جادوگر کالیبراسیون`).
2. Press **`D`** (or click **«دیباگ (D)»**) to open the Debug Overlay.
3. Click **«شروع ضبط (فقط اعداد)»** before playing a round — only numeric landmarks, quality scalars, and derived features are recorded (zero images/video).
4. Have **3 participants** (e.g. yourself + 2 other players) play at least **20 rounds each** (60+ total) spanning `0`, `1`, and `2+` hand transfers.
5. After each session, click **«توقف و دانلود JSON»** to save the numeric-only `RecordedSession` JSON files.
6. Replay any recorded JSON inside the app via **«بارگذاری فایل JSON برای بازپخش»**, or pass the parsed sessions through `replayRecordedSession` + `BeliefFilter` + `makeFinalGuess` in `tests/vision/` to measure real-camera accuracy split by `0`, `1`, and `2+` transfers.
