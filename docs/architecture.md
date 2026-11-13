# Gol Ya Pooch Architecture Notes (Phases 1–4)

## Design Goals

1. **Zero External Runtime Requests:** All assets required at runtime (`@mediapipe/tasks-vision` WASM files, `hand_landmarker.task`, and `Vazirmatn` `.woff2` fonts) live in `public/` and are copied verbatim into `dist/`. PixiJS scenes draw procedural vector art with zero external textures/CDNs. Vite builds with `base: './'` so static deployment works on both localhost and GitHub Pages subpaths.
2. **Privacy by Construction:** Camera frames are drawn onto an in-memory `HTMLCanvasElement` for typed-array `ImageData` color/lighting checks and passed to the local MediaPipe WASM runtime. `SessionRecorder` (`src/vision/recorder.ts`), `IndexedDbProfileStore` (`src/storage/profileStore.ts`), and `IndexedDbProgressionStore` (`src/storage/progressionStore.ts`) only store numeric landmarks, derived kinematics, game events, progression/stats scalars, and expert weights — **never images or video**.
3. **Resilient Hand Identity (`src/vision/handTracker.ts`):** MediaPipe's single-frame `handedness` classification frequently swaps when hands cross. `HandIdentityTracker` tracks `LEFT` and `RIGHT` wrists using velocity-predicted spatial positions (computed from raw wrist landmarks to avoid smoothing lag) and optimal Hungarian assignment (`hungarianAssign`), treating MediaPipe `handedness` purely as a weak tie-breaker (`0.001`).
4. **Explicit Game FSM (`src/game/fsm.ts`):** Game state transitions are pure functions `(GameContext, GameEvent) => GameContext`, making every phase timeout, void-round rule (including Level 3's `NO_HAND_CONTACT`), tab-hidden pause/resume, recoverable camera loss, and full `HUMAN_HIDES` <-> `COMPUTER_HIDES` turn alternation deterministic and directly testable with unit tests and `fast-check` property-based tests.

## Phase 2: Tracking, Behavioral Modeling, and Rule Engine

5. **Per-Frame Kinematic & Geometric Features (`src/vision/features.ts`):**
   - `FeatureExtractor` computes palm-size-normalized joint flexion angles (MCP, PIP, DIP) and total curl per finger, fist compactness (2D convex-hull area + mean fingertip-to-palm-center distance), sliding-window compactness deltas, wrist velocity/acceleration/jitter, 3D palm normal & depth proxies, inter-hand landmark proximity & contact duration, partial-opening events (brief curl dip followed by recovery), per-hand gol-sighting scores, and frame quality flags.
   - **Honest Physical Limitation:** Muscle tension itself is **not directly observable** at standard webcam resolution. Squeezing, relaxing, and hand tension are **approximated** through finger geometry (curl and fist compactness) and temporal dynamics (sliding-window compactness deltas and wrist jitter/tremor).

6. **Event Detector & Bayesian HMM Belief Filter (`src/tracking/eventDetector.ts`, `src/tracking/beliefFilter.ts`):**
   - `EventDetector` converts frame-by-frame `FrameFeatures` during `HIDING` into a structured timeline of discrete events (`CONTACT_START`, `CONTACT_END`, `PARTIAL_OPENING`, `GOL_SIGHTING`, `OCCLUSION_START`, `OCCLUSION_END`, `COMPACTNESS_SHIFT`).
   - `BeliefFilter` runs a 2-state discrete HMM forward pass over `{LEFT, RIGHT}` and exposes live belief solely inside the toggleable Debug Overlay (`D` key).

7. **Behavioral Experts & Hedge Ensemble (`src/ai/behavior/`, `src/ai/ensemble.ts`, `src/ai/guesser.ts`):**
   - Experts predict `P(LEFT)` from past round choices (`frequency`, `ngram`, `markov`, `habituation`), mixed via `HedgeEnsemble`, and fused with tracking belief in log-odds space by `Guesser`.

## Phase 3: Computer Hiding, PixiJS v8, Commit–Reveal, Rules & Levels 1–5

8. **PixiJS v8 Animated Hiding Scenes (`src/render/pixiHidingScene.ts`, `src/ai/hider.ts`):**
   - Procedural vector-art stylized closed fists, open palms, and glowing gol pearl supporting 2-hand (`classic-1v1`) and 4–6-hand (`traditional-team`) layouts across mobile portrait and desktop viewports.
   - Level-driven choreography generator (`generateHidingChoreography`) produces swaps, feints (`FEINT_SWAP`), uneven-tempo bursts/pauses, and multi-hand misdirection (`MISDIRECTION_CROSS`), collapsing to a calm static pause when `reducedMotion` is enabled.

9. **Cryptographic Commit–Reveal Fairness (`src/game/commitReveal.ts`):**
   - Before the human guesses, `createCommitEnvelope` generates a 16-byte random salt via `crypto.getRandomValues` and computes `SHA-256(choice + ":" + salt)` via Web Crypto `subtle.digest`.
   - A short fingerprint is displayed prior to guessing; after guessing, `verifyCommitReveal` recomputes the digest client-side, renders `✓ تأیید شد`, and provides a Persian "این چطور کار می‌کند؟" explanation.

10. **Computer Hiding Policy, Levels 1–5, and Declarative Rules (`src/ai/hider.ts`, `src/game/levels.ts`, `src/game/rules/`):**
    - `traditional-team.json` and `custom.json` extend the schema-validated declarative rule engine (`evaluateGuessCallStep`, `resolveRoundByRules`) with `پوچ` / `گل` multi-guess scoring (`+2`, `+0`, `+1`).
    - Levels 1–4 sample uniformly via unbiased `crypto.getRandomValues` rejection sampling. Level 5 (`استاد گل‌گیر`) runs a Best-of-3 series (`classic-1v1` and `traditional-team`) and pairs an online Hedge human-guesser predictor with a bounded-exploitation best response (`|p_i - 1/K| <= min(0.18, 0.45 / K)`) so the computer stays unpredictable.
    - Level progression and match/round statistics are stored strictly numerically in IndexedDB via `IndexedDbProgressionStore`.

## Phase 4: Consent-Gated Logging, Offline Training & Self-Hosted ONNX Inference

11. **Consent-Gated Numeric Round Logging (`src/storage/roundLogStore.ts`):**
    - `IndexedDbRoundLogStore` defaults to `consentEnabled: false` and refuses to persist any round until the player opts in via the Persian Consent screen or Settings.
    - `assertNumericOnlyPayload` recursively rejects any media keys (`image`, `video`, `canvas`, `pixels`, `rgba`, `blob`, `base64`), typed arrays, DOM media objects, or `data:image/` / `data:video/` URIs before IndexedDB storage and during JSON export/import.
    - Logged rounds record level, rule set, initial hand, 10 Hz resampled `[64 x 24]` feature sequence, raw numeric per-frame vectors, event timeline, HMM belief trace, final guess breakdown, verified ground truth, and timings.

12. **Offline Python 3.11+ Pipeline (`/training/`) & Ship Gate:**
    - `training/gol_training/` loads exported JSON round logs (or generates a deterministic multi-player dataset), splits strictly by player (`split_by_player` with zero leakage), trains a compact **Conv1D + GRU** temporal tracker (`TemporalTrackerNet`), calibrates logistic fusion weights (`[w_hmm, w_temporal, bias]`) on the validation split, and exports `temporal_tracker.onnx` (31 KB).
    - **Ship Gate:** `public/models/temporal_tracker.onnx` and `public/models/fusion_config.json` ship only if the fused model strictly beats the HMM filter-only baseline on the held-out player test split.

13. **Browser ONNX Runtime Web WASM Integration (`src/tracking/temporalModel.ts`, `src/ai/guesser.ts`):**
    - `onnxruntime-web@1.31.1` WASM binaries (`ort-wasm-simd-threaded.wasm`, `ort-wasm-simd-threaded.mjs`) are self-hosted under `public/ort/` and resolved via `document.baseURI` with zero external CDN requests.
    - `TemporalModelRunner` runs warmup on load, executes inference on the resampled 10 Hz hiding sequence (`< 1 ms` warm in Node/desktop Chromium), and fuses with the HMM belief filter via `fuseHmmAndTemporal`. Any missing, HTML-redirected, corrupt, or ship-gate-failed model automatically falls back to HMM-only without crashing.
