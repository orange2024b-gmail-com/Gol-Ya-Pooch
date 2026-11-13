# Gol Ya Pooch Architecture Notes (Phases 1 & 2)

## Design Goals

1. **Zero External Runtime Requests:** All assets required at runtime (`@mediapipe/tasks-vision` WASM files, `hand_landmarker.task`, and `Vazirmatn` `.woff2` fonts) live in `public/` and are copied verbatim into `dist/`. Vite builds with `base: './'` so static deployment works on both localhost and GitHub Pages subpaths.
2. **Privacy by Construction:** Camera frames are drawn onto an in-memory `HTMLCanvasElement` for typed-array `ImageData` color/lighting checks and passed to the local MediaPipe WASM runtime. `SessionRecorder` (`src/vision/recorder.ts`) and IndexedDB behavioral profile storage (`src/storage/profileStore.ts`) only store numeric landmarks, derived kinematics, game events, and expert weights — **never images or video**.
3. **Resilient Hand Identity (`src/vision/handTracker.ts`):** MediaPipe's single-frame `handedness` classification frequently swaps when hands cross. `HandIdentityTracker` tracks `LEFT` and `RIGHT` wrists using velocity-predicted spatial positions (computed from raw wrist landmarks to avoid smoothing lag) and optimal Hungarian assignment (`hungarianAssign`), treating MediaPipe `handedness` purely as a weak tie-breaker (`0.001`).
4. **Explicit Game FSM (`src/game/fsm.ts`):** Game state transitions are pure functions `(GameContext, GameEvent) => GameContext`, making every phase timeout, void-round rule, tab-hidden pause/resume, and recoverable camera loss deterministic and directly testable with unit tests and `fast-check` property-based tests.

## Phase 2: Tracking, Behavioral Modeling, and Rule Engine

5. **Per-Frame Kinematic & Geometric Features (`src/vision/features.ts`):**
   - `FeatureExtractor` computes palm-size-normalized joint flexion angles (MCP, PIP, DIP) and total curl per finger, fist compactness (2D convex-hull area + mean fingertip-to-palm-center distance), sliding-window compactness deltas, wrist velocity/acceleration/jitter, 3D palm normal & depth proxies, inter-hand landmark proximity & contact duration, partial-opening events (brief curl dip followed by recovery), per-hand gol-sighting scores, and frame quality flags.
   - **Honest Physical Limitation:** Muscle tension itself is **not directly observable** at standard webcam resolution. Squeezing, relaxing, and hand tension are **approximated** through finger geometry (curl and fist compactness) and temporal dynamics (sliding-window compactness deltas and wrist jitter/tremor).

6. **Event Detector & Bayesian HMM Belief Filter (`src/tracking/eventDetector.ts`, `src/tracking/beliefFilter.ts`):**
   - `EventDetector` converts frame-by-frame `FrameFeatures` during `HIDING` into a structured timeline of discrete events (`CONTACT_START`, `CONTACT_END`, `PARTIAL_OPENING`, `GOL_SIGHTING`, `OCCLUSION_START`, `OCCLUSION_END`, `COMPACTNESS_SHIFT`).
   - `BeliefFilter` runs a 2-state discrete HMM forward pass over `{LEFT, RIGHT}`:
     - Initialized near-certain (`0.99` / `0.01`) on the hand confirmed during `SHOW_GOL`.
     - Transition model keeps transfer probability near zero when hands are separated, and increases transfer probability with contact duration and partial-opening cues during contact.
     - Observation model applies strong likelihoods to gol sightings, weak conservative likelihoods to compactness/curl asymmetries, and flattened likelihoods on occluded or low-quality frames.
   - **UX Isolation:** Live belief is hidden from the player during `HIDING` and appears only inside the toggleable Debug Overlay (`D` key).

7. **Behavioral Experts & Hedge Ensemble (`src/ai/behavior/`, `src/ai/ensemble.ts`, `src/storage/profileStore.ts`):**
   - Experts predict `P(LEFT)` from past round choices: overall & recent-window frequency, n-grams (`n = 1..3`), win-stay/lose-shift Markov chain, and repeat-after-success / switch-after-being-caught detectors.
   - `HedgeEnsemble` mixes experts with multiplicative weights updated after every verified round and persisted per player profile in IndexedDB via `idb` (numeric-only).

8. **Final Guesser & Declarative Rules (`src/ai/guesser.ts`, `src/game/rules/`, `src/game/scoring.ts`):**
   - `Guesser` fuses tracking belief and behavioral prior in log-odds space so tracking dominates when the hiding phase is clearly observed and behavioral priors take over when tracking confidence is low.
   - Generates human-readable Persian timeline explanations for `ROUND_RESULT`.
   - `classic-1v1.json` is validated at runtime by `src/game/rules/schema.ts` and executed by `src/game/rules/engine.ts` + `src/game/scoring.ts`.
