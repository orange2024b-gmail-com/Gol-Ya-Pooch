# Phase 1 Architecture Notes

## Design Goals

1. **Zero External Runtime Requests:** All assets required at runtime (`@mediapipe/tasks-vision` WASM files, `hand_landmarker.task`, and `Vazirmatn` `.woff2` fonts) live in `public/` and are copied verbatim into `dist/`. Vite builds with `base: './'` so static deployment works on both localhost and GitHub Pages subpaths.
2. **Privacy by Construction:** Camera frames are drawn onto an in-memory `HTMLCanvasElement` for typed-array `ImageData` color/lighting checks and passed to the local MediaPipe WASM runtime. `SessionRecorder` (`src/vision/recorder.ts`) only serializes numeric landmarks, timestamps, quality scalars, and boolean/ratio detection summaries.
3. **Resilient Hand Identity (`src/vision/handTracker.ts`):** MediaPipe's single-frame `handedness` classification frequently swaps when hands cross. `HandIdentityTracker` tracks `LEFT` and `RIGHT` wrists using velocity-predicted spatial positions (computed from raw wrist landmarks to avoid smoothing lag) and optimal Hungarian assignment (`hungarianAssign`), treating MediaPipe `handedness` purely as a tiny tie-breaker (`0.001`).
4. **Explicit Game FSM (`src/game/fsm.ts`):** Game state transitions are pure functions `(GameContext, GameEvent) => GameContext`, making every phase timeout, void-round rule, tab-hidden pause/resume, and recoverable camera loss deterministic and directly testable with unit tests and `fast-check` property-based tests.
