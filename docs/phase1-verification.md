# Phase 1 Verification Summary

## Test Scope & Conditions

- **Live physical camera & real human participants:** Not executed in this headless Linux environment. Real-camera testing on laptop Chrome and Android Chrome remains for owner sign-off before Phase 2.
- **Headless Chromium E2E (`tests/e2e/phase1.spec.ts`):** Executed against the production Vite build (`vite preview`) using Chrome's `--use-fake-device-for-media-stream` and `--use-fake-ui-for-media-stream` to verify real WASM/model loading, camera stream acquisition, UI responsiveness, debug toggle, sample replay, and zero external network requests.
- **Vision Regression (`tests/vision/`):** Executed via `replayRecordedSession` on synthetic/recorded numeric and RGBA fixtures:
  - `tests/vision/handCrossing.test.ts`: 20 hand-crossing sessions (including double crossings, vertical offsets, forced MediaPipe handedness flips at crossing, and 3-frame mid-crossing occlusions) -> **0 identity swaps**.
  - `tests/vision/pinchConfirmation.test.ts`: 40 synthetic/recorded pinch attempts across 4 bead colors and 4 lighting backgrounds -> **40/40 confirmed (100%)** with `N = 15` consecutive frames.
- **FSM Coverage (`tests/unit/fsm.test.ts`, `tests/unit/fsm.property.test.ts`):** Unit tests covering all 12 FSM states and transitions + 300 `fast-check` randomized event sequences (up to 120 events each).
