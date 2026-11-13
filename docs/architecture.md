# Gol Ya Pooch Architecture Notes (Phases 1–5)

## 1. Design Principles & Zero-External-Runtime Guarantee

1. **Zero External Runtime Requests:** All core assets live under `public/` and are copied into `dist/` at build time (`scripts/sync-assets.mjs` + Vite public copy):
   - `public/mediapipe/hand_landmarker.task` & `public/mediapipe/wasm/*` (`@mediapipe/tasks-vision@1.2.3`)
   - `public/ort/ort-wasm-simd-threaded.{wasm,mjs}` (`onnxruntime-web@1.31.1`)
   - `public/models/temporal_tracker.onnx` (31 KB) & `public/models/fusion_config.json`
   - `public/fonts/vazirmatn.css` + `vazirmatn-{arabic,latin}-{400,500,700,800}-normal.woff2`
   - `public/favicon.svg` + procedural inline SVG brand logo (`src/render/brandLogo.ts`) + procedural PixiJS v8 vector hands (`src/render/pixiHidingScene.ts`) + procedural Web Audio synthesizer (`src/audio/soundEngine.ts`).
   - Vite builds with `base: './'` (or `VITE_BASE_PATH` on GitHub Pages) so deployment works cleanly at root or subpath.

2. **Privacy & No-Image/No-Video Proof (Verifiable End-to-End):**
   - **Camera Capture (`src/camera/cameraManager.ts`):** `getUserMedia({ audio: false, video: ... })` renders only to a local `<video>` element and a `320x240` in-memory `HTMLCanvasElement` for typed-array `ImageData` HSV/CIE Lab color sampling and MediaPipe inference. Frames are discarded immediately after each tick.
   - **Recording & Round Logging (`src/vision/recorder.ts`, `src/storage/roundLogStore.ts`):**
     - `IndexedDbRoundLogStore` defaults to `consentEnabled: false`.
     - `assertNumericOnlyPayload` recursively rejects any media-like property names (`image`, `video`, `frame`, `canvas`, `blob`, `pixels`, `rgba`, etc.), typed arrays / `ArrayBuffer`, DOM media objects, and `data:image/` or `data:video/` URIs before persistence, during JSON export, and on JSON import.
   - **Optional LLM Client (`src/llm/commentaryClient.ts`):**
     - Disabled (`enabled: false`) by default.
     - `assertTextOnlyLlmPayload` enforces at runtime that only plain-text `messages` (`system` + `user` strings) are serialized; any image key, multimodal `image_url`, `data:image/`/`data:video/` string, or binary buffer throws and falls back to template commentary.
   - **Playwright Network Audit (`tests/e2e/phase1..5.spec.ts`):** Every E2E test attaches a `page.on('request')` listener verifying `0` requests outside the local origin (`127.0.0.1:4173`), `data:`, and `blob:`.

---

## 2. Layered Module Map

```
src/
├─ main.ts                      # Mounts GolYaPoochApp
├─ app/
│  ├─ appController.ts          # Screen routing, camera/Pixi loop orchestration, settings, UI
│  ├─ benchmark.ts              # In-app DeviceBenchmark (?benchmark=1): RAF FPS, features, HMM, ORT-Web
│  └─ i18n.ts                   # Persian RTL copy, Persian digits, RULES_BOOK_SECTIONS_FA
├─ audio/
│  └─ soundEngine.ts            # Procedural Web Audio API cues + localStorage mute toggle
├─ camera/
│  └─ cameraManager.ts          # getUserMedia (1280x720 -> 640x480 fallback), device switch, loss callback
├─ vision/
│  ├─ handTracker.ts            # MediaPipe HandLandmarker + One-Euro filter + Hungarian left/right tracking
│  ├─ oneEuro.ts                # Adaptive One-Euro smoother
│  ├─ golDetector.ts            # HSV + CIE Lab Mahalanobis color model, pinch ROI, palm open-reveal verification
│  ├─ features.ts               # Normalized finger curl, convex-hull compactness, velocity/jitter, inter-hand distance
│  ├─ quality.ts                # Luminance, contrast, Laplacian blur, 1s calibration stability
│  ├─ recorder.ts               # Numeric-only RecordedSession capture & camera-free deterministic replay
│  └─ sampleSession.ts          # Built-in Phase 1 & Phase 2 deterministic replay sessions
├─ tracking/
│  ├─ eventDetector.ts          # Discrete hiding events (CONTACT, PARTIAL_OPENING, GOL_SIGHTING, OCCLUSION)
│  ├─ beliefFilter.ts           # 2-state Bayesian HMM forward pass over {LEFT, RIGHT}
│  ├─ temporalFeatures.ts       # Fixed-rate 10 Hz resampling/padding into [64 x 24] tensor
│  └─ temporalModel.ts          # Self-hosted onnxruntime-web WASM inference + logistic HMM fusion + fallback
├─ ai/
│  ├─ behavior/                 # Frequency, n-gram (1..3), Markov (WSLS), habituation experts
│  ├─ ensemble.ts               # Multiplicative-weights (Hedge) expert mixer
│  ├─ guesser.ts                # Log-odds fusion of (HMM + ONNX) tracking & behavioral prior + FA explanation
│  └─ hider.ts                  # Unbiased crypto hiding (L1–4), bounded-exploitation Hedge (L5), choreography
├─ game/
│  ├─ fsm.ts                    # Pure transitionGameFsm (19 states), timeouts, void rules, pause/recover
│  ├─ levels.ts                 # Levels 1–5 definitions & unlock progression
│  ├─ commitReveal.ts           # Web Crypto crypto.getRandomValues 16B salt + SHA-256 commitment/verification
│  └─ rules/                    # Schema-validated JSON rules (classic-1v1, traditional-team, custom)
├─ render/
│  ├─ brandLogo.ts              # Custom Persian Shamseh + cupped hands + golden-pearl gol inline SVG
│  ├─ canvasOverlay.ts          # Landmarks, calibration circle, pinch ring, debug belief trace graph
│  └─ pixiHidingScene.ts        # PixiJS v8 procedural 2/4/6-hand animations & touch targets
├─ storage/
│  ├─ profileStore.ts           # IndexedDB behavioral ensemble & history
│  ├─ progressionStore.ts       # IndexedDB unlocked levels & round/match stats
│  └─ roundLogStore.ts          # Consent-gated numeric round log store & JSON export/import
└─ llm/
   └─ commentaryClient.ts       # Optional OpenAI-compatible text-only host commentary (3s AbortController)
```

---

## 3. Vision, Tracking & Machine Learning Notes

- **Resilient Hand Identity (`HandIdentityTracker`):** Uses velocity-predicted wrist positions computed on raw coordinates and optimal Hungarian assignment (`hungarianAssign`), treating MediaPipe `handedness` strictly as a `0.001` weak prior so crossing hands never swap identities.
- **Physical Limitation Transparency:** Muscle tension itself is not directly observable at standard webcam resolution; squeezing and relaxing are **approximated** from finger flexion angles, 2D convex-hull fist compactness, sliding-window compactness deltas, and wrist jitter.
- **Offline Training & Ship Gate (`training/`):**
  - Strictly player-grouped split (`split_by_player`, zero overlap between train/val/test players).
  - Compact Conv1D + GRU (`TemporalTrackerNet`) exported to `public/models/temporal_tracker.onnx` (`31 KB`) with validation-calibrated logistic fusion (`hmmWeight`, `temporalWeight`, `bias`).
  - Ship gate requires fused held-out test accuracy to exceed the HMM-only baseline (`90.12%` vs `75.93%` on the 162-round held-out test split, ECE improved from `0.2022` to `0.0599`).

---

## 4. Optional LLM Commentary & Serverless Proxy Option (Documented Only)

`src/llm/commentaryClient.ts` is **OFF by default**. When enabled in Settings:
- Sends **only** a short Persian text summary (`system` + `user` messages) of the round/level/match outcome.
- Enforces a hard **3000 ms** timeout via `AbortController`; any timeout, network failure, HTTP non-2xx, or malformed payload silently falls back to `generateTemplateCommentary(event)`.
- **Key Management Options for Static Hosting (GitHub Pages):**
  1. **User-supplied personal API key:** Stored strictly in the user's browser `localStorage` (`gol-ya-pooch:llm-config`), never committed or embedded in source code.
  2. **Tiny Serverless Proxy (Recommended if sharing a hosted key — not built in this static repo):** Deploy a lightweight edge function (e.g., Cloudflare Worker or Vercel Edge Function) that holds the provider secret in server environment variables, enforces origin CORS + rate limiting + text-only schema validation, and forwards the OpenAI-compatible request:

```ts
// Reference Cloudflare Worker / Edge proxy blueprint (do NOT embed secrets in client code)
export default {
  async fetch(request: Request, env: { LLM_API_KEY: string; LLM_ENDPOINT: string }): Promise<Response> {
    if (request.method === 'OPTIONS') {
      return new Response(null, {
        headers: {
          'Access-Control-Allow-Origin': 'https://<your-username>.github.io',
          'Access-Control-Allow-Methods': 'POST, OPTIONS',
          'Access-Control-Allow-Headers': 'Content-Type',
        },
      });
    }
    if (request.method !== 'POST') return new Response('Method Not Allowed', { status: 405 });

    const body = (await request.json()) as { model?: string; messages?: Array<{ role: string; content: unknown }> };
    // Enforce text-only messages on proxy too
    if (!Array.isArray(body.messages) || body.messages.some((m) => typeof m.content !== 'string')) {
      return new Response('Text-only messages required', { status: 400 });
    }

    const upstream = await fetch(env.LLM_ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${env.LLM_API_KEY}`,
      },
      body: JSON.stringify({
        model: body.model ?? 'gpt-4o-mini',
        messages: body.messages.slice(0, 4),
        max_tokens: 140,
        temperature: 0.7,
      }),
    });

    return new Response(upstream.body, {
      status: upstream.status,
      headers: {
        'Content-Type': 'application/json',
        'Access-Control-Allow-Origin': 'https://<your-username>.github.io',
      },
    });
  },
};
```

In the game Settings UI, the player simply enters the Worker URL as **Endpoint** and leaves the **API Key** field blank.
