# Manual Verification Checklist (Per Device & Browser)

Use this checklist on physical devices before tagging a public release.

> **Prerequisite:** Serve over **HTTPS** (GitHub Pages URL or local HTTPS tunnel) for mobile camera `getUserMedia` access. Open Browser DevTools -> Network tab before testing to confirm **0 external runtime requests** (with Optional LLM Commentary left OFF).

---

## 1. Common Functional Checks (All Devices & Browsers)

| # | Area | Manual Steps | Expected Outcome |
|---|---|---|---|
| 1 | **Zero External Requests** | Open DevTools Network tab -> hard reload app -> complete Splash, Menu, Calibration, 1 Practice round, and `?benchmark=1`. | Every request is same-origin (`./fonts/vazirmatn*`, `./favicon.svg`, `./mediapipe/*`, `./ort/*`, `./models/*`) or `blob:`/`data:`. **0 external network calls.** |
| 2 | **Branding & Typography** | Inspect Splash & Topbar on light/dark display. | Custom Shamseh + cupped hands + golden pearl SVG logo renders crisply; Persian Vazirmatn font active; Persian digits (`۰–۹`) in HUD, scores, timers, and stats; RTL layout aligned. |
| 3 | **Audio & Accessibility** | Toggle Topbar `🔊 صدا` / `🔇 بی‌صدا` -> reload page -> toggle `Reduced Motion` and `High Contrast` in Menu Settings -> press `Tab` / `D` / `Escape`. | Mute state persists in `localStorage`; animations stop under `Reduced Motion`; borders/text sharpen in `High Contrast`; focus ring visible; `D` opens Debug, `Escape` exits sub-screens. |
| 4 | **Calibration Wizard** | Grant camera -> test poor light (cover lens or backlight) -> show both hands for 1+ sec -> place colored bead inside central circle and click `ثبت رنگ گل`. | Lighting hint (`بیشتر نور بدهید / پشت به پنجره ننشینید`) triggers when dark; hand stability reaches 100% after 1 s; `تأیید و ورود به منوی بازی` unlocks only after all 4 checks pass. |
| 5 | **Human Hides -> Computer Guesses** | Start Level 1 -> hold gol between thumb & index -> confirm `گل شناسایی شد ✓` -> hide in one fist (try 0, 1, and 2+ transfers) -> reveal both open palms. | Live pinch ring fills over ~0.5 s; **belief % is NEVER shown during HIDING**; Debug overlay (`D`) shows live HMM belief graph & ONNX status `READY`; open palms verify ground truth and display Persian explanation + timeline. |
| 6 | **Void & Pause Rules** | (a) Move one hand out of frame > 0.5 s during HIDING. (b) In Level 3, finish HIDING with 0 hand contacts. (c) Switch browser tab mid-round and return. (d) Revoke camera permission mid-round. | (a) `ROUND_VOID` (`دست‌ها بیش از نیم ثانیه از کادر خارج شدند`). (b) `ROUND_VOID` (`NO_HAND_CONTACT`). (c) Enters `PAUSED` and resumes timer cleanly. (d) Recovers to `CAMERA_ERROR` screen without crash. |
| 7 | **Computer Hides & Commit–Reveal** | Enter Practice or Computer-Hides turn -> note Commit fingerprint before guessing -> tap hand -> check Result. | PixiJS animated hands shuffle; SHA-256 fingerprint is visible **before** guessing; Result verifies hash (`✓ تأیید شد`) and `این چطور کار می‌کند؟` expands Persian explanation. |
| 8 | **Traditional Team & Levels 1–5** | Switch to Level 4 or `traditional-team` (4 or 6 hands) -> call `پوچ` on empty hand, then `گل` on gol hand -> unlock Level 2 by winning Level 1. | `پوچ` opens empty hand; direct `گل` gives `+2`; `گل` after `پوچ` gives `+0` + turn pass; `پوچ` on gol gives hider `+1`; progression persists across reload. |
| 9 | **Consent, Numeric Export & LLM Fallback** | Check Consent screen (`پیش‌فرض خاموش`) -> opt in -> play 1 camera/replay round -> Export JSON. Enable LLM with invalid URL `http://127.0.0.1:9/v1` and click Test. | Consent defaults OFF; exported JSON contains zero images/video (`no-images-no-video`); unreachable LLM silently falls back to Persian template within 3 s with 0 errors. |
| 10 | **In-App Benchmark (`?benchmark=1`)** | Open `<url>?benchmark=1` -> wait for results. | Displays measured RAF FPS, Feature Extraction ms, HMM step ms, ONNX WASM inference ms (`< 10 ms`), and 24/30 FPS budget status. |

---

## 2. Device & Browser Matrix

### Desktop Chrome / Edge (Windows / macOS / Linux)
- **Camera:** 1280×720 @ 30 FPS (`mirrored` ON by default).
- **Targets:** Hand-tracking FPS `>= 30 FPS`; ONNX WASM inference `< 2 ms`; WebGL PixiJS smooth 60 FPS; `D` debug hotkey works.
- **Status:** Automated E2E verified on headless Linux Chromium VM (see `docs/test-report.md`); physical webcam lighting/skin-tone check pending owner sign-off.

### Desktop Firefox (Windows / macOS / Linux)
- **Specific Checks:**
  - Verify `getUserMedia` camera permission prompt and device dropdown switching (`cameraDevices`).
  - Verify `vazirmatn-arabic-*.woff2` renders crisp RTL Persian without fallback font flicker.
  - Verify `onnxruntime-web` WASM loads (`READY (wasm, ...)` or graceful fallback if SharedArrayBuffer restricted — `numThreads = 1` avoids COOP/COEP requirement).
- **Status:** Pending manual owner check.

### Desktop Safari (macOS)
- **Specific Checks:**
  - Verify `<video playsInline muted>` autoplays `MediaStream` immediately after permission click.
  - Verify Web Audio `ProceduralSoundEngine` unlocks on first button click (`ctx.resume()`).
  - Verify IndexedDB (`idb`) persists progression and consent settings across reloads.
- **Status:** Pending manual owner check.

### Android Chrome (Mid-Range Phone, Portrait)
- **Specific Checks:**
  - Front camera (`facingMode: 'user'`) selected automatically; switch-camera dropdown works if multiple cameras reported.
  - Portrait layout (`360–412 px` width) shows zero horizontal overflow; buttons meet `>= 48 px` touch target; 4-hand & 6-hand PixiJS grid taps are accurate.
  - Hand tracking sustains `>= 24 FPS` (if slower, verify graceful resolution downgrade banner `سرعت پردازش پایین است؛ رزولوشن برای روانی بازی کاهش یافت`).
  - Run `?benchmark=1` and confirm ONNX inference `< 10 ms`.
- **Status:** Pending physical Android hardware check.

### iOS Safari (iPhone, Portrait)
- **Specific Checks:**
  - Verify `viewport-fit=cover`, `playsInline`, and `muted` allow live camera preview without fullscreen takeover.
  - Verify WebGL PixiJS scene renders without context loss when returning from background tab (`visibilitychange` pause/resume).
  - Verify Web Audio cues play after first user tap and mute button silences them reliably.
  - Run `?benchmark=1` and verify ONNX WASM single-threaded inference `< 10 ms`.
- **Status:** Pending physical iOS hardware check.
