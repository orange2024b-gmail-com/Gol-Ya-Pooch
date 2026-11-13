# گل یا پوچ (Gol ya Pooch)

<div dir="rtl" lang="fa">

بازی سنتی ایرانی **«گل یا پوچ»** با بینایی ماشین تماماً سمت کاربر (Client-Side) برای مرورگرهای دسکتاپ و موبایل.

## وضعیت فعلی: فاز ۱ (شالوده بینایی ماشین و اسکلت بازی)

- **پردازش کاملاً محلی و حفظ حریم خصوصی:** تمام فایل‌های `@mediapipe/tasks-vision` (WASM)، مدل `hand_landmarker.task` و فونت `Vazirmatn` به‌صورت Self-Hosted در پوشه `public/` قرار دارند و در حین بازی **هیچ درخواست خارجی** و **هیچ ارسال یا ذخیره تصویر/ویدیو** انجام نمی‌شود.
- **رابط کاربری فارسی و راست‌به‌چپ (RTL):** با ارقام فارسی، پشتیبانی از موبایل عمودی و دسکتاپ، پیش‌نمایش آینه‌ای دوربین و پنل دیباگ (`D`).
- **خط لوله بینایی ماشین (`src/vision/`):**
  - تشخیص دو دست با `HandLandmarker` در حالت `VIDEO` و فیلتر One-Euro همراه با نگهداری مختصات خام.
  - حفظ پایداری هویت دست چپ/راست با پیوستگی فضایی (تخصیص مجارستانی / Hungarian بر اساس موقعیت مچ و پیش‌بینی سرعت، با وزن ضعیف برچسب MediaPipe) و ردیابی `lostFrames` / انسداد (`occluded`).
  - کالیبراسیون رنگ گل در فضاهای HSV و CIE Lab، تشخیص پینچ (انگشت شست و اشاره) با تأیید در `N=15` فریم پیاپی (`گل شناسایی شد ✓`)، و بررسی کف دست باز در مرحله بازگشایی.
  - حالت ضبط و بازپخش عددی (`src/vision/recorder.ts`) برای ذخیره و بازپخش دنباله نقاط کلیدی در JSON بدون دوربین.
- **ماشین حالت بازی (`src/game/fsm.ts`):** مدیریت تمام فازها، تایمرها، قوانین ابطال دور (خروج دست > ۰٫۵ ثانیه، نبود گل در پینچ، نبود/تکرار گل در بازگشایی)، توقف خودکار در زمان مخفی‌شدن تب و بازیابی خطای دوربین.

## اجرای محلی (توسعه و آزمون)

</div>

```bash
npm ci
npm run lint
npm run typecheck
npm test
npm run build
npm run test:e2e
npm run dev
```

---

## English Overview

**Gol ya Pooch** is a client-side computer vision web implementation of the traditional Iranian hand-guessing game, built with Vite and TypeScript (strict mode).

### Phase 1 Scope (Completed)

- **Zero external runtime requests & strict privacy:** All MediaPipe WASM binaries, `hand_landmarker.task`, and `Vazirmatn` `.woff2` fonts are self-hosted in `public/`. Video frames are processed strictly in-memory inside the browser; nothing is uploaded or persisted.
- **Persian RTL UI:** Full Persian localization with Persian numerals, responsive portrait mobile and desktop layouts, mirrored preview toggle, reduced-motion option, and live debug overlay (`D` key or UI toggle).
- **Vision Pipeline (`src/vision/`):**
  - MediaPipe `HandLandmarker` (`VIDEO` mode, `numHands: 2`) with One-Euro smoothing and raw landmark retention.
  - Spatial-continuity left/right hand identity tracking via Hungarian assignment on velocity-predicted wrist positions (`HandIdentityTracker`), tracking per-hand `lostFrames` and `occluded` flags.
  - HSV + CIE Lab gol color calibration, pinch posture + ROI color-blob detection with `N=15` consecutive frame confirmation, and dual open-palm reveal verification.
  - Numeric-only session recording and deterministic camera-free replay (`src/vision/recorder.ts`).
- **Game State Machine (`src/game/fsm.ts`):** Explicit FSM with phase timeouts, void-round rules (`> 500 ms` hand loss, unconfirmed pinch, neither/both reveal), tab-visibility pause/resume, recoverable camera loss handling, and placeholder Phase 1 computer guess.

### Project Structure

- `public/mediapipe/` — Self-hosted MediaPipe WASM runtime and `hand_landmarker.task`
- `public/fonts/` — Self-hosted `Vazirmatn` font files and stylesheet
- `src/vision/` — Hand tracking, One-Euro filter, quality checks, gol detector, record/replay
- `src/game/fsm.ts` — Explicit Phase 1 game FSM
- `src/camera/` — `getUserMedia` camera lifecycle and resolution fallback
- `src/app/`, `src/render/`, `src/styles/` — Persian RTL UI, overlay renderer, and responsive styles
- `tests/` — Vitest unit tests, `fast-check` property tests, replay vision regression tests, and Playwright E2E tests
- `docs/` — Architecture and verification notes
