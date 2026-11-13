# گل یا پوچ (Gol ya Pooch)

<div dir="rtl" lang="fa">

بازی سنتی ایرانی **«گل یا پوچ»** با بینایی ماشین تماماً سمت کاربر (Client-Side) برای مرورگرهای دسکتاپ و موبایل — بدون سرور، بدون ردیاب، و با تضمین کامل حریم خصوصی.

## ویژگی‌های کلیدی (فاز ۱ تا ۵ — نسخه کامل)

- **پردازش ۱۰۰٪ محلی و بدون درخواست خارجی:** تمام فایل‌های `@mediapipe/tasks-vision` (WASM)، مدل `hand_landmarker.task`، باینری‌های `onnxruntime-web` WASM، مدل زمانی `temporal_tracker.onnx`، فونت `Vazirmatn`، و لوگوی اختصاصی SVG/Favicon به‌صورت Self-Hosted در `public/` قرار دارند. در اجرای اصلی بازی **هیچ درخواست خارجی** و **هیچ ارسال یا ذخیره تصویر/ویدیو** انجام نمی‌شود.
- **نشان اختصاصی و تایپوگرافی فارسی (RTL):** لوگوی هندسی شمسه + دستان دربرگیرنده گُل زرین (`public/favicon.svg` و SVG درون‌خطی)، تایپوگرافی دقیق با فونت Self-Hosted **Vazirmatn**، ارقام فارسی هم‌عرض (`tabular-nums`)، چیدمان واکنش‌گرا برای موبایل عمودی و دسکتاپ.
- **بینایی ماشین و ردیابی پایدار (`src/vision/`, `src/tracking/`):**
  - تشخیص دو دست با `HandLandmarker` در حالت `VIDEO`، هموارسازی One-Euro و حفظ پایداری هویت دست چپ/راست با تخصیص مجارستانی (Hungarian) روی موقعیت مچ.
  - جادوگر کالیبراسیون چهارمرحله‌ای (نور محیط، پایداری ۱ ثانیه‌ای هر دو دست، فاصله، و کالیبراسیون رنگ گل در HSV و CIE Lab).
  - تشخیص پینچ (شست و اشاره) با تأیید `N=15` فریم پیاپی (`گل شناسایی شد ✓`)، ردیابی تماس دست‌ها، نیمه‌بازشدن و رؤیت گل در حین پنهان‌کاری، و بررسی کف باز هر دو دست در بازگشایی.
  - فیلتر باور بیزین (HMM) + مدل زمانی یادگرفته‌شده **Conv1D+GRU (`temporal_tracker.onnx`، ۳۱ کیلوبایت)** روی `onnxruntime-web` با ترکیب لجستیک کالیبره‌شده و بازگشت خودکار به HMM.
- **هوش مصنوعی پنهان‌کاری، انیمیشن PixiJS v8 و تعهد رمزنگاری (`src/ai/`, `src/render/`, `src/game/`):**
  - در نوبت رایانه، دست‌های دوبعدی متحرک با PixiJS v8 (۲، ۴ یا ۶ دست) و حرکات فریبنده (Feint / Misdirection) گل را پنهان می‌کنند.
  - تضمین عدم تقلب با **Commit–Reveal (SHA-256)** پیش از حدس بازیکن و راستی‌آزمایی خودکار پس از حدس (`✓ تأیید شد`).
  - قوانین داده-محور JSON (`classic-1v1` و `traditional-team` با فراخوانی «پوچ» و «گل» و `custom.json`) در ۵ سطح از «تازه‌کار» تا «استاد گل‌گیر».
- **صدا، دسترسی‌پذیری، بنچمارک و تفسیرگر اختیاری LLM (`src/audio/`, `src/llm/`, `src/app/`):**
  - افکت‌های صوتی رویه‌ای (Web Audio API بدون فایل خارجی) با دکمه بی‌صدا (`🔊` / `🔇`)، کاهش انیمیشن (`Reduced Motion`)، حالت کنتراست بالا (`High Contrast`)، و پیمایش کیبورد (`D` برای دیباگ، `Escape` برای بازگشت).
  - ابزار **بنچمارک عملکرد دستگاه** (`?benchmark=1` یا از منو) برای اندازه‌گیری واقعی RAF FPS، استخراج ویژگی‌ها، فیلتر HMM و استنتاج ONNX.
  - تفسیرگر اختیاری میزبان با LLM (**به‌طور پیش‌فرض خاموش**): فقط خلاصه متنی رویدادها را ارسال می‌کند (هرگز تصویر ارسال نمی‌شود)، کلید کاربر فقط در `localStorage` می‌ماند و با مهلت ۳ ثانیه‌ای (`AbortController`) به‌صورت خودکار و بی‌صدا به جملات آماده فارسی برمی‌گردد.

## مستندات بازی

- [کتابچه قوانین کامل فارسی (`docs/rules-fa.md`)](./docs/rules-fa.md)
- [معماری ۵ فازی و مستندات پراکسی بدون سرور LLM (`docs/architecture.md`)](./docs/architecture.md)
- [چک‌لیست آزمون دستی روی دستگاه‌ها و مرورگرها (`docs/manual-test.md`)](./docs/manual-test.md)
- [گزارش نهایی آزمون‌ها و بنچمارک (`docs/test-report.md`)](./docs/test-report.md)

## اجرا، آزمون و استقرار

</div>

```bash
# Install dependencies
npm ci

# Sync self-hosted MediaPipe, Vazirmatn, and ORT-Web WASM assets
npm run assets:sync

# Code quality & tests
npm run lint
npm run typecheck
npm run test:coverage
npm run build
npm run test:e2e

# Local dev or preview server (with ?benchmark=1 for live device benchmark)
npm run dev
npm run preview
```

### Offline Training Pipeline (`/training`)

```bash
cd training
python3 -m venv .venv && ./.venv/bin/pip install -r requirements.txt
./.venv/bin/pytest -v
./.venv/bin/python -m gol_training.train \
  --output-dir ../public/models \
  --report-path ../docs/phase4-training-report.md \
  --parity-fixture-path ../tests/unit/fixtures/temporal_parity.json
```

### GitHub Pages Deployment

1. Push to `main` (or trigger `Deploy to GitHub Pages` workflow manually via `workflow_dispatch`).
2. `.github/workflows/deploy.yml` runs `npm ci`, `lint`, `typecheck`, `test`, sets `VITE_BASE_PATH` to the repository subpath, builds `dist/`, and publishes via `actions/deploy-pages@v4` over HTTPS (required for browser `getUserMedia` camera access).

---

## English Overview

**Gol ya Pooch (گل یا پوچ)** is a production-quality, privacy-first, client-side computer vision web game implementing the traditional Iranian hand-guessing game against an AI opponent using the device webcam (plus a camera-free practice mode).

### Complete Feature Set (Phases 1–5)

- **Zero External Runtime Requests & Strict Privacy:** Self-hosted MediaPipe `@mediapipe/tasks-vision` WASM + `hand_landmarker.task`, `onnxruntime-web` WASM binaries, `temporal_tracker.onnx` (31 KB), `Vazirmatn` `.woff2` fonts, `public/favicon.svg`, and procedural SVG + Web Audio synthesis. **No image or video frame ever leaves the device or is persisted.**
- **Custom SVG Shamseh Brand Logo & Refined Persian RTL Typography:** Bespoke geometric shamseh & cupped-hands SVG logo (`src/render/brandLogo.ts`, `public/favicon.svg`), self-hosted Vazirmatn typography with `tabular-nums`, high-contrast non-color-only cues, `Reduced Motion`, `High Contrast`, and mobile portrait (`390x844`) touch ergonomics.
- **Vision & Tracking Pipeline (`src/vision/`, `src/tracking/`):**
  - 2-hand `HandLandmarker` (`VIDEO` mode) with One-Euro filtering and velocity-predicted Hungarian wrist assignment to prevent left/right identity swaps during hand crossings.
  - 4-step Calibration Wizard (luminance/contrast, 1 s hand stability, distance, HSV + CIE Lab gol color calibration).
  - Pinch confirmation (`N=15` consecutive frames), hiding-phase ROI gol sightings, contact/partial-opening event detection, Bayesian HMM belief filter, and self-hosted `onnxruntime-web` Conv1D+GRU temporal inference (`< 1 ms` desktop Chromium WASM) with calibrated logistic fusion and automatic HMM fallback.
- **Computer Hiding, PixiJS v8, Rules & Fairness (`src/ai/`, `src/render/`, `src/game/`):**
  - PixiJS v8 animated stylized hands (`2`, `4`, or `6` hands) with swaps, feints, and multi-hand misdirection.
  - Cryptographic **Commit–Reveal (`SHA-256`)** via Web Crypto proving computer choices were committed before the player guesses.
  - Schema-validated JSON rules (`classic-1v1.json`, `traditional-team.json`, `custom.json`) and 5 progressive levels (`Beginner` to `Master Gol-Gir` Best-of-3 with bounded exploitation).
- **Procedural Sound, Benchmark & Optional LLM Commentary (`src/audio/`, `src/llm/`, `src/app/`):**
  - Synthesized Web Audio sound effects with persistent mute toggle (`localStorage`).
  - In-app **Device Performance Benchmark** (`?benchmark=1` or from Menu/Debug) measuring live RAF FPS, feature extraction, HMM filter step, and ORT-Web inference timings.
  - **Optional LLM Host Commentary (OFF by default):** OpenAI-compatible text-only client storing user-provided keys strictly in `localStorage`, enforcing a `assertTextOnlyLlmPayload` guard against media, and silently falling back to rich Persian templates on error or after a hard `3000 ms` `AbortController` timeout.

### Documentation

- [Persian Rule Book (`docs/rules-fa.md`)](./docs/rules-fa.md)
- [5-Phase Architecture & Serverless LLM Proxy Guide (`docs/architecture.md`)](./docs/architecture.md)
- [Manual Device & Browser Verification Checklist (`docs/manual-test.md`)](./docs/manual-test.md)
- [Final Phase 5 Test Campaign Report (`docs/test-report.md`)](./docs/test-report.md)
