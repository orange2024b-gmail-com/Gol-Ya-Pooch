const PERSIAN_DIGITS = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'] as const;

export function toPersianDigits(value: number | string): string {
  return String(value).replace(/\d/g, (digit) => PERSIAN_DIGITS[Number(digit)] ?? digit);
}

export function formatSecondsFa(ms: number): string {
  const sec = Math.max(0, Math.ceil(ms / 1000));
  return `${toPersianDigits(sec)} ثانیه`;
}

export function sideLabelFa(side: 'LEFT' | 'RIGHT'): string {
  return side === 'LEFT' ? 'دست چپ' : 'دست راست';
}

export const FA = {
  appTitle: 'گل یا پوچ',
  appSubtitle: 'بازی سنتی ایرانی در برابر هوش مصنوعی با بینایی ماشین (فاز ۱)',
  splashStartBtn: 'شروع و راه‌اندازی دوربین',
  splashPrivacyBadge: '۱۰۰٪ پردازش روی دستگاه شما — هیچ تصویر یا ویدیویی ارسال نمی‌شود',
  permissionTitle: 'دسترسی به دوربین و تضمین حریم خصوصی',
  permissionBody:
    'این بازی برای تشخیص حرکت دست‌ها و پیدا کردن گل از دوربین دستگاه شما استفاده می‌کند. تمام پردازش‌ها به‌صورت محلی روی مرورگر شما انجام می‌شود و هیچ فریم تصویر یا ویدیویی هرگز ذخیره یا به سرور ارسال نمی‌شود.',
  permissionChecklist: [
    'پردازش بینایی ماشین کاملاً آفلاین و محلی (Self-Hosted WASM)',
    'بدون ارسال تصویر، بدون ردیاب و بدون درخواست به سرورهای خارجی',
    'تنظیم خودکار دوربین سلفی در موبایل با امکان تغییر دوربین',
  ],
  permissionGrantBtn: 'اجازه دسترسی به دوربین',
  permissionDeniedHint:
    'دسترسی دوربین رد شد یا دوربینی یافت نشد. در تنظیمات مرورگر دسترسی دوربین را فعال کنید و دوباره تلاش کنید.',
  cameraSelectLabel: 'انتخاب دوربین',
  calibrationTitle: 'جادوگر کالیبراسیون دوربین و گل',
  calibrationIntro:
    'برای شروع بازی، هر چهار بررسی زیر باید با موفقیت انجام شود:',
  calCheckLighting: '۱. روشنایی و کنتراست محیط',
  calCheckHands: '۲. تشخیص هر دو دست (حداقل ۱ ثانیه با اطمینان > ۷۰٪)',
  calCheckDistance: '۳. فاصله دست‌ها از دوربین',
  calCheckGolColor: '۴. کالیبراسیون رنگ گل (مهره رنگی داخل دایره)',
  calGolInstruction:
    'گل (مثلاً یک مهره یا توپک کوچک قرمز/سبز) را داخل دایره مرکزی نگه دارید و دکمه ثبت رنگ را بزنید.',
  calSampleGolBtn: 'ثبت رنگ گل',
  calRecalibrateGolBtn: 'ثبت مجدد رنگ گل',
  calLightingShiftWarn: 'نور محیط تغییر کرده است؛ پیشنهاد می‌شود رنگ گل را دوباره ثبت کنید.',
  calCompleteBtn: 'تأیید و ورود به منوی بازی',
  calMirrorToggle: 'نمایش آینه‌ای دوربین',
  calLandmarkToggle: 'نمایش نقاط کلیدی دست',
  menuTitle: 'منوی اصلی (فاز ۱)',
  menuPlayBtn: 'شروع بازی تمرینی (حدس آزمایشی)',
  menuRecalibrateBtn: 'کالیبراسیون مجدد',
  menuSettingsHeading: 'تنظیمات و دسترسی‌پذیری',
  menuPhaseNote:
    'فاز ۱ شامل شالوده بینایی ماشین، تشخیص گل و چرخه کامل بازی است. مدل حدس هوشمند رایانه در فاز ۲ اضافه می‌شود.',
  reducedMotionLabel: 'کاهش انیمیشن‌ها (Reduced Motion)',
  debugToggleLabel: 'نمایش پنل دیباگ (کلید D)',
  showGolPrompt: 'گل را بین شست و اشاره نشان بده',
  golConfirmedBadge: 'گل شناسایی شد ✓',
  hidingIndicator: 'در حال تماشا…',
  hidingInstruction: 'گل را در یکی از مشت‌هایتان پنهان کنید. دست‌ها در کادر بمانند.',
  finishHidingBtn: 'پایان مخفی‌سازی (آماده حدس)',
  guessPhaseTitle: 'حدس آزمایشی رایانه (فاز ۱)',
  guessPlaceholderNote:
    'توجه: در فاز ۱، حدس رایانه به‌صورت آزمایشی (تصادفی) انتخاب می‌شود.',
  proceedToRevealBtn: 'رفتن به مرحله باز کردن دست‌ها',
  revealPrompt: 'هر دو دست را باز کنید و کف دست‌ها را رو به دوربین بگیرید',
  revealWaitingHands: 'در انتظار باز شدن کامل هر دو دست…',
  roundResultComputerWon: 'رایانه درست حدس زد! (+۱ امتیاز رایانه)',
  roundResultHumanWon: 'شما رایانه را فریب دادید! (+۱ امتیاز شما)',
  nextRoundBtn: 'دور بعدی',
  replayVoidBtn: 'تکرار دور',
  resetMatchBtn: 'شروع مجدد مسابقه',
  exitToMenuBtn: 'بازگشت به منو',
  pausedTitle: 'بازی متوقف شد (تب مخفی شده است)',
  pausedBody: 'با بازگشت به صفحه، زمان و امتیازهای شما از همان نقطه ادامه می‌یابد.',
  cameraErrorTitle: 'قطع ارتباط دوربین',
  cameraRecoverBtn: 'تلاش مجدد برای اتصال دوربین',
  cameraRetryPermissionBtn: 'بازگشت به صفحه مجوز دوربین',
  debugTitle: 'پنل دیباگ بینایی و ضبط/بازپخش',
  debugStartRec: 'شروع ضبط (فقط اعداد)',
  debugStopRec: 'توقف و دانلود JSON',
  debugReplaySample: 'بازپخش نمونه آزمایشی (بدون دوربین)',
  debugLoadJson: 'بارگذاری فایل JSON برای بازپخش',
  lowFpsWarning: 'سرعت پردازش پایین است؛ رزولوشن برای روانی بازی کاهش یافت.',
} as const;
