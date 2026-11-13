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

export function handIndexLabelFa(handIndex: number, handsCount: number): string {
  if (handsCount === 2) {
    return handIndex === 0 ? 'دست چپ' : 'دست راست';
  }
  return `دست ${toPersianDigits(handIndex + 1)}`;
}

export const FA = {
  appTitle: 'گل یا پوچ',
  appSubtitle: 'بازی سنتی ایرانی در برابر هوش مصنوعی با بینایی ماشین (فاز ۴)',
  splashStartBtn: 'شروع و راه‌اندازی دوربین',
  splashPracticeBtn: 'بازی تمرینی با پنهان‌کاری رایانه (بدون دوربین)',
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
  calibrationIntro: 'برای شروع بازی، هر چهار بررسی زیر باید با موفقیت انجام شود:',
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
  menuTitle: 'منوی اصلی بازی',
  menuPlayBtn: 'شروع مسابقه (سطح انتخاب‌شده)',
  menuLevelsBtn: 'انتخاب سطح (۱ تا ۵)',
  menuStatsBtn: 'آمار و پیشرفت',
  menuRulesBtn: 'قوانین و حالت تیمی',
  menuConsentBtn: 'ثبت داده و حریم خصوصی',
  menuRecalibrateBtn: 'کالیبراسیون مجدد',
  menuSettingsHeading: 'تنظیمات و دسترسی‌پذیری',
  menuPhaseNote:
    'فاز ۴ فعال است: مدل زمانی ONNX Runtime Web (کاملاً محلی)، ترکیب کالیبره‌شده با فیلتر باور و ثبت داده عددی با رضایت کاربر.',
  targetScoreLabel: 'امتیاز هدف برای برد مسابقه',
  resetProfileMemoryBtn: 'پاک‌سازی حافظه رفتاری و آمار',
  reducedMotionLabel: 'کاهش انیمیشن‌ها (Reduced Motion)',
  debugToggleLabel: 'نمایش پنل دیباگ (کلید D)',
  showGolPrompt: 'گل را بین شست و اشاره نشان بده',
  golConfirmedBadge: 'گل شناسایی شد ✓',
  hidingIndicator: 'در حال تماشا…',
  hidingInstruction: 'گل را در یکی از مشت‌هایتان پنهان کنید. دست‌ها در کادر بمانند.',
  finishHidingBtn: 'پایان مخفی‌سازی (آماده حدس)',
  guessPhaseTitle: 'حدس هوشمند رایانه',
  proceedToRevealBtn: 'رفتن به مرحله باز کردن دست‌ها',
  revealPrompt: 'هر دو دست را باز کنید و کف دست‌ها را رو به دوربین بگیرید',
  revealWaitingHands: 'در انتظار باز شدن کامل هر دو دست…',
  roundResultComputerWon: 'رایانه این دور را برد!',
  roundResultHumanWon: 'شما این دور را بردید!',
  matchWinnerHuman: '🏆 تبریک! شما برنده مسابقه شدید!',
  matchWinnerComputer: '🤖 رایانه برنده این مسابقه شد!',
  nextRoundBtn: 'دور بعدی',
  nextSeriesMatchBtn: 'مسابقه بعدی در سری ۳ تایی',
  replayVoidBtn: 'تکرار دور',
  resetMatchBtn: 'شروع مسابقه جدید',
  exitToMenuBtn: 'بازگشت به منو',
  pausedTitle: 'بازی متوقف شد (تب مخفی شده است)',
  pausedBody: 'با بازگشت به صفحه، زمان و امتیازهای شما از همان نقطه ادامه می‌یابد.',
  cameraErrorTitle: 'قطع ارتباط دوربین',
  cameraRecoverBtn: 'تلاش مجدد برای اتصال دوربین',
  cameraRetryPermissionBtn: 'بازگشت به صفحه مجوز دوربین',
  debugTitle: 'پنل دیباگ بینایی، فیلتر باور و ضبط/بازپخش',
  debugStartRec: 'شروع ضبط (فقط اعداد)',
  debugStopRec: 'توقف و دانلود JSON',
  debugReplaySample: 'بازپخش نمونه آزمایشی (بدون دوربین)',
  debugReplayTransfer: 'بازپخش نمونه انتقال (فاز ۲)',
  debugLoadJson: 'بارگذاری فایل JSON برای بازپخش',
  lowFpsWarning: 'سرعت پردازش پایین است؛ رزولوشن برای روانی بازی کاهش یافت.',
  computerHidesTitle: 'نوبت پنهان‌کاری رایانه',
  computerHidesDesc: 'رایانه با انیمیشن و حرکات فریبنده گل را پنهان می‌کند…',
  computerHidesReducedDesc: 'حالت کاهش انیمیشن فعال است؛ دست‌ها بدون حرکت سریع آماده حدس می‌شوند.',
  humanGuessTitle: 'نوبت حدس شما',
  humanGuessPromptClassic: 'روی دستی که فکر می‌کنید گل در آن است ضربه بزنید.',
  humanGuessPromptTeam: 'حالت تیمی: روی دست‌های مشکوک به خالی «پوچ» بزنید یا با «گل» جای اصلی را اعلام کنید.',
  poochActionBtn: 'پوچ',
  golActionBtn: 'گل',
  commitLabel: 'اثر انگشت تعهد رمزنگاری (پیش از حدس):',
  commitVerifying: 'در حال بررسی امضای SHA-256…',
  howItWorksBtn: 'این چطور کار می‌کند؟',
  levelSelectTitle: 'انتخاب سطح (۱ تا ۵)',
  statsTitle: 'آمار و پیشرفت بازیکن',
  rulesTitle: 'قوانین بازی و تنظیمات حالت تیمی',
  consentTitle: 'رضایت‌نامه اختیاری ثبت داده عددی و حریم خصوصی',
  consentDefaultOffBadge: 'پیش‌فرض خاموش (بدون رضایت هیچ داده‌ای ذخیره نمی‌شود)',
  consentActiveBadge: 'ثبت محلی داده عددی فعال است ✓',
  consentExplanation:
    'با فعال‌سازی این گزینه، تنها ویژگی‌های عددی حرکت دست، خط زمانی رویدادها، ردپای باور، حدس نهایی، نتیجه تأییدشده و زمان‌بندی‌ها به‌صورت محلی در IndexedDB مرورگر شما ذخیره می‌شوند. هیچ تصویر، عکس یا ویدیویی هرگز ذخیره، ثبت یا ارسال نمی‌شود.',
  consentChecklist: [
    'پیش‌فرض خاموش است: تا زمانی که خودتان فعال نکنید، هیچ دور بازی ثبت نمی‌شود.',
    'فقط اعداد و رویدادها: مختصات نرمال‌شده، خمش انگشتان، تماس‌ها و زمان‌ها ذخیره می‌شوند.',
    'هرگز تصویر یا ویدیو ذخیره نمی‌شود (کنترل سخت‌گیرانه در زمان اجرا و خروجی JSON).',
    'داده‌ها فقط روی دستگاه شما می‌مانند و هر زمان می‌توانید خروجی JSON بگیرید یا پاک کنید.',
  ],
  consentToggleLabel: 'اجازه ذخیره محلی ویژگی‌های عددی دورها برای بهبود مدل',
  exportLogsBtn: 'خروجی JSON دورهای ثبت‌شده',
  importLogsBtn: 'وارد کردن فایل JSON',
  clearLogsBtn: 'پاک‌سازی لاگ‌های ثبت‌شده',
} as const;
