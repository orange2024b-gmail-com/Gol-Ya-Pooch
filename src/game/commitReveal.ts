export interface CommitPublicInfo {
  commitHex: string;
  fingerprint: string;
  handsCount: number;
  createdAtMs: number;
}

export interface CommitSecretReveal {
  handIndex: number;
  choice: string;
  saltHex: string;
  preimage: string;
}

export interface CommitEnvelope {
  publicInfo: CommitPublicInfo;
  secret: CommitSecretReveal;
}

export interface CommitVerificationResult {
  valid: boolean;
  expectedCommitHex: string;
  recomputedCommitHex: string;
  choice: string;
  saltHex: string;
  preimage: string;
  statusFa: string;
  reason?: string;
}

export const COMMIT_REVEAL_EXPLANATION_FA = {
  title: 'این چطور کار می‌کند؟ (تضمین عدالت رمزنگاری)',
  summary:
    'قبل از اینکه شما حدس بزنید، رایانه دستِ دارای گل را انتخاب می‌کند و آن را با یک کلید تصادفی ۱۶ بایتی (Salt) ترکیب کرده و خلاصه رمزنگاری SHA-256 می‌سازد. اثر انگشت این تعهد روی صفحه نمایش داده می‌شود تا مطمئن شوید رایانه پس از حدس شما نمی‌تواند جای گل را عوض کند.',
  steps: [
    '۱. پیش از حدس شما: رایانه انتخاب خود را با یک مقدار تصادفی ۱۶ بایتی ترکیب و هش SHA-256 تولید می‌کند.',
    '۲. نمایش اثر انگشت تعهد: بخش کوتاه‌شده هش (Fingerprint) روی صفحه قفل می‌شود.',
    '۳. پس از حدس شما: انتخاب واقعی (Choice) و کلید تصادفی (Salt) آشکار می‌شوند و مرورگر شما همان لحظه هش SHA-256 را دوباره محاسبه می‌کند.',
    '۴. تأیید نهایی: اگر هش محاسبه‌شده با تعهد اولیه دقیقاً برابر باشد، نشان «✓ تأیید شد» نمایش داده می‌شود.',
  ],
} as const;

const HEX_CHARS_64 = /^[0-9a-f]{64}$/i;
const HEX_CHARS_32 = /^[0-9a-f]{32}$/i;

export function bytesToHex(bytes: Uint8Array): string {
  let out = '';
  for (let i = 0; i < bytes.length; i++) {
    out += bytes[i]!.toString(16).padStart(2, '0');
  }
  return out;
}

export function generateSaltHex(byteLength = 16): string {
  const bytes = new Uint8Array(byteLength);
  if (
    typeof globalThis.crypto !== 'undefined' &&
    typeof globalThis.crypto.getRandomValues === 'function'
  ) {
    globalThis.crypto.getRandomValues(bytes);
  } else {
    throw new Error('Web Crypto getRandomValues is required for commit-reveal salt generation');
  }
  return bytesToHex(bytes);
}

export function formatChoiceToken(handIndex: number, handsCount: number): string {
  if (!Number.isInteger(handIndex) || handIndex < 0 || handIndex >= handsCount) {
    throw new RangeError(`handIndex ${handIndex} out of bounds for handsCount=${handsCount}`);
  }
  if (handsCount === 2) {
    return handIndex === 0 ? 'LEFT' : 'RIGHT';
  }
  return `HAND_${handIndex + 1}`;
}

export function buildCommitPreimage(choice: string, saltHex: string): string {
  return `${choice}:${saltHex}`;
}

export async function sha256Hex(input: string): Promise<string> {
  if (
    typeof globalThis.crypto === 'undefined' ||
    typeof globalThis.crypto.subtle === 'undefined'
  ) {
    throw new Error('Web Crypto subtle.digest is required for SHA-256 commitments');
  }
  const encoded = new TextEncoder().encode(input);
  const digest = await globalThis.crypto.subtle.digest('SHA-256', encoded);
  return bytesToHex(new Uint8Array(digest));
}

export function formatCommitFingerprint(commitHex: string, length = 12): string {
  const clean = commitHex.toLowerCase();
  if (clean.length <= length) return clean;
  return `${clean.slice(0, length)}…`;
}

export async function createCommitEnvelope(
  handIndex: number,
  handsCount: number,
  customSaltBytes?: Uint8Array
): Promise<CommitEnvelope> {
  if (customSaltBytes && customSaltBytes.length !== 16) {
    throw new RangeError('Salt must be exactly 16 bytes');
  }
  const choice = formatChoiceToken(handIndex, handsCount);
  const saltHex = customSaltBytes ? bytesToHex(customSaltBytes) : generateSaltHex(16);
  const preimage = buildCommitPreimage(choice, saltHex);
  const commitHex = await sha256Hex(preimage);

  return {
    publicInfo: {
      commitHex,
      fingerprint: formatCommitFingerprint(commitHex, 12),
      handsCount,
      createdAtMs: Date.now(),
    },
    secret: {
      handIndex,
      choice,
      saltHex,
      preimage,
    },
  };
}

export async function verifyCommitReveal(
  expectedCommitHex: string,
  choice: string,
  saltHex: string
): Promise<CommitVerificationResult> {
  const normalizedExpected = expectedCommitHex.trim().toLowerCase();
  const normalizedChoice = choice.trim();
  const normalizedSalt = saltHex.trim().toLowerCase();
  const preimage = buildCommitPreimage(normalizedChoice, normalizedSalt);

  if (!HEX_CHARS_64.test(normalizedExpected)) {
    return {
      valid: false,
      expectedCommitHex: normalizedExpected,
      recomputedCommitHex: '',
      choice: normalizedChoice,
      saltHex: normalizedSalt,
      preimage,
      statusFa: '✗ خطا در قالب تعهد اولیه',
      reason: 'INVALID_COMMIT_FORMAT',
    };
  }

  if (normalizedChoice.length === 0) {
    return {
      valid: false,
      expectedCommitHex: normalizedExpected,
      recomputedCommitHex: '',
      choice: normalizedChoice,
      saltHex: normalizedSalt,
      preimage,
      statusFa: '✗ انتخاب پنهان نامعتبر است',
      reason: 'EMPTY_CHOICE',
    };
  }

  if (!HEX_CHARS_32.test(normalizedSalt)) {
    return {
      valid: false,
      expectedCommitHex: normalizedExpected,
      recomputedCommitHex: '',
      choice: normalizedChoice,
      saltHex: normalizedSalt,
      preimage,
      statusFa: '✗ کلید تصادفی (Salt) نامعتبر است',
      reason: 'INVALID_SALT_FORMAT',
    };
  }

  const recomputedCommitHex = await sha256Hex(preimage);
  const valid = recomputedCommitHex === normalizedExpected;

  return {
    valid,
    expectedCommitHex: normalizedExpected,
    recomputedCommitHex,
    choice: normalizedChoice,
    saltHex: normalizedSalt,
    preimage,
    statusFa: valid ? '✓ تأیید شد' : '✗ عدم تطابق تعهد رمزنگاری!',
    ...(valid ? {} : { reason: 'HASH_MISMATCH' }),
  };
}
