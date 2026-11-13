import { handIndexLabelFa, sideLabelFa, toPersianDigits } from '../app/i18n';
import type { LevelId } from '../game/levels';

export const LLM_CONFIG_STORAGE_KEY = 'gol-ya-pooch:llm-config';
export const LLM_DEFAULT_TIMEOUT_MS = 3000;

export interface LlmCommentaryConfig {
  enabled: boolean;
  endpointUrl: string;
  model: string;
  apiKey: string;
  timeoutMs: number;
}

export function createDefaultLlmConfig(): LlmCommentaryConfig {
  return {
    enabled: false,
    endpointUrl: '',
    model: 'gpt-4o-mini',
    apiKey: '',
    timeoutMs: LLM_DEFAULT_TIMEOUT_MS,
  };
}

export interface LlmStorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

function getSafeStorage(override?: LlmStorageLike): LlmStorageLike | null {
  if (override) return override;
  try {
    if (typeof globalThis.localStorage !== 'undefined') {
      return globalThis.localStorage;
    }
  } catch {
    return null;
  }
  return null;
}

export function loadLlmConfig(storageOverride?: LlmStorageLike): LlmCommentaryConfig {
  const defaults = createDefaultLlmConfig();
  const storage = getSafeStorage(storageOverride);
  if (!storage) return defaults;
  try {
    const raw = storage.getItem(LLM_CONFIG_STORAGE_KEY);
    if (!raw) return defaults;
    const parsed = JSON.parse(raw) as Partial<LlmCommentaryConfig>;
    return {
      enabled: Boolean(parsed.enabled),
      endpointUrl: typeof parsed.endpointUrl === 'string' ? parsed.endpointUrl.trim() : '',
      model:
        typeof parsed.model === 'string' && parsed.model.trim().length > 0
          ? parsed.model.trim()
          : defaults.model,
      apiKey: typeof parsed.apiKey === 'string' ? parsed.apiKey : '',
      timeoutMs:
        typeof parsed.timeoutMs === 'number' &&
        Number.isFinite(parsed.timeoutMs) &&
        parsed.timeoutMs > 0
          ? Math.min(LLM_DEFAULT_TIMEOUT_MS, parsed.timeoutMs)
          : LLM_DEFAULT_TIMEOUT_MS,
    };
  } catch {
    return defaults;
  }
}

export function saveLlmConfig(
  config: LlmCommentaryConfig,
  storageOverride?: LlmStorageLike
): LlmCommentaryConfig {
  const normalized: LlmCommentaryConfig = {
    enabled: Boolean(config.enabled),
    endpointUrl: config.endpointUrl.trim(),
    model: config.model.trim() || 'gpt-4o-mini',
    apiKey: config.apiKey,
    timeoutMs: Math.max(
      200,
      Math.min(LLM_DEFAULT_TIMEOUT_MS, Math.round(config.timeoutMs || LLM_DEFAULT_TIMEOUT_MS))
    ),
  };
  const storage = getSafeStorage(storageOverride);
  try {
    storage?.setItem(LLM_CONFIG_STORAGE_KEY, JSON.stringify(normalized));
  } catch {
    // Ignore storage quota/access errors
  }
  return normalized;
}

export function clearLlmApiKey(storageOverride?: LlmStorageLike): LlmCommentaryConfig {
  const current = loadLlmConfig(storageOverride);
  return saveLlmConfig({ ...current, apiKey: '', enabled: false }, storageOverride);
}

export type CommentaryEvent =
  | {
      kind: 'LEVEL_INTRO';
      levelId: LevelId;
      levelNameFa: string;
      ruleSetNameFa: string;
    }
  | {
      kind: 'ROUND_RESULT';
      levelId: LevelId;
      levelNameFa: string;
      roundNumber: number;
      humanScore: number;
      computerScore: number;
      targetScore: number;
      winner: 'HUMAN' | 'COMPUTER';
      role: 'HUMAN_HIDES' | 'COMPUTER_HIDES';
      revealedHand?: 'LEFT' | 'RIGHT' | null;
      computerGuess?: 'LEFT' | 'RIGHT' | null;
      computerGuessProb?: number | null;
      handsCount?: number;
      secretHandIndex?: number | null;
      explanationFa?: string | null;
      timelineEvents?: readonly string[];
    }
  | {
      kind: 'MATCH_COMPLETE';
      levelId: LevelId;
      levelNameFa: string;
      winner: 'HUMAN' | 'COMPUTER';
      humanScore: number;
      computerScore: number;
      unlockedNextLevel?: boolean;
    };

export interface CommentaryResponse {
  textFa: string;
  source: 'template' | 'llm';
  latencyMs: number;
  fallbackReason?: string;
}

const LEVEL_DIALOGUE_FA: Record<LevelId, string> = {
  1: 'به میدان گل یا پوچ خوش آمدید! در سطح تازه‌کار فرصت کافی برای مخفی‌سازی دارید.',
  2: 'در سطح آماتور سرعت دست‌ها بیشتر می‌شود؛ مراقب حرکت‌های فریبنده باشید.',
  3: 'سطح حرفه‌ای آغاز شد! فراموش نکنید در حین پنهان‌کاری حتماً دست‌ها یک بار تماس داشته باشند.',
  4: 'نوبت بازی تیمی سنتی است! با «پوچ» دست‌های خالی را کنار بزنید یا با «گل» امتیاز طلایی بگیرید.',
  5: 'مبارزه نهایی استاد گل‌گیر! سری بهترین از ۳ مسابقه با زمان کوتاه و هوش مصنوعی تطبیقی.',
};

export function generateTemplateCommentary(event: CommentaryEvent): string {
  if (event.kind === 'LEVEL_INTRO') {
    return LEVEL_DIALOGUE_FA[event.levelId];
  }

  if (event.kind === 'MATCH_COMPLETE') {
    const scoreStr = `${toPersianDigits(event.humanScore)} بر ${toPersianDigits(event.computerScore)}`;
    if (event.winner === 'HUMAN') {
      return event.unlockedNextLevel
        ? `آفرین! با نتیجه ${scoreStr} در سطح ${event.levelNameFa} پیروز شدید و سطح بعدی را باز کردید.`
        : `تبریک! مسابقه سطح ${event.levelNameFa} را با امتیاز ${scoreStr} به سود خود تمام کردید.`;
    }
    return `این بار رایانه با نتیجه ${scoreStr} در سطح ${event.levelNameFa} برنده شد؛ یک دست دیگر انتقام بگیرید!`;
  }

  // ROUND_RESULT
  const scoreLine = `امتیاز فعلی: شما ${toPersianDigits(event.humanScore)} — رایانه ${toPersianDigits(event.computerScore)}`;
  if (event.role === 'HUMAN_HIDES') {
    const probPct = Math.round((event.computerGuessProb ?? 0.5) * 100);
    const revStr = event.revealedHand ? sideLabelFa(event.revealedHand) : 'دست پنهان';
    if (event.winner === 'COMPUTER') {
      return `چشم تیزبین رایانه با اطمینان ${toPersianDigits(probPct)}٪ گل را در ${revStr} شکار کرد! ${event.explanationFa ?? ''} (${scoreLine})`.trim();
    }
    return `چه فریب تمیزی! گل در ${revStr} بود و حدس رایانه (${toPersianDigits(probPct)}٪) به پوچ خورد. (${scoreLine})`;
  }

  const secretLabel =
    event.secretHandIndex !== null && event.secretHandIndex !== undefined
      ? handIndexLabelFa(event.secretHandIndex, event.handsCount ?? 2)
      : 'دست پنهان';
  if (event.winner === 'HUMAN') {
    return `حدس عالی! گل واقعاً در ${secretLabel} پنهان شده بود و صحت قرعه هم رمزنگاری‌شده تأیید شد. (${scoreLine})`;
  }
  return `این بار حقه رایانه گرفت و گل در ${secretLabel} بود؛ امضای SHA-256 نشان می‌دهد انتخاب از قبل ثابت بوده است. (${scoreLine})`;
}

const FORBIDDEN_MEDIA_KEYS = new Set([
  'image',
  'images',
  'image_url',
  'video',
  'videos',
  'frame',
  'frames',
  'canvas',
  'blob',
  'pixels',
  'rgba',
]);

const FORBIDDEN_DATA_URI_RE = /data:(image|video|audio)\//i;

/**
 * Enforces that no image, video, binary, or multimodal payload can ever leave the device
 * via the optional LLM commentary module.
 */
export function assertTextOnlyLlmPayload(value: unknown, path = 'payload'): void {
  if (value === null || value === undefined) return;
  if (typeof value === 'string') {
    if (FORBIDDEN_DATA_URI_RE.test(value)) {
      throw new Error(`LLM payload rejected forbidden media URI at ${path}`);
    }
    return;
  }
  if (typeof value === 'number' || typeof value === 'boolean') return;

  if (ArrayBuffer.isView(value) || value instanceof ArrayBuffer) {
    throw new Error(`LLM payload rejected binary buffer at ${path}`);
  }

  if (Array.isArray(value)) {
    for (let i = 0; i < value.length; i++) {
      assertTextOnlyLlmPayload(value[i], `${path}[${i}]`);
    }
    return;
  }

  if (typeof value === 'object') {
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      if (FORBIDDEN_MEDIA_KEYS.has(k.toLowerCase())) {
        throw new Error(`LLM payload rejected forbidden media key "${k}" at ${path}`);
      }
      if (k === 'type' && typeof v === 'string' && v.toLowerCase().includes('image')) {
        throw new Error(`LLM payload rejected multimodal image content at ${path}`);
      }
      assertTextOnlyLlmPayload(v, `${path}.${k}`);
    }
  }
}

export function buildTextOnlyPrompt(event: CommentaryEvent): {
  system: string;
  user: string;
} {
  const system =
    'تو مجری خوش‌سخن و محترمانه بازی سنتی ایرانی «گل یا پوچ» هستی. فقط یک یا دو جمله کوتاه، گرم و فارسی بنویس. فقط از اطلاعات متنی داده‌شده استفاده کن.';

  if (event.kind === 'LEVEL_INTRO') {
    return {
      system,
      user: `شروع سطح ${event.levelId} (${event.levelNameFa}) با قوانین ${event.ruleSetNameFa}. یک جمله خوش‌آمدگویی و انگیزشی کوتاه بگو.`,
    };
  }

  if (event.kind === 'MATCH_COMPLETE') {
    return {
      system,
      user: `پایان مسابقه در سطح ${event.levelNameFa}. برنده: ${event.winner === 'HUMAN' ? 'بازیکن' : 'رایانه'}. امتیاز نهایی: بازیکن ${event.humanScore} - رایانه ${event.computerScore}. یک جمله تبریک یا تشویق کوتاه بگو.`,
    };
  }

  const timelineText =
    event.timelineEvents && event.timelineEvents.length > 0
      ? `رویدادها: ${event.timelineEvents.slice(0, 3).join(' | ')}`
      : 'بدون رویداد خاص';

  return {
    system,
    user: `دور ${event.roundNumber} در سطح ${event.levelNameFa}. نقش: ${
      event.role === 'HUMAN_HIDES' ? 'بازیکن پنهان کرد' : 'رایانه پنهان کرد'
    }. برنده دور: ${event.winner === 'HUMAN' ? 'بازیکن' : 'رایانه'}. امتیاز: بازیکن ${event.humanScore} - رایانه ${event.computerScore} (هدف ${event.targetScore}). ${
      event.explanationFa ?? ''
    } ${timelineText}`,
  };
}

export interface FetchLike {
  (input: string, init?: RequestInit): Promise<Response>;
}

export async function generateHostCommentary(
  event: CommentaryEvent,
  config: LlmCommentaryConfig,
  options: { fetchImpl?: FetchLike } = {}
): Promise<CommentaryResponse> {
  const t0 = performance.now();
  const templateText = generateTemplateCommentary(event);

  if (!config.enabled) {
    return {
      textFa: templateText,
      source: 'template',
      latencyMs: performance.now() - t0,
      fallbackReason: 'disabled',
    };
  }

  const endpoint = config.endpointUrl.trim();
  if (!endpoint) {
    return {
      textFa: templateText,
      source: 'template',
      latencyMs: performance.now() - t0,
      fallbackReason: 'missing-endpoint',
    };
  }

  const fetchFn: FetchLike | undefined =
    options.fetchImpl ??
    (typeof globalThis.fetch === 'function' ? globalThis.fetch.bind(globalThis) : undefined);
  if (!fetchFn) {
    return {
      textFa: templateText,
      source: 'template',
      latencyMs: performance.now() - t0,
      fallbackReason: 'fetch-unavailable',
    };
  }

  const prompt = buildTextOnlyPrompt(event);
  const requestBody = {
    model: config.model || 'gpt-4o-mini',
    messages: [
      { role: 'system', content: prompt.system },
      { role: 'user', content: prompt.user },
    ],
    temperature: 0.7,
    max_tokens: 140,
  };

  try {
    assertTextOnlyLlmPayload(requestBody);
  } catch (err) {
    return {
      textFa: templateText,
      source: 'template',
      latencyMs: performance.now() - t0,
      fallbackReason: err instanceof Error ? err.message : 'payload-rejected',
    };
  }

  const timeoutMs = Math.max(
    100,
    Math.min(LLM_DEFAULT_TIMEOUT_MS, config.timeoutMs || LLM_DEFAULT_TIMEOUT_MS)
  );
  const controller = new AbortController();
  const timerId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (config.apiKey.trim().length > 0) {
      headers.Authorization = `Bearer ${config.apiKey.trim()}`;
    }

    const res = await fetchFn(endpoint, {
      method: 'POST',
      headers,
      body: JSON.stringify(requestBody),
      signal: controller.signal,
    });

    if (!res.ok) {
      return {
        textFa: templateText,
        source: 'template',
        latencyMs: performance.now() - t0,
        fallbackReason: `http-${res.status}`,
      };
    }

    const json = (await res.json()) as {
      choices?: Array<{ message?: { content?: unknown } }>;
    };
    const content = json?.choices?.[0]?.message?.content;
    if (typeof content !== 'string' || content.trim().length === 0) {
      return {
        textFa: templateText,
        source: 'template',
        latencyMs: performance.now() - t0,
        fallbackReason: 'empty-response',
      };
    }

    return {
      textFa: content.trim().slice(0, 360),
      source: 'llm',
      latencyMs: performance.now() - t0,
    };
  } catch (err) {
    const reason =
      err instanceof Error && err.name === 'AbortError'
        ? 'timeout-3000ms'
        : 'network-error';
    return {
      textFa: templateText,
      source: 'template',
      latencyMs: performance.now() - t0,
      fallbackReason: reason,
    };
  } finally {
    clearTimeout(timerId);
  }
}
