import { describe, expect, it, vi } from 'vitest';
import {
  LLM_DEFAULT_TIMEOUT_MS,
  assertTextOnlyLlmPayload,
  buildTextOnlyPrompt,
  clearLlmApiKey,
  createDefaultLlmConfig,
  generateHostCommentary,
  generateTemplateCommentary,
  loadLlmConfig,
  saveLlmConfig,
  type CommentaryEvent,
  type LlmStorageLike,
} from '../../src/llm/commentaryClient';

function createMemoryStorage(): LlmStorageLike {
  const map = new Map<string, string>();
  return {
    getItem: (k) => map.get(k) ?? null,
    setItem: (k, v) => {
      map.set(k, v);
    },
    removeItem: (k) => {
      map.delete(k);
    },
  };
}

const sampleRoundEvent: CommentaryEvent = {
  kind: 'ROUND_RESULT',
  levelId: 3,
  levelNameFa: 'حرفه‌ای',
  roundNumber: 2,
  humanScore: 1,
  computerScore: 2,
  targetScore: 7,
  winner: 'COMPUTER',
  role: 'HUMAN_HIDES',
  revealedHand: 'RIGHT',
  computerGuess: 'RIGHT',
  computerGuessProb: 0.87,
  explanationFa: 'در ثانیه ۱.۸ دست‌ها به هم رسیدند و دست چپ کمی باز شد.',
  timelineEvents: ['ثانیه ۱.۸: دست‌ها به هم رسیدند'],
};

describe('Optional LLM Commentary Client (src/llm/)', () => {
  it('defaults to OFF and makes zero fetch calls when disabled', async () => {
    const def = createDefaultLlmConfig();
    expect(def.enabled).toBe(false);
    expect(def.apiKey).toBe('');
    expect(def.timeoutMs).toBe(LLM_DEFAULT_TIMEOUT_MS);

    const fetchMock = vi.fn();
    const res = await generateHostCommentary(sampleRoundEvent, def, {
      fetchImpl: fetchMock,
    });

    expect(fetchMock).not.toHaveBeenCalled();
    expect(res.source).toBe('template');
    expect(res.fallbackReason).toBe('disabled');
    expect(res.textFa).toContain('۸۷٪');
  });

  it('persists config only in localStorage and clears API key cleanly', () => {
    const storage = createMemoryStorage();
    expect(loadLlmConfig(storage).enabled).toBe(false);

    saveLlmConfig(
      {
        enabled: true,
        endpointUrl: 'https://example.invalid/v1/chat/completions',
        model: 'test-model',
        apiKey: 'secret-user-key',
        timeoutMs: 2500,
      },
      storage
    );

    const loaded = loadLlmConfig(storage);
    expect(loaded.enabled).toBe(true);
    expect(loaded.apiKey).toBe('secret-user-key');
    expect(loaded.model).toBe('test-model');
    expect(loaded.timeoutMs).toBe(2500);

    const cleared = clearLlmApiKey(storage);
    expect(cleared.enabled).toBe(false);
    expect(cleared.apiKey).toBe('');
  });

  it('enforces text-only payload guard and rejects images, video, and binary buffers', () => {
    const prompt = buildTextOnlyPrompt(sampleRoundEvent);
    expect(() =>
      assertTextOnlyLlmPayload({
        model: 'gpt-4o-mini',
        messages: [
          { role: 'system', content: prompt.system },
          { role: 'user', content: prompt.user },
        ],
      })
    ).not.toThrow();

    expect(() =>
      assertTextOnlyLlmPayload({ image_url: 'https://example.com/frame.png' })
    ).toThrow(/forbidden media key/i);

    expect(() =>
      assertTextOnlyLlmPayload({
        messages: [{ role: 'user', content: 'data:image/png;base64,AAAA' }],
      })
    ).toThrow(/forbidden media URI/i);

    expect(() =>
      assertTextOnlyLlmPayload({
        messages: [{ role: 'user', content: [{ type: 'image_url', text: 'x' }] }],
      })
    ).toThrow(/multimodal image/i);

    expect(() =>
      assertTextOnlyLlmPayload({
        extra: new Uint8Array([1, 2, 3]),
      })
    ).toThrow(/binary buffer/i);
  });

  it('aborts via AbortController and falls back silently on timeout or network/HTTP errors', async () => {
    const slowFetch = (_input: string, init?: RequestInit): Promise<Response> =>
      new Promise((_resolve, reject) => {
        init?.signal?.addEventListener('abort', () => {
          const err = new Error('Aborted');
          err.name = 'AbortError';
          reject(err);
        });
      });

    const timeoutRes = await generateHostCommentary(
      sampleRoundEvent,
      {
        enabled: true,
        endpointUrl: 'http://127.0.0.1:9999/v1/chat/completions',
        model: 'test',
        apiKey: 'user-key',
        timeoutMs: 120,
      },
      { fetchImpl: slowFetch }
    );

    expect(timeoutRes.source).toBe('template');
    expect(timeoutRes.fallbackReason).toBe('timeout-3000ms');
    expect(timeoutRes.textFa).toBe(generateTemplateCommentary(sampleRoundEvent));

    const errorRes = await generateHostCommentary(
      sampleRoundEvent,
      {
        enabled: true,
        endpointUrl: 'http://127.0.0.1:9999/v1/chat/completions',
        model: 'test',
        apiKey: '',
        timeoutMs: 3000,
      },
      {
        fetchImpl: () => Promise.resolve(new Response('Server Error', { status: 500 })),
      }
    );
    expect(errorRes.source).toBe('template');
    expect(errorRes.fallbackReason).toBe('http-500');
  });

  it('returns live LLM commentary when enabled and OpenAI-compatible endpoint responds in time', async () => {
    let capturedBody = '';
    const okFetch = (_input: string, init?: RequestInit): Promise<Response> => {
      capturedBody = typeof init?.body === 'string' ? init.body : '';
      return Promise.resolve(
        new Response(
          JSON.stringify({
            choices: [{ message: { content: 'چه بازی تماشایی! رایانه با دقت دست راست را حدس زد.' } }],
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } }
        )
      );
    };

    const res = await generateHostCommentary(
      sampleRoundEvent,
      {
        enabled: true,
        endpointUrl: 'http://127.0.0.1:8787/v1/chat/completions',
        model: 'local-fa',
        apiKey: 'user-token',
        timeoutMs: 3000,
      },
      { fetchImpl: okFetch }
    );

    expect(res.source).toBe('llm');
    expect(res.textFa).toContain('چه بازی تماشایی');
    expect(capturedBody).not.toMatch(/image|video|canvas/i);
  });
});
