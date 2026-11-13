import { expect, test } from '@playwright/test';

test.describe('Phase 5 E2E: Branding, Vazirmatn, Audio Mute, LLM Fallback, Rules, Benchmark & Zero External Requests', () => {
  test('verifies custom SVG logo, Vazirmatn font, favicon, audio mute persistence, rules book, LLM silent fallback, and zero external requests', async ({
    page,
  }) => {
    const externalRequests: string[] = [];
    const pageErrors: string[] = [];

    page.on('pageerror', (err) => pageErrors.push(err.message));
    page.on('request', (req) => {
      const url = req.url();
      if (
        !url.startsWith('http://127.0.0.1:4173') &&
        !url.startsWith('data:') &&
        !url.startsWith('blob:')
      ) {
        externalRequests.push(url);
      }
    });

    // Mock unreachable/503 local LLM endpoint on same origin so we test fallback without touching external networks
    await page.route('**/__mock_llm', (route) =>
      route.fulfill({ status: 503, body: 'LLM unreachable' })
    );

    await page.goto('/');
    await expect(page.getByTestId('brand-logo')).toBeVisible();
    await expect(page.getByTestId('hero-brand-logo')).toBeVisible();
    await expect(page.getByTestId('brand-logo').locator('svg')).toBeVisible();

    // Verify favicon.svg & Vazirmatn CSS are self-hosted and loaded
    const faviconRes = await page.request.get('/favicon.svg');
    expect(faviconRes.status()).toBe(200);
    const fontCssRes = await page.request.get('/fonts/vazirmatn.css');
    expect(fontCssRes.status()).toBe(200);

    const fontLoaded = await page.evaluate(async () => {
      await document.fonts.ready;
      return Array.from(document.fonts).some((f) => f.family.includes('Vazirmatn'));
    });
    expect(fontLoaded).toBe(true);

    // Toggle audio mute in topbar and verify persistence across reload
    await expect(page.getByTestId('topbar-mute-btn')).toContainText('صدا');
    await page.getByTestId('topbar-mute-btn').click();
    await expect(page.getByTestId('topbar-mute-btn')).toContainText('بی‌صدا');
    await page.reload();
    await expect(page.getByTestId('topbar-mute-btn')).toContainText('بی‌صدا');
    await page.getByTestId('topbar-mute-btn').click();
    await expect(page.getByTestId('topbar-mute-btn')).toContainText('🔊 صدا');

    // Enter Menu -> verify Rules screen includes complete Persian rules
    await page.getByTestId('splash-practice-btn').click();
    await page.getByTestId('menu-rules-btn').click();
    await expect(page.getByTestId('rules-book-sections')).toBeVisible();
    await expect(page.getByTestId('rules-book-sections')).toContainText('آماده‌سازی و کالیبراسیون');
    await page.getByRole('button', { name: 'بازگشت به منو' }).click();

    // Verify Optional LLM Commentary is OFF by default, test silent fallback when enabled
    await expect(page.getByTestId('llm-settings-panel')).toBeVisible();
    await expect(page.getByTestId('llm-enabled-toggle')).not.toBeChecked();

    await page.getByTestId('llm-enabled-toggle').check();
    await page.getByTestId('llm-endpoint-input').fill('http://127.0.0.1:4173/__mock_llm');
    await page.getByTestId('llm-test-btn').click();
    await expect(page.getByTestId('llm-status-notice')).toContainText('بازگشت ایمن به مجری آماده', {
      timeout: 8_000,
    });

    // Play a round in practice mode and confirm host commentary falls back cleanly on ROUND_RESULT
    await page.getByTestId('menu-play-btn').click();
    await expect(page.getByTestId('hand-guess-btn-0')).toBeVisible({ timeout: 10_000 });
    await page.getByTestId('hand-guess-btn-0').click();
    await expect(page.getByTestId('host-commentary-box')).toBeVisible({ timeout: 8_000 });
    await expect(page.getByTestId('host-commentary-box')).toContainText('گزارش مجری');

    // Clean up LLM key & disable
    await page.getByRole('button', { name: 'بازگشت به منو' }).click();
    await page.getByTestId('llm-clear-key-btn').click();
    await expect(page.getByTestId('llm-enabled-toggle')).not.toBeChecked();

    expect(externalRequests).toEqual([]);
    expect(pageErrors).toEqual([]);
  });

  test('runs in-app Performance Benchmark via ?benchmark=1 and reports real FPS, feature, HMM, and ONNX timings', async ({
    page,
  }) => {
    const externalRequests: string[] = [];
    const pageErrors: string[] = [];

    page.on('pageerror', (err) => pageErrors.push(err.message));
    page.on('request', (req) => {
      const url = req.url();
      if (
        !url.startsWith('http://127.0.0.1:4173') &&
        !url.startsWith('data:') &&
        !url.startsWith('blob:')
      ) {
        externalRequests.push(url);
      }
    });

    await page.goto('/?benchmark=1');
    await expect(page.getByTestId('benchmark-screen')).toBeVisible();
    await expect(page.getByTestId('benchmark-results')).toBeVisible({ timeout: 20_000 });
    await expect(page.getByTestId('benchmark-results')).toContainText('FPS');
    await expect(page.getByTestId('benchmark-results')).toContainText('ONNX');
    await expect(page.getByTestId('benchmark-meta')).toContainText('ort=wasm');

    const metaText = await page.getByTestId('benchmark-meta').innerText();
    const resultsText = await page.getByTestId('benchmark-results').innerText();
    console.info('[E2E_BENCHMARK_META]', metaText);
    console.info('[E2E_BENCHMARK_RESULTS]', resultsText.replace(/\n+/g, ' | '));

    expect(externalRequests).toEqual([]);
    expect(pageErrors).toEqual([]);
  });

  test('renders refined Persian typography, logo, and menu cleanly on mobile portrait (390x844) without overflow', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/');
    await expect(page.getByTestId('hero-brand-logo')).toBeVisible();
    await expect(page.getByTestId('splash-start-btn')).toBeVisible();

    const splashOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth
    );
    expect(splashOverflow).toBe(false);

    await page.getByTestId('splash-practice-btn').click();
    await expect(page.getByTestId('menu-play-btn')).toBeVisible();
    const menuOverflow = await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth
    );
    expect(menuOverflow).toBe(false);
  });
});
