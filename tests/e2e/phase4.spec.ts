import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';

test.describe('Phase 4 E2E: Consent-gated logging, JSON export, self-hosted ONNX WASM inference & fallback', () => {
  test('verifies default-off consent, opt-in round logging, numeric-only JSON export, and self-hosted ONNX inference with 0 external requests', async ({
    page,
  }) => {
    const externalRequests: string[] = [];
    const localOrtRequests: string[] = [];
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
      } else if (url.includes('/ort/') || url.includes('/models/') || url.includes('.wasm')) {
        localOrtRequests.push(url);
      }
    });

    await page.goto('/');
    await expect(page.getByTestId('splash-practice-btn')).toBeVisible();

    // Go to Menu -> Consent screen
    await page.getByTestId('splash-practice-btn').click();
    await expect(page.getByTestId('menu-consent-btn')).toBeVisible();
    await page.getByTestId('menu-consent-btn').click();

    // Assert consent is OFF by default
    await expect(page.getByTestId('consent-screen')).toBeVisible();
    await expect(page.getByTestId('consent-status-badge')).toContainText('پیش‌فرض خاموش');
    await expect(page.getByTestId('consent-toggle-checkbox')).not.toBeChecked();

    // Opt in to numeric-only local logging
    await page.getByTestId('consent-toggle-checkbox').check();
    await expect(page.getByTestId('consent-status-badge')).toContainText('فعال است ✓');

    // Open Debug Panel and verify self-hosted ONNX Runtime Web is READY
    await page.getByTestId('toggle-debug-btn').click();
    await expect(page.getByTestId('debug-panel')).toBeVisible();
    await expect(page.locator('[data-debug-metric="ONNX Temporal"]')).toContainText(
      'READY (wasm, phase4-cnn-v1)',
      { timeout: 15_000 }
    );

    // Run live ONNX benchmark in debug panel and confirm measured latency
    await page.getByTestId('debug-run-onnx-btn').click();
    await expect(page.locator('[data-debug-metric="ONNX Temporal"]')).toContainText('ms', {
      timeout: 10_000,
    });

    // Run Phase 2 transfer sample replay -> completes round, runs HMM + ONNX fusion, logs round to IndexedDB
    await page.getByTestId('debug-replay-transfer-btn').click();
    await expect(page.getByTestId('replay-status')).toContainText('بازپخش کامل شد', {
      timeout: 20_000,
    });
    await expect(page.locator('[data-debug-metric="Fused Tracking"]')).toContainText('HMM+ONNX');

    // Navigate to Menu -> Stats -> Export JSON
    await page.getByRole('button', { name: 'بازگشت به منو' }).click();
    await page.getByTestId('menu-stats-btn').click();
    await expect(page.getByTestId('stats-logs-section')).toContainText('۱ دور ثبت‌شده', {
      timeout: 10_000,
    });

    const downloadPromise = page.waitForEvent('download');
    await page.getByTestId('stats-export-logs-btn').click();
    const download = await downloadPromise;
    const downloadPath = await download.path();
    if (!downloadPath) throw new Error('Expected download path');

    const exportedRaw = readFileSync(downloadPath, 'utf8');
    const exported = JSON.parse(exportedRaw) as {
      schemaVersion: number;
      consentEnabled: boolean;
      privacyNotice: string;
      rounds: Array<Record<string, unknown>>;
    };
    expect(exported.schemaVersion).toBe(1);
    expect(exported.consentEnabled).toBe(true);
    expect(exported.privacyNotice).toContain('no-images-no-video');
    expect(exported.rounds.length).toBeGreaterThanOrEqual(1);
    expect(exportedRaw).not.toMatch(/data:image|data:video|frameImage|videoBlob/i);

    // Clear logs cleanly
    await page.getByTestId('stats-clear-logs-btn').click();
    await expect(page.getByTestId('stats-logs-section')).toContainText('۰ دور ثبت‌شده');

    expect(localOrtRequests.some((u) => u.includes('/models/temporal_tracker.onnx'))).toBe(true);
    expect(localOrtRequests.some((u) => u.includes('/ort/ort-wasm-simd-threaded.wasm'))).toBe(true);
    expect(externalRequests).toEqual([]);
    expect(pageErrors).toEqual([]);
  });

  test('falls back gracefully to HMM-only when ONNX model fails to load (404)', async ({
    page,
  }) => {
    const pageErrors: string[] = [];
    page.on('pageerror', (err) => pageErrors.push(err.message));

    await page.route('**/models/temporal_tracker.onnx', (route) =>
      route.fulfill({ status: 404, body: 'Missing model' })
    );

    await page.goto('/');
    await page.getByTestId('toggle-debug-btn').click();
    await expect(page.locator('[data-debug-metric="ONNX Temporal"]')).toContainText('FALLBACK', {
      timeout: 10_000,
    });

    await page.getByTestId('debug-replay-transfer-btn').click();
    await expect(page.getByTestId('replay-status')).toContainText('بازپخش کامل شد', {
      timeout: 20_000,
    });
    await expect(page.locator('[data-debug-metric="Fused Tracking"]')).toContainText('HMM-only');
    expect(pageErrors).toEqual([]);
  });
});
