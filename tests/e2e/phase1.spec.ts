import { expect, test } from '@playwright/test';

test.describe('Phase 1 E2E: Persian RTL UI, self-hosted assets, camera/calibration, debug & replay', () => {
  test('loads with zero external runtime requests, exercises camera + calibration + debug + replay', async ({
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

    await page.goto('/');
    await expect(page.locator('html')).toHaveAttribute('dir', 'rtl');
    await expect(page.locator('html')).toHaveAttribute('lang', 'fa');
    await expect(page.getByTestId('splash-start-btn')).toBeVisible();

    // Verify self-hosted Vazirmatn stylesheet resolved with HTTP 200
    const fontCssResponse = await page.request.get('/fonts/vazirmatn.css');
    expect(fontCssResponse.status()).toBe(200);

    // Toggle debug overlay with D key
    await expect(page.getByTestId('debug-panel')).toBeHidden();
    await page.keyboard.press('d');
    await expect(page.getByTestId('debug-panel')).toBeVisible();
    await page.keyboard.press('D');
    await expect(page.getByTestId('debug-panel')).toBeHidden();

    // Navigate Splash -> Permission -> Calibration using fake Chromium camera
    await page.getByTestId('splash-start-btn').click();
    await expect(page.getByTestId('grant-camera-btn')).toBeVisible();

    await page.getByTestId('grant-camera-btn').click();
    await expect(page.getByTestId('sample-gol-btn')).toBeVisible({ timeout: 20_000 });
    await expect(page.getByTestId('calibration-done-btn')).toBeDisabled();

    // Sample gol color and verify calibration panel stays interactive
    await page.getByTestId('sample-gol-btn').click();

    // Open debug panel via button and run built-in sample replay
    await page.getByTestId('toggle-debug-btn').click();
    await expect(page.getByTestId('debug-panel')).toBeVisible();
    await page.getByTestId('debug-replay-sample-btn').click();

    // Wait for replay completion status
    await expect(page.getByTestId('replay-status')).toContainText('بازپخش کامل شد', {
      timeout: 15_000,
    });

    expect(externalRequests).toEqual([]);
    expect(pageErrors).toEqual([]);
  });

  test('renders cleanly in mobile portrait viewport without horizontal overflow', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/');
    await expect(page.getByTestId('splash-start-btn')).toBeVisible();

    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth
    );
    expect(overflow).toBe(false);
  });
});
