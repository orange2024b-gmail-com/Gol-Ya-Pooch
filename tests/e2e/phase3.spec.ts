import { expect, test } from '@playwright/test';

test.describe('Phase 3 E2E: PixiJS computer-hides, commit-reveal UI, traditional-team pooch/gol, level selector & zero external requests', () => {
  test('exercises PixiJS hiding scene, commit-reveal verification, level selector, stats, and traditional-team pooch/gol with zero external requests', async ({
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
    await expect(page.getByTestId('splash-practice-btn')).toBeVisible();

    // Enter practice mode (computer hides without camera) -> Menu
    await page.getByTestId('splash-practice-btn').click();
    await expect(page.getByTestId('menu-levels-btn')).toBeVisible();

    // Check Level Selector shows all 5 levels (1 unlocked, 2-5 locked)
    await page.getByTestId('menu-levels-btn').click();
    await expect(page.getByTestId('level-select-grid')).toBeVisible();
    await expect(page.getByTestId('select-level-btn-1')).toBeEnabled();
    await expect(page.getByTestId('select-level-btn-2')).toBeDisabled();
    await expect(page.getByTestId('select-level-btn-5')).toBeDisabled();

    // Start Level 1 (classic-1v1, 2 hands) -> COMPUTER_HIDES with PixiJS canvas
    await page.getByTestId('select-level-btn-1').click();
    await expect(page.getByTestId('pixi-stage')).toBeVisible();
    await expect(page.getByTestId('pixi-hiding-canvas')).toBeVisible({ timeout: 10_000 });
    const canvasBox = await page.getByTestId('pixi-hiding-canvas').boundingBox();
    expect(canvasBox?.width ?? 0).toBeGreaterThan(200);
    expect(canvasBox?.height ?? 0).toBeGreaterThan(150);

    // Commit fingerprint is visible BEFORE human guesses
    await expect(page.getByTestId('commit-fingerprint-value')).toBeVisible();
    await expect(page.getByTestId('commit-fingerprint-value')).not.toContainText('در حال تولید');

    // Wait for shuffle choreography to transition to HUMAN_GUESS
    await expect(page.getByTestId('hand-guess-btn-0')).toBeVisible({ timeout: 10_000 });
    const fpBeforeGuess = await page.getByTestId('commit-fingerprint-value').innerText();
    expect(fpBeforeGuess.length).toBeGreaterThanOrEqual(8);

    // Human guesses hand 0 -> ROUND_RESULT with commit-reveal verification
    await page.getByTestId('hand-guess-btn-0').click();
    await expect(page.getByTestId('commit-verify-badge')).toContainText('✓ تأیید شد', {
      timeout: 5_000,
    });

    // Toggle "این چطور کار می‌کند؟" explanation
    await page.getByTestId('how-it-works-btn').click();
    await expect(page.getByTestId('how-it-works-box')).toBeVisible();
    await expect(page.getByTestId('how-it-works-box')).toContainText('SHA-256');

    // Return to Menu -> Rules -> switch to traditional-team (4 hands)
    await page.getByRole('button', { name: 'بازگشت به منو' }).click();
    await page.getByTestId('menu-rules-btn').click();
    await expect(page.getByTestId('team-hands-select')).toBeVisible();
    await page.getByTestId('team-hands-select').selectOption('4');
    await page.getByTestId('apply-rule-traditional-team').click();
    await page.getByRole('button', { name: 'بازگشت به منو' }).click();

    // Play traditional-team round
    await page.getByTestId('menu-play-btn').click();
    await expect(page.getByTestId('hand-pooch-btn-0')).toBeVisible({ timeout: 10_000 });
    await expect(page.getByTestId('remaining-guesses-info')).toBeVisible();

    // Call POOCH on hand 0; if round didn't end immediately (i.e. hand 0 was empty), call GOL on hand 1
    await page.getByTestId('hand-pooch-btn-0').click();
    const stillGuessing = await page.getByTestId('hand-gol-btn-1').isVisible().catch(() => false);
    if (stillGuessing) {
      await page.getByTestId('hand-gol-btn-1').click();
    }

    await expect(page.getByTestId('commit-verify-badge')).toContainText('✓ تأیید شد', {
      timeout: 5_000,
    });

    // Check Stats screen shows recorded rounds
    await page.getByRole('button', { name: 'بازگشت به منو' }).click();
    await page.getByTestId('menu-stats-btn').click();
    await expect(page.getByTestId('stats-grid')).toBeVisible();

    expect(externalRequests).toEqual([]);
    expect(pageErrors).toEqual([]);
  });

  test('renders PixiJS computer-hides scene cleanly in mobile portrait without horizontal overflow', async ({
    page,
  }) => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto('/');
    await page.getByTestId('splash-practice-btn').click();
    await page.getByTestId('menu-play-btn').click();
    await expect(page.getByTestId('pixi-hiding-canvas')).toBeVisible({ timeout: 10_000 });
    await expect(page.getByTestId('hand-guess-btn-0')).toBeVisible({ timeout: 10_000 });

    const overflow = await page.evaluate(
      () => document.documentElement.scrollWidth > window.innerWidth
    );
    expect(overflow).toBe(false);
  });
});
