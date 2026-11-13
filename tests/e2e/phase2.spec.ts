import { expect, test } from '@playwright/test';

test.describe('Phase 2 E2E: Belief isolation during HIDING, live debug belief graph, guess % & Persian explanation', () => {
  test('hides belief from player during HIDING, renders live belief graph in debug overlay, and shows guess % + explanation in RESULT', async ({
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
    await expect(page.getByTestId('replay-transfer-btn')).toBeVisible();

    // Launch Phase 2 transfer sample replay
    await page.getByTestId('replay-transfer-btn').click();

    // Verify debug panel opens with live belief chart & timeline during replay
    await expect(page.getByTestId('debug-panel')).toBeVisible();
    await expect(page.getByTestId('debug-belief-chart')).toBeVisible();
    await expect(page.getByTestId('debug-timeline-list')).toBeVisible();

    // Wait for HIDING phase in main panel and assert player-facing panel does NOT leak belief
    await expect(page.getByTestId('finish-hiding-btn')).toBeVisible({ timeout: 10_000 });
    const playerPanelText = await page.getByTestId('screen-panel').innerText();
    expect(playerPanelText).toContain('در حال تماشا');
    expect(playerPanelText).not.toContain('P(LEFT)');
    expect(playerPanelText).not.toContain('HMM');

    // Wait for replay completion -> ROUND_RESULT with Persian explanation, timeline, and target progress
    await expect(page.getByTestId('replay-status')).toContainText('بازپخش کامل شد', {
      timeout: 15_000,
    });
    await expect(page.getByTestId('round-explanation')).toBeVisible();
    await expect(page.getByTestId('match-target-progress')).toBeVisible();

    const explanationText = await page.getByTestId('round-explanation').innerText();
    expect(explanationText).toContain('تحلیل رایانه');
    expect(explanationText).toContain('ثانیه');

    expect(externalRequests).toEqual([]);
    expect(pageErrors).toEqual([]);
  });
});
