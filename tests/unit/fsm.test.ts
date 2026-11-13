import { describe, expect, it } from 'vitest';
import {
  ALL_FSM_STATES,
  createInitialGameContext,
  transitionGameFsm,
  type GameContext,
  type GameEvent,
} from '../../src/game/fsm';

function runSequence(initial: GameContext, events: GameEvent[]): GameContext {
  return events.reduce((ctx, ev) => transitionGameFsm(ctx, ev), initial);
}

describe('Game FSM: 100% state & transition coverage', () => {
  it('covers splash -> permission -> calibration -> menu -> full valid round -> result -> next round / reset / exit', () => {
    let ctx = createInitialGameContext();
    expect(ctx.state).toBe('SPLASH');

    // No-op on invalid event in SPLASH
    expect(transitionGameFsm(ctx, { type: 'START_GAME' }).state).toBe('SPLASH');

    ctx = transitionGameFsm(ctx, { type: 'GO_PERMISSION' });
    expect(ctx.state).toBe('PERMISSION');
    expect(transitionGameFsm(ctx, { type: 'START_GAME' }).state).toBe('PERMISSION');

    ctx = transitionGameFsm(ctx, { type: 'CAMERA_READY' });
    expect(ctx.state).toBe('CALIBRATION');
    expect(transitionGameFsm(ctx, { type: 'EXIT_TO_MENU' }).state).toBe('CALIBRATION');

    ctx = transitionGameFsm(ctx, { type: 'CALIBRATION_COMPLETE' });
    expect(ctx.state).toBe('MENU');
    expect(ctx.calibrated).toBe(true);

    // Re-entering calibration and exiting back to menu once calibrated
    ctx = transitionGameFsm(ctx, { type: 'OPEN_CALIBRATION' });
    expect(ctx.state).toBe('CALIBRATION');
    ctx = transitionGameFsm(ctx, { type: 'EXIT_TO_MENU' });
    expect(ctx.state).toBe('MENU');

    // Also verify uncalibrated START_GAME redirects to CALIBRATION
    const uncal = transitionGameFsm(
      { ...createInitialGameContext(), state: 'MENU', calibrated: false },
      { type: 'START_GAME' }
    );
    expect(uncal.state).toBe('CALIBRATION');

    // Also verify PERMISSION -> MENU directly when already calibrated
    const recal = transitionGameFsm(
      { ...createInitialGameContext(), state: 'PERMISSION', calibrated: true },
      { type: 'CAMERA_READY' }
    );
    expect(recal.state).toBe('MENU');

    // Start game
    ctx = transitionGameFsm(ctx, { type: 'START_GAME' });
    expect(ctx.state).toBe('SHOW_GOL');
    expect(ctx.pinchConfirmed).toBe(false);

    // Tick without timeout
    ctx = transitionGameFsm(ctx, { type: 'TICK', dtMs: 500, maxLostDurationMs: 100 });
    expect(ctx.state).toBe('SHOW_GOL');
    expect(ctx.phaseRemainingMs).toBe(11_500);

    // Confirm pinch -> HIDING
    ctx = transitionGameFsm(ctx, { type: 'PINCH_CONFIRMED', side: 'LEFT' });
    expect(ctx.state).toBe('HIDING');
    expect(ctx.pinchConfirmed).toBe(true);
    expect(ctx.initialShownHand).toBe('LEFT');

    // Tick inside HIDING -> then timeout -> COMPUTER_GUESS
    ctx = transitionGameFsm(ctx, { type: 'TICK', dtMs: 1000, maxLostDurationMs: 0 });
    expect(ctx.state).toBe('HIDING');
    ctx = transitionGameFsm(ctx, { type: 'TICK', dtMs: 15_000, maxLostDurationMs: 0, randomBit: 0 });
    expect(ctx.state).toBe('COMPUTER_GUESS');
    expect(ctx.computerGuess).toBe('LEFT');

    // Also test FINISH_HIDING shortcut
    const fastHide = transitionGameFsm(
      { ...ctx, state: 'HIDING', phaseRemainingMs: 5000 },
      { type: 'FINISH_HIDING' }
    );
    expect(fastHide.state).toBe('COMPUTER_GUESS');

    // COMPUTER_GUESS tick & timeout -> REVEAL_VERIFY
    ctx = transitionGameFsm(ctx, { type: 'TICK', dtMs: 500 });
    expect(ctx.state).toBe('COMPUTER_GUESS');
    ctx = transitionGameFsm(ctx, { type: 'TICK', dtMs: 5000 });
    expect(ctx.state).toBe('REVEAL_VERIFY');

    // Also test PROCEED_TO_REVEAL shortcut
    const skipGuess = transitionGameFsm(
      { ...ctx, state: 'COMPUTER_GUESS', phaseRemainingMs: 1000 },
      { type: 'PROCEED_TO_REVEAL' }
    );
    expect(skipGuess.state).toBe('REVEAL_VERIFY');

    // HANDS_NOT_OPEN stays in REVEAL_VERIFY
    ctx = transitionGameFsm(ctx, { type: 'REVEAL_VERIFIED', outcome: 'HANDS_NOT_OPEN' });
    expect(ctx.state).toBe('REVEAL_VERIFY');

    // Correct computer guess -> computer scores +1
    ctx = transitionGameFsm(ctx, { type: 'REVEAL_VERIFIED', outcome: 'LEFT' });
    expect(ctx.state).toBe('ROUND_RESULT');
    expect(ctx.computerScore).toBe(1);
    expect(ctx.humanScore).toBe(0);
    expect(ctx.lastRoundWinner).toBe('COMPUTER');

    // Next round -> human scores +1
    ctx = transitionGameFsm(ctx, { type: 'NEXT_ROUND' });
    expect(ctx.state).toBe('SHOW_GOL');
    expect(ctx.roundNumber).toBe(2);
    ctx = runSequence(ctx, [
      { type: 'PINCH_CONFIRMED', side: 'RIGHT' },
      { type: 'TICK', dtMs: 20_000, maxLostDurationMs: 0, randomBit: 0 },
      { type: 'PROCEED_TO_REVEAL' },
      { type: 'REVEAL_VERIFIED', outcome: 'RIGHT' },
    ]);
    expect(ctx.state).toBe('ROUND_RESULT');
    expect(ctx.humanScore).toBe(1);
    expect(ctx.computerScore).toBe(1);
    expect(ctx.lastRoundWinner).toBe('HUMAN');

    // Reset match & exit to menu
    const resetCtx = transitionGameFsm(ctx, { type: 'RESET_MATCH' });
    expect(resetCtx.state).toBe('SHOW_GOL');
    expect(resetCtx.humanScore).toBe(0);
    expect(resetCtx.computerScore).toBe(0);

    expect(transitionGameFsm(ctx, { type: 'EXIT_TO_MENU' }).state).toBe('MENU');
  });

  it('enforces all void-round rules (hands lost > 0.5s, show timeout, reveal neither/both/timeout)', () => {
    const baseShow = runSequence(createInitialGameContext(), [
      { type: 'GO_PERMISSION' },
      { type: 'CAMERA_READY' },
      { type: 'CALIBRATION_COMPLETE' },
      { type: 'START_GAME' },
    ]);

    // 1. Hands lost > 500ms in SHOW_GOL
    const lostShow = transitionGameFsm(baseShow, {
      type: 'TICK',
      dtMs: 33,
      maxLostDurationMs: 501,
    });
    expect(lostShow.state).toBe('ROUND_VOID');
    expect(lostShow.voidReason).toBe('HANDS_LOST');

    // 2. Show timeout
    const timeoutShow = transitionGameFsm(baseShow, {
      type: 'TICK',
      dtMs: 15_000,
      maxLostDurationMs: 0,
    });
    expect(timeoutShow.state).toBe('ROUND_VOID');
    expect(timeoutShow.voidReason).toBe('SHOW_TIMEOUT');

    // 3. Hands lost > 500ms in HIDING
    const hiding = transitionGameFsm(baseShow, { type: 'PINCH_CONFIRMED', side: 'LEFT' });
    const lostHide = transitionGameFsm(hiding, {
      type: 'TICK',
      dtMs: 33,
      maxLostDurationMs: 550,
    });
    expect(lostHide.state).toBe('ROUND_VOID');
    expect(lostHide.voidReason).toBe('HANDS_LOST');

    // 4. Reveal NEITHER, BOTH, and timeout
    const revealState = runSequence(hiding, [
      { type: 'TICK', dtMs: 15_000, randomBit: 1 },
      { type: 'PROCEED_TO_REVEAL' },
    ]);
    expect(revealState.state).toBe('REVEAL_VERIFY');

    const neitherVoid = transitionGameFsm(revealState, {
      type: 'REVEAL_VERIFIED',
      outcome: 'NEITHER',
    });
    expect(neitherVoid.state).toBe('ROUND_VOID');
    expect(neitherVoid.voidReason).toBe('REVEAL_NEITHER');

    const bothVoid = transitionGameFsm(revealState, {
      type: 'REVEAL_VERIFIED',
      outcome: 'BOTH',
    });
    expect(bothVoid.state).toBe('ROUND_VOID');
    expect(bothVoid.voidReason).toBe('REVEAL_BOTH');

    const revealTimed = transitionGameFsm(revealState, { type: 'TICK', dtMs: 500 });
    expect(revealTimed.state).toBe('REVEAL_VERIFY');
    const revealTimeout = transitionGameFsm(revealTimed, { type: 'TICK', dtMs: 20_000 });
    expect(revealTimeout.state).toBe('ROUND_VOID');
    expect(revealTimeout.voidReason).toBe('REVEAL_TIMEOUT');

    // ROUND_VOID -> NEXT_ROUND replays same roundNumber, RESET_MATCH resets, EXIT_TO_MENU exits
    const replayed = transitionGameFsm(neitherVoid, { type: 'NEXT_ROUND' });
    expect(replayed.state).toBe('SHOW_GOL');
    expect(replayed.roundNumber).toBe(1);
    expect(transitionGameFsm(neitherVoid, { type: 'RESET_MATCH' }).state).toBe('SHOW_GOL');
    expect(transitionGameFsm(neitherVoid, { type: 'EXIT_TO_MENU' }).state).toBe('MENU');
  });

  it('handles tab-hidden pause/resume and recoverable camera loss across states', () => {
    const inHide = runSequence(createInitialGameContext(), [
      { type: 'GO_PERMISSION' },
      { type: 'CAMERA_READY' },
      { type: 'CALIBRATION_COMPLETE' },
      { type: 'START_GAME' },
      { type: 'PINCH_CONFIRMED', side: 'RIGHT' },
    ]);

    const paused = transitionGameFsm(inHide, { type: 'TAB_HIDDEN' });
    expect(paused.state).toBe('PAUSED');
    expect(paused.pausedFrom).toBe('HIDING');
    // Repeated TAB_HIDDEN is idempotent
    expect(transitionGameFsm(paused, { type: 'TAB_HIDDEN' }).state).toBe('PAUSED');

    const resumed = transitionGameFsm(paused, { type: 'TAB_VISIBLE' });
    expect(resumed.state).toBe('HIDING');
    expect(resumed.phaseRemainingMs).toBe(inHide.phaseRemainingMs);

    // Exiting from paused
    expect(transitionGameFsm(paused, { type: 'EXIT_TO_MENU' }).state).toBe('MENU');

    // Camera loss mid-round -> CAMERA_ERROR -> CAMERA_RECOVERED -> ROUND_VOID (no crash, scores preserved)
    const camErr = transitionGameFsm(inHide, { type: 'CAMERA_LOST' });
    expect(camErr.state).toBe('CAMERA_ERROR');
    // Repeated CAMERA_LOST is idempotent
    expect(transitionGameFsm(camErr, { type: 'CAMERA_LOST' }).state).toBe('CAMERA_ERROR');

    const recovered = transitionGameFsm(camErr, { type: 'CAMERA_RECOVERED' });
    expect(recovered.state).toBe('ROUND_VOID');
    expect(recovered.voidReason).toBe('CAMERA_INTERRUPTED');

    // Camera loss while in CALIBRATION recovers directly to CALIBRATION
    const inCal = runSequence(createInitialGameContext(), [
      { type: 'GO_PERMISSION' },
      { type: 'CAMERA_READY' },
    ]);
    const calErr = transitionGameFsm(inCal, { type: 'CAMERA_LOST', messageFa: 'خطا' });
    expect(calErr.state).toBe('CAMERA_ERROR');
    expect(transitionGameFsm(calErr, { type: 'CAMERA_RECOVERED' }).state).toBe('CALIBRATION');
    expect(transitionGameFsm(calErr, { type: 'RETRY_PERMISSION' }).state).toBe('PERMISSION');

    // Exiting from active gameplay states
    for (const activeState of ['SHOW_GOL', 'HIDING', 'COMPUTER_GUESS', 'REVEAL_VERIFY'] as const) {
      const exited = transitionGameFsm(
        { ...inHide, state: activeState },
        { type: 'EXIT_TO_MENU' }
      );
      expect(exited.state).toBe('MENU');
    }

    // All FSM states are valid and reachable, and unhandled events are safe no-ops
    expect(ALL_FSM_STATES.length).toBe(12);
    for (const s of ALL_FSM_STATES) {
      const sample = { ...inHide, state: s };
      const same = transitionGameFsm(sample, { type: 'GO_PERMISSION' });
      if (s === 'SPLASH') {
        expect(same.state).toBe('PERMISSION');
      } else {
        expect(same.state).toBe(s);
      }
    }
    expect(transitionGameFsm(inHide, { type: 'TAB_VISIBLE' }).state).toBe('HIDING');
    expect(
      transitionGameFsm(createInitialGameContext(), { type: 'CAMERA_LOST' }).state
    ).toBe('SPLASH');
    expect(
      transitionGameFsm({ ...inHide, state: 'HIDING' }, { type: 'TICK', dtMs: 20_000 }).state
    ).toBe('COMPUTER_GUESS');
  });

  it('supports Phase 2 rule engine target score, AI guess payloads, and match winner lifecycle', () => {
    let ctx = createInitialGameContext({ targetScore: 2 });
    ctx = transitionGameFsm(ctx, { type: 'GO_PERMISSION' });
    ctx = transitionGameFsm(ctx, { type: 'CAMERA_READY' });
    ctx = transitionGameFsm(ctx, { type: 'CALIBRATION_COMPLETE' });
    ctx = transitionGameFsm(ctx, { type: 'SET_TARGET_SCORE', targetScore: 2 });
    expect(ctx.ruleSet.targetScore).toBe(2);

    ctx = transitionGameFsm(ctx, { type: 'START_GAME' });
    ctx = transitionGameFsm(ctx, { type: 'PINCH_CONFIRMED', side: 'LEFT' });
    ctx = transitionGameFsm(ctx, {
      type: 'FINISH_HIDING',
      aiGuess: {
        guess: 'RIGHT',
        probability: 0.84,
        explanationFa: 'در ثانیه ۳.۲ دست‌ها به هم رسیدند.',
        timelineEvents: ['ثانیه ۳.۲: دست‌ها به هم رسیدند'],
      },
    });
    expect(ctx.state).toBe('COMPUTER_GUESS');
    expect(ctx.computerGuess).toBe('RIGHT');
    expect(ctx.computerGuessProb).toBeCloseTo(0.84, 4);
    expect(ctx.explanationFa).toContain('۳.۲');

    ctx = transitionGameFsm(ctx, { type: 'PROCEED_TO_REVEAL' });
    ctx = transitionGameFsm(ctx, { type: 'REVEAL_VERIFIED', outcome: 'RIGHT' });
    expect(ctx.state).toBe('ROUND_RESULT');
    expect(ctx.computerScore).toBe(1);
    expect(ctx.matchWinner).toBeNull();
    expect(ctx.ruleHiderOwner).toBe('COMPUTER');

    // Second round: computer wins again and reaches targetScore=2
    ctx = transitionGameFsm(ctx, { type: 'NEXT_ROUND' });
    ctx = transitionGameFsm(ctx, { type: 'PINCH_CONFIRMED', side: 'LEFT' });
    ctx = transitionGameFsm(ctx, {
      type: 'FINISH_HIDING',
      aiGuess: { guess: 'LEFT', probability: 0.91, explanationFa: 'بدون تماس باقی ماند.' },
    });
    ctx = transitionGameFsm(ctx, { type: 'PROCEED_TO_REVEAL' });
    ctx = transitionGameFsm(ctx, { type: 'REVEAL_VERIFIED', outcome: 'LEFT' });
    expect(ctx.computerScore).toBe(2);
    expect(ctx.matchWinner).toBe('COMPUTER');

    // NEXT_ROUND after match completion starts a fresh match
    ctx = transitionGameFsm(ctx, { type: 'NEXT_ROUND' });
    expect(ctx.state).toBe('SHOW_GOL');
    expect(ctx.computerScore).toBe(0);
    expect(ctx.humanScore).toBe(0);
    expect(ctx.matchWinner).toBeNull();
  });
});
