import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import {
  ALL_FSM_STATES,
  createInitialGameContext,
  transitionGameFsm,
  type GameEvent,
} from '../../src/game/fsm';

const eventArb: fc.Arbitrary<GameEvent> = fc.oneof(
  fc.constant<GameEvent>({ type: 'GO_PERMISSION' }),
  fc.constant<GameEvent>({ type: 'CAMERA_READY' }),
  fc.constant<GameEvent>({ type: 'CALIBRATION_COMPLETE' }),
  fc.constant<GameEvent>({ type: 'OPEN_CALIBRATION' }),
  fc.constant<GameEvent>({ type: 'START_GAME' }),
  fc.record({
    type: fc.constant<'PINCH_CONFIRMED'>('PINCH_CONFIRMED'),
    side: fc.constantFrom<'LEFT' | 'RIGHT'>('LEFT', 'RIGHT'),
  }),
  fc.constant<GameEvent>({ type: 'FINISH_HIDING' }),
  fc.constant<GameEvent>({ type: 'PROCEED_TO_REVEAL' }),
  fc.record({
    type: fc.constant<'REVEAL_VERIFIED'>('REVEAL_VERIFIED'),
    outcome: fc.constantFrom<'LEFT' | 'RIGHT' | 'NEITHER' | 'BOTH' | 'HANDS_NOT_OPEN'>(
      'LEFT',
      'RIGHT',
      'NEITHER',
      'BOTH',
      'HANDS_NOT_OPEN'
    ),
  }),
  fc.constant<GameEvent>({ type: 'NEXT_ROUND' }),
  fc.constant<GameEvent>({ type: 'RESET_MATCH' }),
  fc.constant<GameEvent>({ type: 'EXIT_TO_MENU' }),
  fc.record({
    type: fc.constant<'TICK'>('TICK'),
    dtMs: fc.integer({ min: 0, max: 15_000 }),
    maxLostDurationMs: fc.integer({ min: 0, max: 2_000 }),
    randomBit: fc.constantFrom<0 | 1>(0, 1),
  }),
  fc.constant<GameEvent>({ type: 'TAB_HIDDEN' }),
  fc.constant<GameEvent>({ type: 'TAB_VISIBLE' }),
  fc.constant<GameEvent>({ type: 'CAMERA_LOST' }),
  fc.constant<GameEvent>({ type: 'CAMERA_RECOVERED' }),
  fc.constant<GameEvent>({ type: 'RETRY_PERMISSION' })
);

describe('Game FSM property-based invariants (fast-check)', () => {
  it('never enters an undefined state and preserves score/phase invariants under arbitrary event sequences', () => {
    fc.assert(
      fc.property(fc.array(eventArb, { minLength: 1, maxLength: 120 }), (events) => {
        let ctx = createInitialGameContext();
        const validStateSet = new Set<string>(ALL_FSM_STATES);

        for (const ev of events) {
          ctx = transitionGameFsm(ctx, ev);

          expect(validStateSet.has(ctx.state)).toBe(true);
          expect(ctx.phaseRemainingMs).toBeGreaterThanOrEqual(0);
          expect(ctx.humanScore).toBeGreaterThanOrEqual(0);
          expect(ctx.computerScore).toBeGreaterThanOrEqual(0);
          expect(ctx.humanScore + ctx.computerScore).toBe(ctx.validRoundsCompleted);

          if (ctx.state === 'HIDING') {
            expect(ctx.pinchConfirmed).toBe(true);
            expect(ctx.initialShownHand).not.toBeNull();
          }

          if (ctx.state === 'ROUND_RESULT') {
            expect(ctx.revealedHand === 'LEFT' || ctx.revealedHand === 'RIGHT').toBe(true);
            expect(ctx.computerGuess === 'LEFT' || ctx.computerGuess === 'RIGHT').toBe(true);
            expect(ctx.lastRoundWinner === 'HUMAN' || ctx.lastRoundWinner === 'COMPUTER').toBe(true);
          }

          if (ctx.state === 'PAUSED') {
            expect(ctx.pausedFrom).not.toBeNull();
          }
        }
      }),
      { numRuns: 300 }
    );
  });
});
