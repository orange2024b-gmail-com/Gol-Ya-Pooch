import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import {
  ALL_FSM_STATES,
  createInitialGameContext,
  transitionGameFsm,
  type GameEvent,
} from '../../src/game/fsm';
import type { LevelId } from '../../src/game/levels';

const staticSetup = {
  secretGolHandIndex: 0,
  commitPublic: {
    commitHex: 'a'.repeat(64),
    fingerprint: 'aaaaaaaaaaaa…',
    handsCount: 2,
    createdAtMs: 0,
  },
  commitSecret: {
    handIndex: 0,
    choice: 'LEFT',
    saltHex: 'b'.repeat(32),
    preimage: `LEFT:${'b'.repeat(32)}`,
  },
};

const eventArb: fc.Arbitrary<GameEvent> = fc.oneof(
  fc.constant<GameEvent>({ type: 'GO_PERMISSION' }),
  fc.constant<GameEvent>({ type: 'ENTER_PRACTICE_MENU' }),
  fc.constant<GameEvent>({ type: 'OPEN_LEVEL_SELECT' }),
  fc.constant<GameEvent>({ type: 'OPEN_STATS' }),
  fc.constant<GameEvent>({ type: 'OPEN_RULES' }),
  fc.constant<GameEvent>({ type: 'OPEN_CONSENT' }),
  fc.constant<GameEvent>({ type: 'OPEN_BENCHMARK' }),
  fc.record({
    type: fc.constant<'SELECT_LEVEL'>('SELECT_LEVEL'),
    levelId: fc.constantFrom<LevelId>(1, 2, 3, 4, 5),
    teamHandsCount: fc.constantFrom<4 | 6>(4, 6),
  }),
  fc.constant<GameEvent>({ type: 'CAMERA_READY' }),
  fc.constant<GameEvent>({ type: 'CALIBRATION_COMPLETE' }),
  fc.constant<GameEvent>({ type: 'OPEN_CALIBRATION' }),
  fc.record({
    type: fc.constant<'START_GAME'>('START_GAME'),
    computerHideSetup: fc.constant(staticSetup),
  }),
  fc.record({
    type: fc.constant<'PINCH_CONFIRMED'>('PINCH_CONFIRMED'),
    side: fc.constantFrom<'LEFT' | 'RIGHT'>('LEFT', 'RIGHT'),
  }),
  fc.record({
    type: fc.constant<'FINISH_HIDING'>('FINISH_HIDING'),
    contactCount: fc.integer({ min: 0, max: 3 }),
  }),
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
  fc.constant<GameEvent>({ type: 'COMPUTER_SHUFFLE_DONE' }),
  fc.record({
    type: fc.constant<'HUMAN_GUESS_CALL'>('HUMAN_GUESS_CALL'),
    handIndex: fc.integer({ min: 0, max: 5 }),
    callType: fc.constantFrom<'GOL' | 'POOCH'>('GOL', 'POOCH'),
  }),
  fc.record({
    type: fc.constant<'NEXT_ROUND'>('NEXT_ROUND'),
    computerHideSetup: fc.constant(staticSetup),
  }),
  fc.constant<GameEvent>({ type: 'RESET_MATCH' }),
  fc.constant<GameEvent>({ type: 'EXIT_TO_MENU' }),
  fc.record({
    type: fc.constant<'TICK'>('TICK'),
    dtMs: fc.integer({ min: 0, max: 15_000 }),
    maxLostDurationMs: fc.integer({ min: 0, max: 2_000 }),
    randomBit: fc.constantFrom<0 | 1>(0, 1),
    contactCount: fc.integer({ min: 0, max: 3 }),
  }),
  fc.constant<GameEvent>({ type: 'TAB_HIDDEN' }),
  fc.constant<GameEvent>({ type: 'TAB_VISIBLE' }),
  fc.constant<GameEvent>({ type: 'CAMERA_LOST' }),
  fc.constant<GameEvent>({ type: 'CAMERA_RECOVERED' }),
  fc.constant<GameEvent>({ type: 'RETRY_PERMISSION' })
);

describe('Game FSM property-based invariants (fast-check, Phases 1-3)', () => {
  it('never enters an undefined state and preserves score, turn ownership, contact, and phase invariants', () => {
    fc.assert(
      fc.property(fc.array(eventArb, { minLength: 1, maxLength: 140 }), (events) => {
        let ctx = createInitialGameContext();
        const validStateSet = new Set<string>(ALL_FSM_STATES);

        for (const ev of events) {
          ctx = transitionGameFsm(ctx, ev);

          expect(validStateSet.has(ctx.state)).toBe(true);
          expect(ctx.phaseRemainingMs).toBeGreaterThanOrEqual(0);
          expect(ctx.humanScore).toBeGreaterThanOrEqual(0);
          expect(ctx.computerScore).toBeGreaterThanOrEqual(0);
          expect(ctx.humanScore + ctx.computerScore).toBe(ctx.totalAwardedPoints);
          expect(ctx.ruleHiderOwner).not.toBe(ctx.ruleGuesserOwner);

          if (ctx.state === 'HIDING') {
            expect(ctx.pinchConfirmed).toBe(true);
            expect(ctx.initialShownHand).not.toBeNull();
          }

          if (ctx.state === 'COMPUTER_GUESS' && ctx.config.minHandContactsRequired > 0) {
            expect(ctx.hidingContactCount).toBeGreaterThanOrEqual(
              ctx.config.minHandContactsRequired
            );
          }

          if (ctx.state === 'ROUND_RESULT') {
            expect(ctx.lastRoundWinner === 'HUMAN' || ctx.lastRoundWinner === 'COMPUTER').toBe(
              true
            );
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
