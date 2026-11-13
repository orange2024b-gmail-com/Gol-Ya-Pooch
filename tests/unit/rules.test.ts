import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import classic1v1Raw from '../../src/game/rules/classic-1v1.json';
import {
  CLASSIC_1V1_RULES,
  RuleSchemaError,
  isValidRuleSet,
  loadRuleSet,
  resolveRoundByRules,
  validateRuleSet,
  type PlayerRole,
  type RoundGuessOutcome,
} from '../../src/game/rules/engine';
import {
  applyRoundOutcomeToScore,
  createMatchScoreState,
  isMatchOver,
} from '../../src/game/scoring';

describe('Rule engine & classic-1v1 scoring (src/game/rules/, src/game/scoring.ts)', () => {
  it('validates classic-1v1.json and rejects malformed rule schemas', () => {
    const validated = validateRuleSet(classic1v1Raw);
    expect(validated.id).toBe('classic-1v1');
    expect(validated.targetScore).toBe(7);
    expect(validated.handsCount).toBe(2);
    expect(isValidRuleSet(classic1v1Raw)).toBe(true);

    expect(() => validateRuleSet(null)).toThrow(RuleSchemaError);
    expect(() => validateRuleSet({ ...classic1v1Raw, targetScore: 0 })).toThrow(RuleSchemaError);
    expect(() =>
      validateRuleSet({
        ...classic1v1Raw,
        scoring: { ...classic1v1Raw.scoring, voidRoundPoints: 1 },
      })
    ).toThrow(RuleSchemaError);
    expect(isValidRuleSet({ id: '' })).toBe(false);

    const customTarget = loadRuleSet(classic1v1Raw, { targetScore: 5 });
    expect(customTarget.targetScore).toBe(5);
  });

  it('tracks scores toward configurable target and manages turn ownership cleanly', () => {
    const rules = loadRuleSet(classic1v1Raw, { targetScore: 3 });
    let match = createMatchScoreState(rules, 'HUMAN');
    expect(match.currentHider).toBe('HUMAN');
    expect(match.currentGuesser).toBe('COMPUTER');
    expect(isMatchOver(match)).toBe(false);

    // Wrong guess -> hider (HUMAN) +1 and keeps turn
    match = applyRoundOutcomeToScore(match, 'WRONG');
    expect(match.humanScore).toBe(1);
    expect(match.computerScore).toBe(0);
    expect(match.currentHider).toBe('HUMAN');
    expect(match.lastTurnPassed).toBe(false);

    // Void round -> 0 points, turn unchanged
    match = applyRoundOutcomeToScore(match, 'VOID');
    expect(match.humanScore).toBe(1);
    expect(match.validRoundsCompleted).toBe(1);

    // Correct guess -> guesser (COMPUTER) +1 and turn passes to COMPUTER
    match = applyRoundOutcomeToScore(match, 'CORRECT_DIRECT');
    expect(match.computerScore).toBe(1);
    expect(match.currentHider).toBe('COMPUTER');
    expect(match.currentGuesser).toBe('HUMAN');
    expect(match.lastTurnPassed).toBe(true);

    // Human guesses wrong twice -> Computer wins match at 3
    match = applyRoundOutcomeToScore(match, 'WRONG');
    match = applyRoundOutcomeToScore(match, 'WRONG');
    expect(match.computerScore).toBe(3);
    expect(match.matchWinner).toBe('COMPUTER');
    expect(isMatchOver(match)).toBe(true);
  });

  it('fast-check property: rule engine always defines distinct turn owners and awards exact rule points', () => {
    const roleArb = fc.constantFrom<PlayerRole>('HUMAN', 'COMPUTER');
    const outcomeArb = fc.constantFrom<RoundGuessOutcome>(
      'CORRECT_DIRECT',
      'CORRECT_AFTER_POOCH',
      'WRONG',
      'VOID'
    );
    const ruleArb = fc.record({
      targetScore: fc.integer({ min: 1, max: 15 }),
      correctDirect: fc.integer({ min: 1, max: 3 }),
      correctPooch: fc.integer({ min: 0, max: 3 }),
      wrongHider: fc.integer({ min: 1, max: 3 }),
      passOnCorrect: fc.boolean(),
      keepOnWrong: fc.boolean(),
    });

    fc.assert(
      fc.property(
        ruleArb,
        roleArb,
        fc.array(outcomeArb, { minLength: 1, maxLength: 40 }),
        (cfg, initialHider, outcomes) => {
          const rules = validateRuleSet({
            ...CLASSIC_1V1_RULES,
            targetScore: cfg.targetScore,
            scoring: {
              correctGolNoWrongCalls: cfg.correctDirect,
              correctGolAfterPoochCalls: cfg.correctPooch,
              wrongGuessOrOutOfGuessesHiderPoints: cfg.wrongHider,
              voidRoundPoints: 0,
            },
            turnPolicy: {
              passTurnOnCorrectGuess: cfg.passOnCorrect,
              keepTurnOnWrongGuess: cfg.keepOnWrong,
            },
          });

          let state = createMatchScoreState(rules, initialHider);
          let expectedTotalPoints = 0;

          for (const outcome of outcomes) {
            const res = resolveRoundByRules(rules, {
              hider: state.currentHider,
              guesser: state.currentGuesser,
              guessOutcome: outcome,
            });

            // Turn ownership always defined & distinct
            expect(res.nextHider === 'HUMAN' || res.nextHider === 'COMPUTER').toBe(true);
            expect(res.nextGuesser === 'HUMAN' || res.nextGuesser === 'COMPUTER').toBe(true);
            expect(res.nextHider).not.toBe(res.nextGuesser);

            // Points awarded match rule set
            if (outcome === 'VOID') {
              expect(res.totalPointsAwarded).toBe(0);
            } else if (outcome === 'CORRECT_DIRECT') {
              expect(res.totalPointsAwarded).toBe(cfg.correctDirect);
            } else if (outcome === 'CORRECT_AFTER_POOCH') {
              expect(res.totalPointsAwarded).toBe(cfg.correctPooch);
            } else {
              expect(res.totalPointsAwarded).toBe(cfg.wrongHider);
            }

            expectedTotalPoints += res.totalPointsAwarded;
            state = applyRoundOutcomeToScore(state, outcome);

            expect(state.humanScore + state.computerScore).toBe(expectedTotalPoints);
            expect(state.currentHider).not.toBe(state.currentGuesser);
          }
        }
      ),
      { numRuns: 200 }
    );
  });
});
