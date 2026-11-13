import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import classic1v1Raw from '../../src/game/rules/classic-1v1.json';
import customRaw from '../../src/game/rules/custom.json';
import traditionalTeamRaw from '../../src/game/rules/traditional-team.json';
import {
  CLASSIC_1V1_RULES,
  CUSTOM_TEMPLATE_RULES,
  RuleSchemaError,
  TRADITIONAL_TEAM_RULES,
  evaluateGuessCallStep,
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

describe('Rule engine, classic-1v1, traditional-team & custom (src/game/rules/, src/game/scoring.ts)', () => {
  it('validates classic-1v1.json, traditional-team.json, and custom.json and rejects malformed rule schemas', () => {
    const validated = validateRuleSet(classic1v1Raw);
    expect(validated.id).toBe('classic-1v1');
    expect(validated.targetScore).toBe(7);
    expect(validated.handsCount).toBe(2);
    expect(isValidRuleSet(classic1v1Raw)).toBe(true);

    const team = validateRuleSet(traditionalTeamRaw);
    expect(team.id).toBe('traditional-team');
    expect(team.handsCount).toBe(4);
    expect(team.maxGuessesPerRound).toBe(3);
    expect(team.allowPoochCalls).toBe(true);
    expect(team.scoring.correctGolNoWrongCalls).toBe(2);
    expect(team.scoring.correctGolAfterPoochCalls).toBe(0);
    expect(team.scoring.wrongGuessOrOutOfGuessesHiderPoints).toBe(1);
    expect(TRADITIONAL_TEAM_RULES.id).toBe('traditional-team');

    const custom = validateRuleSet(customRaw);
    expect(custom.id).toBe('custom');
    expect(CUSTOM_TEMPLATE_RULES.id).toBe('custom');

    expect(() => validateRuleSet(null)).toThrow(RuleSchemaError);
    expect(() => validateRuleSet({ ...classic1v1Raw, targetScore: 0 })).toThrow(RuleSchemaError);
    expect(() =>
      validateRuleSet({
        ...classic1v1Raw,
        scoring: { ...classic1v1Raw.scoring, voidRoundPoints: 1 },
      })
    ).toThrow(RuleSchemaError);
    expect(isValidRuleSet({ id: '' })).toBe(false);

    const teamSix = loadRuleSet(traditionalTeamRaw, { targetScore: 5, handsCount: 6 });
    expect(teamSix.targetScore).toBe(5);
    expect(teamSix.handsCount).toBe(6);
  });

  it('implements traditional-team scoring (+2 direct, +0 after pooch, +1 hider on pooch-gol or out-of-guesses) and turn logic', () => {
    const rules = loadRuleSet(traditionalTeamRaw, { handsCount: 6, targetScore: 7 });
    let match = createMatchScoreState(rules, 'COMPUTER');

    // 1. Direct GOL on first call -> +2 for guesser (HUMAN) and turn passes to HUMAN
    const stepDirect = evaluateGuessCallStep(
      rules,
      { guessesUsed: 0, poochCallsCount: 0, openedEmptyHands: [] },
      3,
      3,
      'GOL'
    );
    expect(stepDirect.roundEnded).toBe(true);
    expect(stepDirect.outcome).toBe('CORRECT_DIRECT');
    match = applyRoundOutcomeToScore(match, stepDirect.outcome!);
    expect(match.humanScore).toBe(2);
    expect(match.currentHider).toBe('HUMAN');

    // 2. Correct POOCH call then GOL -> +0 points and turn passes back to COMPUTER
    let prog = { guessesUsed: 0, poochCallsCount: 0, openedEmptyHands: [] as number[] };
    const pooch1 = evaluateGuessCallStep(rules, prog, 4, 0, 'POOCH');
    expect(pooch1.roundEnded).toBe(false);
    expect(pooch1.openedHandIndex).toBe(0);
    prog = pooch1.nextProgress;

    const golAfterPooch = evaluateGuessCallStep(rules, prog, 4, 4, 'GOL');
    expect(golAfterPooch.roundEnded).toBe(true);
    expect(golAfterPooch.outcome).toBe('CORRECT_AFTER_POOCH');
    match = applyRoundOutcomeToScore(match, golAfterPooch.outcome!);
    expect(match.computerScore).toBe(0);
    expect(match.currentHider).toBe('COMPUTER');

    // 3. POOCH hits the gol -> WRONG, hider (COMPUTER) scores +1 and keeps turn
    const poochHit = evaluateGuessCallStep(
      rules,
      { guessesUsed: 0, poochCallsCount: 0, openedEmptyHands: [] },
      2,
      2,
      'POOCH'
    );
    expect(poochHit.roundEnded).toBe(true);
    expect(poochHit.outcome).toBe('WRONG');
    expect(poochHit.reason).toBe('POOCH_HIT_GOL');
    match = applyRoundOutcomeToScore(match, poochHit.outcome!);
    expect(match.computerScore).toBe(1);
    expect(match.currentHider).toBe('COMPUTER');

    // 4. Three POOCH calls without calling GOL -> OUT_OF_GUESSES -> WRONG (+1 hider, keeps turn)
    let p2 = { guessesUsed: 0, poochCallsCount: 0, openedEmptyHands: [] as number[] };
    p2 = evaluateGuessCallStep(rules, p2, 5, 0, 'POOCH').nextProgress;
    p2 = evaluateGuessCallStep(rules, p2, 5, 1, 'POOCH').nextProgress;
    const thirdPooch = evaluateGuessCallStep(rules, p2, 5, 2, 'POOCH');
    expect(thirdPooch.roundEnded).toBe(true);
    expect(thirdPooch.outcome).toBe('WRONG');
    expect(thirdPooch.reason).toBe('OUT_OF_GUESSES');
  });

  it('tracks classic-1v1 scores toward configurable target and manages turn ownership cleanly', () => {
    const rules = loadRuleSet(classic1v1Raw, { targetScore: 3 });
    let match = createMatchScoreState(rules, 'HUMAN');
    expect(match.currentHider).toBe('HUMAN');
    expect(match.currentGuesser).toBe('COMPUTER');
    expect(isMatchOver(match)).toBe(false);

    match = applyRoundOutcomeToScore(match, 'WRONG');
    expect(match.humanScore).toBe(1);
    expect(match.currentHider).toBe('HUMAN');
    expect(match.lastTurnPassed).toBe(false);

    match = applyRoundOutcomeToScore(match, 'VOID');
    expect(match.humanScore).toBe(1);
    expect(match.validRoundsCompleted).toBe(1);

    match = applyRoundOutcomeToScore(match, 'CORRECT_DIRECT');
    expect(match.computerScore).toBe(1);
    expect(match.currentHider).toBe('COMPUTER');
    expect(match.lastTurnPassed).toBe(true);

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

            expect(res.nextHider === 'HUMAN' || res.nextHider === 'COMPUTER').toBe(true);
            expect(res.nextGuesser === 'HUMAN' || res.nextGuesser === 'COMPUTER').toBe(true);
            expect(res.nextHider).not.toBe(res.nextGuesser);

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
