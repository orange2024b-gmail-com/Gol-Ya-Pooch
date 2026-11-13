import classic1v1Json from './classic-1v1.json';
import { validateRuleSet, type RuleSetDefinition } from './schema';

export * from './schema';

export type PlayerRole = 'HUMAN' | 'COMPUTER';

export type RoundGuessOutcome =
  | 'CORRECT_DIRECT'
  | 'CORRECT_AFTER_POOCH'
  | 'WRONG'
  | 'VOID';

export interface RoundResolutionInput {
  hider: PlayerRole;
  guesser: PlayerRole;
  guessOutcome: RoundGuessOutcome;
}

export interface RoundResolutionOutput {
  humanDelta: number;
  computerDelta: number;
  totalPointsAwarded: number;
  winner: PlayerRole | null;
  nextHider: PlayerRole;
  nextGuesser: PlayerRole;
  turnPassed: boolean;
  isValidRound: boolean;
}

export const CLASSIC_1V1_RULES: RuleSetDefinition = validateRuleSet(classic1v1Json);

export function loadRuleSet(
  raw: unknown,
  overrides: { targetScore?: number } = {}
): RuleSetDefinition {
  const base = validateRuleSet(raw);
  if (overrides.targetScore !== undefined) {
    return validateRuleSet({ ...base, targetScore: overrides.targetScore });
  }
  return base;
}

function oppositeRole(role: PlayerRole): PlayerRole {
  return role === 'HUMAN' ? 'COMPUTER' : 'HUMAN';
}

/**
 * Generic rule interpreter computing points awarded and next turn ownership
 * solely from the validated JSON rule set.
 */
export function resolveRoundByRules(
  rules: RuleSetDefinition,
  input: RoundResolutionInput
): RoundResolutionOutput {
  if (input.hider === input.guesser) {
    throw new Error('Hider and guesser must be distinct players');
  }

  if (input.guessOutcome === 'VOID') {
    return {
      humanDelta: 0,
      computerDelta: 0,
      totalPointsAwarded: rules.scoring.voidRoundPoints,
      winner: null,
      nextHider: input.hider,
      nextGuesser: input.guesser,
      turnPassed: false,
      isValidRound: false,
    };
  }

  let points: number;
  let winner: PlayerRole;
  let turnPassed: boolean;

  if (input.guessOutcome === 'CORRECT_DIRECT') {
    points = rules.scoring.correctGolNoWrongCalls;
    winner = input.guesser;
    turnPassed = rules.turnPolicy.passTurnOnCorrectGuess;
  } else if (input.guessOutcome === 'CORRECT_AFTER_POOCH') {
    points = rules.scoring.correctGolAfterPoochCalls;
    winner = input.guesser;
    turnPassed = rules.turnPolicy.passTurnOnCorrectGuess;
  } else {
    points = rules.scoring.wrongGuessOrOutOfGuessesHiderPoints;
    winner = input.hider;
    turnPassed = !rules.turnPolicy.keepTurnOnWrongGuess;
  }

  const nextHider = turnPassed ? oppositeRole(input.hider) : input.hider;
  const nextGuesser = oppositeRole(nextHider);

  return {
    humanDelta: winner === 'HUMAN' ? points : 0,
    computerDelta: winner === 'COMPUTER' ? points : 0,
    totalPointsAwarded: points,
    winner,
    nextHider,
    nextGuesser,
    turnPassed,
    isValidRound: true,
  };
}
