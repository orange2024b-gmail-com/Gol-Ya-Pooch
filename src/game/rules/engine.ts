import classic1v1Json from './classic-1v1.json';
import customJson from './custom.json';
import { validateRuleSet, type RuleSetDefinition } from './schema';
import traditionalTeamJson from './traditional-team.json';

export * from './schema';

export type PlayerRole = 'HUMAN' | 'COMPUTER';

export type RoundGuessOutcome =
  | 'CORRECT_DIRECT'
  | 'CORRECT_AFTER_POOCH'
  | 'WRONG'
  | 'VOID';

export type GuessCallType = 'GOL' | 'POOCH';

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

export interface TeamGuessProgress {
  guessesUsed: number;
  poochCallsCount: number;
  openedEmptyHands: number[];
}

export interface TeamCallStepResult {
  roundEnded: boolean;
  outcome: RoundGuessOutcome | null;
  openedHandIndex: number | null;
  nextProgress: TeamGuessProgress;
  reason: 'GOL_DIRECT' | 'GOL_AFTER_POOCH' | 'WRONG_GOL' | 'POOCH_HIT_GOL' | 'OUT_OF_GUESSES' | 'EMPTY_POOCH_OPENED';
}

export const CLASSIC_1V1_RULES: RuleSetDefinition = validateRuleSet(classic1v1Json);
export const TRADITIONAL_TEAM_RULES: RuleSetDefinition = validateRuleSet(traditionalTeamJson);
export const CUSTOM_TEMPLATE_RULES: RuleSetDefinition = validateRuleSet(customJson);

export const RULE_PRESETS: Record<string, RuleSetDefinition> = {
  'classic-1v1': CLASSIC_1V1_RULES,
  'traditional-team': TRADITIONAL_TEAM_RULES,
  custom: CUSTOM_TEMPLATE_RULES,
};

export interface RuleLoadOverrides {
  targetScore?: number;
  handsCount?: number;
  maxGuessesPerRound?: number;
  hidingTimeoutMs?: number;
}

export function loadRuleSet(
  raw: unknown,
  overrides: RuleLoadOverrides = {}
): RuleSetDefinition {
  const base = validateRuleSet(raw);
  const next: RuleSetDefinition = {
    ...base,
    ...(overrides.targetScore !== undefined ? { targetScore: overrides.targetScore } : {}),
    ...(overrides.handsCount !== undefined ? { handsCount: overrides.handsCount } : {}),
    ...(overrides.maxGuessesPerRound !== undefined
      ? { maxGuessesPerRound: overrides.maxGuessesPerRound }
      : {}),
    timeoutsMs: {
      ...base.timeoutsMs,
      ...(overrides.hidingTimeoutMs !== undefined ? { hiding: overrides.hidingTimeoutMs } : {}),
    },
  };
  return validateRuleSet(next);
}

function oppositeRole(role: PlayerRole): PlayerRole {
  return role === 'HUMAN' ? 'COMPUTER' : 'HUMAN';
}

/**
 * Evaluates a single guesser call ("GOL" or "POOCH") against the hidden gol hand index
 * according to the validated RuleSetDefinition.
 */
export function evaluateGuessCallStep(
  rules: RuleSetDefinition,
  progress: TeamGuessProgress,
  secretGolHandIndex: number,
  calledHandIndex: number,
  callType: GuessCallType
): TeamCallStepResult {
  if (
    !Number.isInteger(calledHandIndex) ||
    calledHandIndex < 0 ||
    calledHandIndex >= rules.handsCount
  ) {
    throw new RangeError(`calledHandIndex ${calledHandIndex} out of bounds for handsCount=${rules.handsCount}`);
  }
  if (progress.openedEmptyHands.includes(calledHandIndex)) {
    throw new Error(`Hand ${calledHandIndex} is already open`);
  }

  const effectiveCallType: GuessCallType =
    rules.allowPoochCalls && callType === 'POOCH' ? 'POOCH' : 'GOL';

  const isGolHand = calledHandIndex === secretGolHandIndex;

  if (effectiveCallType === 'GOL') {
    if (isGolHand) {
      const outcome: RoundGuessOutcome =
        progress.poochCallsCount === 0 ? 'CORRECT_DIRECT' : 'CORRECT_AFTER_POOCH';
      return {
        roundEnded: true,
        outcome,
        openedHandIndex: calledHandIndex,
        nextProgress: {
          ...progress,
          guessesUsed: progress.guessesUsed + 1,
        },
        reason: progress.poochCallsCount === 0 ? 'GOL_DIRECT' : 'GOL_AFTER_POOCH',
      };
    }

    return {
      roundEnded: true,
      outcome: 'WRONG',
      openedHandIndex: calledHandIndex,
      nextProgress: {
        guessesUsed: progress.guessesUsed + 1,
        poochCallsCount: progress.poochCallsCount,
        openedEmptyHands: [...progress.openedEmptyHands, calledHandIndex],
      },
      reason: 'WRONG_GOL',
    };
  }

  // POOCH call
  if (isGolHand) {
    return {
      roundEnded: true,
      outcome: 'WRONG',
      openedHandIndex: calledHandIndex,
      nextProgress: {
        ...progress,
        guessesUsed: progress.guessesUsed + 1,
      },
      reason: 'POOCH_HIT_GOL',
    };
  }

  const nextGuessesUsed = progress.guessesUsed + 1;
  const nextPoochCalls = progress.poochCallsCount + 1;
  const nextOpened = [...progress.openedEmptyHands, calledHandIndex];
  const remainingClosed = rules.handsCount - nextOpened.length;

  if (nextGuessesUsed >= rules.maxGuessesPerRound || remainingClosed <= 0) {
    return {
      roundEnded: true,
      outcome: 'WRONG',
      openedHandIndex: calledHandIndex,
      nextProgress: {
        guessesUsed: nextGuessesUsed,
        poochCallsCount: nextPoochCalls,
        openedEmptyHands: nextOpened,
      },
      reason: 'OUT_OF_GUESSES',
    };
  }

  return {
    roundEnded: false,
    outcome: null,
    openedHandIndex: calledHandIndex,
    nextProgress: {
      guessesUsed: nextGuessesUsed,
      poochCallsCount: nextPoochCalls,
      openedEmptyHands: nextOpened,
    },
    reason: 'EMPTY_POOCH_OPENED',
  };
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
