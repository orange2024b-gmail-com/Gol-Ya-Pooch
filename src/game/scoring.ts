import {
  CLASSIC_1V1_RULES,
  resolveRoundByRules,
  type PlayerRole,
  type RoundGuessOutcome,
  type RuleSetDefinition,
} from './rules/engine';

export interface MatchScoreState {
  rules: RuleSetDefinition;
  humanScore: number;
  computerScore: number;
  currentHider: PlayerRole;
  currentGuesser: PlayerRole;
  validRoundsCompleted: number;
  roundNumber: number;
  lastRoundWinner: PlayerRole | null;
  lastTurnPassed: boolean;
  matchWinner: PlayerRole | null;
}

export function createMatchScoreState(
  rules: RuleSetDefinition = CLASSIC_1V1_RULES,
  initialHider: PlayerRole = 'HUMAN'
): MatchScoreState {
  return {
    rules,
    humanScore: 0,
    computerScore: 0,
    currentHider: initialHider,
    currentGuesser: initialHider === 'HUMAN' ? 'COMPUTER' : 'HUMAN',
    validRoundsCompleted: 0,
    roundNumber: 1,
    lastRoundWinner: null,
    lastTurnPassed: false,
    matchWinner: null,
  };
}

export function isMatchOver(state: MatchScoreState): boolean {
  return state.matchWinner !== null;
}

export function applyRoundOutcomeToScore(
  state: MatchScoreState,
  guessOutcome: RoundGuessOutcome
): MatchScoreState {
  const resolution = resolveRoundByRules(state.rules, {
    hider: state.currentHider,
    guesser: state.currentGuesser,
    guessOutcome,
  });

  const humanScore = state.humanScore + resolution.humanDelta;
  const computerScore = state.computerScore + resolution.computerDelta;
  const validRoundsCompleted =
    state.validRoundsCompleted + (resolution.isValidRound ? 1 : 0);

  let matchWinner: PlayerRole | null = state.matchWinner;
  if (matchWinner === null) {
    if (humanScore >= state.rules.targetScore) {
      matchWinner = 'HUMAN';
    } else if (computerScore >= state.rules.targetScore) {
      matchWinner = 'COMPUTER';
    }
  }

  return {
    ...state,
    humanScore,
    computerScore,
    currentHider: resolution.nextHider,
    currentGuesser: resolution.nextGuesser,
    validRoundsCompleted,
    roundNumber: resolution.isValidRound ? state.roundNumber + 1 : state.roundNumber,
    lastRoundWinner: resolution.winner,
    lastTurnPassed: resolution.turnPassed,
    matchWinner,
  };
}
