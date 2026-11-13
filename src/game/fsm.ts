import type { HandSide, RevealOutcome } from '../vision/types';
import type { CommitPublicInfo, CommitSecretReveal } from './commitReveal';
import {
  buildRuleSetForLevel,
  getLevelDefinition,
  type LevelId,
} from './levels';
import {
  CLASSIC_1V1_RULES,
  evaluateGuessCallStep,
  loadRuleSet,
  resolveRoundByRules,
  type GuessCallType,
  type PlayerRole,
  type RuleSetDefinition,
  type TeamGuessProgress,
} from './rules/engine';

export const ALL_FSM_STATES = [
  'SPLASH',
  'PERMISSION',
  'CALIBRATION',
  'MENU',
  'LEVEL_SELECT',
  'STATS',
  'RULES',
  'CONSENT',
  'BENCHMARK',
  'SHOW_GOL',
  'HIDING',
  'COMPUTER_GUESS',
  'REVEAL_VERIFY',
  'COMPUTER_HIDES',
  'HUMAN_GUESS',
  'ROUND_RESULT',
  'ROUND_VOID',
  'PAUSED',
  'CAMERA_ERROR',
] as const;

export type GameStateName = (typeof ALL_FSM_STATES)[number];

export type VoidReason =
  | 'HANDS_LOST'
  | 'SHOW_TIMEOUT'
  | 'NO_HAND_CONTACT'
  | 'REVEAL_NEITHER'
  | 'REVEAL_BOTH'
  | 'REVEAL_TIMEOUT'
  | 'CAMERA_INTERRUPTED';

export interface FsmConfig {
  showDurationMs: number;
  hideDurationMs: number;
  guessDisplayMs: number;
  revealDurationMs: number;
  computerShuffleDurationMs: number;
  humanGuessDurationMs: number;
  maxHandLostMs: number;
  targetScore: number;
  minHandContactsRequired: number;
  enableTurnAlternation: boolean;
}

export const DEFAULT_FSM_CONFIG: FsmConfig = {
  showDurationMs: CLASSIC_1V1_RULES.timeoutsMs.showGol,
  hideDurationMs: CLASSIC_1V1_RULES.timeoutsMs.hiding,
  guessDisplayMs: CLASSIC_1V1_RULES.timeoutsMs.guessDisplay,
  revealDurationMs: CLASSIC_1V1_RULES.timeoutsMs.revealVerify,
  computerShuffleDurationMs: 2200,
  humanGuessDurationMs: 20_000,
  maxHandLostMs: CLASSIC_1V1_RULES.voidConditions.maxHandLostMs,
  targetScore: CLASSIC_1V1_RULES.targetScore,
  minHandContactsRequired: 0,
  enableTurnAlternation: true,
};

export interface AiGuessPayload {
  guess: HandSide;
  probability: number;
  explanationFa: string;
  timelineEvents?: string[];
}

export interface ComputerHideSetupPayload {
  secretGolHandIndex: number;
  commitPublic: CommitPublicInfo;
  commitSecret: CommitSecretReveal;
  shuffleDurationMs?: number;
}

export interface GameContext {
  state: GameStateName;
  calibrated: boolean;
  practiceMode: boolean;
  levelId: LevelId;
  teamHandsCount: 4 | 6;
  bestOfMatches: 1 | 3;
  seriesMatchIndex: number;
  seriesHumanWins: number;
  seriesComputerWins: number;
  seriesWinner: PlayerRole | null;
  ruleSet: RuleSetDefinition;
  ruleHiderOwner: PlayerRole;
  ruleGuesserOwner: PlayerRole;
  roundNumber: number;
  validRoundsCompleted: number;
  totalAwardedPoints: number;
  humanScore: number;
  computerScore: number;
  matchWinner: PlayerRole | null;
  phaseRemainingMs: number;
  pinchConfirmed: boolean;
  initialShownHand: HandSide | null;
  hidingContactCount: number;
  computerGuess: HandSide | null;
  computerGuessProb: number | null;
  explanationFa: string | null;
  timelineEvents: string[];
  revealedHand: HandSide | null;
  lastRoundWinner: 'HUMAN' | 'COMPUTER' | null;
  // Computer-hides / Human-guesses state
  computerSecretGolHandIndex: number | null;
  commitPublic: CommitPublicInfo | null;
  commitSecret: CommitSecretReveal | null;
  teamGuessProgress: TeamGuessProgress;
  lastHumanCalledHandIndex: number | null;
  lastHumanCallType: GuessCallType | null;
  lastComputerHidesOutcomeReason: string | null;
  voidReason: VoidReason | null;
  voidMessageFa: string | null;
  pausedFrom: Exclude<GameStateName, 'PAUSED' | 'CAMERA_ERROR'> | null;
  cameraErrorReturnState: Exclude<GameStateName, 'CAMERA_ERROR' | 'PAUSED'> | null;
  cameraErrorMessageFa: string | null;
  config: FsmConfig;
}

export type GameEvent =
  | { type: 'GO_PERMISSION' }
  | { type: 'ENTER_PRACTICE_MENU' }
  | { type: 'OPEN_LEVEL_SELECT' }
  | { type: 'OPEN_STATS' }
  | { type: 'OPEN_RULES' }
  | { type: 'OPEN_CONSENT' }
  | { type: 'OPEN_BENCHMARK' }
  | {
      type: 'SELECT_LEVEL';
      levelId: LevelId;
      teamHandsCount?: 4 | 6;
      customRuleSet?: RuleSetDefinition;
    }
  | { type: 'SET_TEAM_HANDS_COUNT'; handsCount: 4 | 6 }
  | { type: 'SET_CUSTOM_RULES'; ruleSet: RuleSetDefinition }
  | { type: 'CAMERA_READY' }
  | { type: 'CALIBRATION_COMPLETE' }
  | { type: 'OPEN_CALIBRATION' }
  | { type: 'SET_TARGET_SCORE'; targetScore: number }
  | { type: 'START_GAME'; initialHider?: PlayerRole; computerHideSetup?: ComputerHideSetupPayload }
  | { type: 'PINCH_CONFIRMED'; side: HandSide }
  | {
      type: 'FINISH_HIDING';
      aiGuess?: AiGuessPayload;
      contactCount?: number;
    }
  | { type: 'PROCEED_TO_REVEAL' }
  | { type: 'REVEAL_VERIFIED'; outcome: RevealOutcome }
  | { type: 'SET_COMPUTER_COMMIT'; setup: ComputerHideSetupPayload }
  | { type: 'COMPUTER_SHUFFLE_DONE' }
  | { type: 'HUMAN_GUESS_CALL'; handIndex: number; callType: GuessCallType }
  | { type: 'NEXT_ROUND'; computerHideSetup?: ComputerHideSetupPayload }
  | { type: 'RESET_MATCH'; initialHider?: PlayerRole; computerHideSetup?: ComputerHideSetupPayload }
  | { type: 'EXIT_TO_MENU' }
  | {
      type: 'TICK';
      dtMs: number;
      maxLostDurationMs?: number;
      randomBit?: 0 | 1;
      aiGuess?: AiGuessPayload;
      contactCount?: number;
    }
  | { type: 'TAB_HIDDEN' }
  | { type: 'TAB_VISIBLE' }
  | { type: 'CAMERA_LOST'; messageFa?: string }
  | { type: 'CAMERA_RECOVERED' }
  | { type: 'RETRY_PERMISSION' };

export const VOID_MESSAGES_FA: Record<VoidReason, string> = {
  HANDS_LOST: 'دست‌ها بیش از نیم ثانیه از کادر خارج شدند؛ دور باطل شد.',
  SHOW_TIMEOUT: 'زمان نمایش گل به پایان رسید؛ دور باطل شد.',
  NO_HAND_CONTACT:
    'در سطح حرفه‌ای حداقل یک بار تماس دست‌ها در حین مخفی‌سازی الزامی است؛ دور باطل شد.',
  REVEAL_NEITHER: 'گل در هیچ‌کدام از دست‌ها دیده نشد؛ دور باطل شد.',
  REVEAL_BOTH: 'گل در هر دو دست تشخیص داده شد؛ دور باطل شد.',
  REVEAL_TIMEOUT: 'زمان باز کردن هر دو دست به پایان رسید؛ دور باطل شد.',
  CAMERA_INTERRUPTED: 'ارتباط دوربین قطع و دوباره وصل شد؛ دور فعلی باطل و تکرار می‌شود.',
};

const EMPTY_TEAM_PROGRESS: TeamGuessProgress = {
  guessesUsed: 0,
  poochCallsCount: 0,
  openedEmptyHands: [],
};

export function createInitialGameContext(
  config: Partial<FsmConfig> = {},
  ruleSet: RuleSetDefinition = CLASSIC_1V1_RULES,
  levelId: LevelId = 1
): GameContext {
  const level = getLevelDefinition(levelId);
  const mergedConfig: FsmConfig = {
    showDurationMs: ruleSet.timeoutsMs.showGol,
    hideDurationMs: ruleSet.timeoutsMs.hiding,
    guessDisplayMs: ruleSet.timeoutsMs.guessDisplay,
    revealDurationMs: ruleSet.timeoutsMs.revealVerify,
    computerShuffleDurationMs: level.animation.shuffleDurationMs,
    humanGuessDurationMs: level.humanGuessDurationMs,
    maxHandLostMs: ruleSet.voidConditions.maxHandLostMs,
    targetScore: ruleSet.targetScore,
    minHandContactsRequired: level.minHandContactsRequired,
    enableTurnAlternation: true,
    ...config,
  };
  const effectiveRules = loadRuleSet(ruleSet, {
    targetScore: mergedConfig.targetScore,
    hidingTimeoutMs: mergedConfig.hideDurationMs,
  });

  return {
    state: 'SPLASH',
    calibrated: false,
    practiceMode: false,
    levelId,
    teamHandsCount: 4,
    bestOfMatches: level.bestOfMatches,
    seriesMatchIndex: 0,
    seriesHumanWins: 0,
    seriesComputerWins: 0,
    seriesWinner: null,
    ruleSet: effectiveRules,
    ruleHiderOwner: 'HUMAN',
    ruleGuesserOwner: 'COMPUTER',
    roundNumber: 1,
    validRoundsCompleted: 0,
    totalAwardedPoints: 0,
    humanScore: 0,
    computerScore: 0,
    matchWinner: null,
    phaseRemainingMs: 0,
    pinchConfirmed: false,
    initialShownHand: null,
    hidingContactCount: 0,
    computerGuess: null,
    computerGuessProb: null,
    explanationFa: null,
    timelineEvents: [],
    revealedHand: null,
    lastRoundWinner: null,
    computerSecretGolHandIndex: null,
    commitPublic: null,
    commitSecret: null,
    teamGuessProgress: { ...EMPTY_TEAM_PROGRESS, openedEmptyHands: [] },
    lastHumanCalledHandIndex: null,
    lastHumanCallType: null,
    lastComputerHidesOutcomeReason: null,
    voidReason: null,
    voidMessageFa: null,
    pausedFrom: null,
    cameraErrorReturnState: null,
    cameraErrorMessageFa: null,
    config: mergedConfig,
  };
}

function makeFallbackGuess(randomBit?: 0 | 1): AiGuessPayload {
  let guess: HandSide;
  if (randomBit === 0) {
    guess = 'LEFT';
  } else if (randomBit === 1) {
    guess = 'RIGHT';
  } else if (
    typeof globalThis.crypto !== 'undefined' &&
    typeof globalThis.crypto.getRandomValues === 'function'
  ) {
    const arr = new Uint8Array(1);
    globalThis.crypto.getRandomValues(arr);
    guess = (arr[0]! & 1) === 0 ? 'LEFT' : 'RIGHT';
  } else {
    guess = Math.random() < 0.5 ? 'LEFT' : 'RIGHT';
  }
  return {
    guess,
    probability: 0.5,
    explanationFa: 'بر اساس اطلاعات موجود، نزدیک‌ترین گزینه انتخاب شد.',
    timelineEvents: [],
  };
}

function transitionToVoid(ctx: GameContext, reason: VoidReason): GameContext {
  return {
    ...ctx,
    state: 'ROUND_VOID',
    phaseRemainingMs: 0,
    voidReason: reason,
    voidMessageFa: VOID_MESSAGES_FA[reason],
    revealedHand: null,
    lastRoundWinner: null,
  };
}

function applyRoundOutcomeToContext(
  ctx: GameContext,
  hider: PlayerRole,
  guesser: PlayerRole,
  guessOutcome: 'CORRECT_DIRECT' | 'CORRECT_AFTER_POOCH' | 'WRONG',
  extras: Partial<GameContext>
): GameContext {
  const resolution = resolveRoundByRules(ctx.ruleSet, {
    hider,
    guesser,
    guessOutcome,
  });

  const nextHumanScore = ctx.humanScore + resolution.humanDelta;
  const nextComputerScore = ctx.computerScore + resolution.computerDelta;
  let matchWinner = ctx.matchWinner;
  if (matchWinner === null) {
    if (nextHumanScore >= ctx.ruleSet.targetScore) {
      matchWinner = 'HUMAN';
    } else if (nextComputerScore >= ctx.ruleSet.targetScore) {
      matchWinner = 'COMPUTER';
    }
  }

  let seriesHumanWins = ctx.seriesHumanWins;
  let seriesComputerWins = ctx.seriesComputerWins;
  let seriesWinner = ctx.seriesWinner;

  if (matchWinner !== null && ctx.matchWinner === null) {
    if (ctx.bestOfMatches === 3) {
      if (matchWinner === 'HUMAN') seriesHumanWins += 1;
      else seriesComputerWins += 1;

      if (seriesHumanWins >= 2) seriesWinner = 'HUMAN';
      else if (seriesComputerWins >= 2) seriesWinner = 'COMPUTER';
    } else {
      seriesWinner = matchWinner;
    }
  }

  const nextHider = ctx.config.enableTurnAlternation ? resolution.nextHider : 'HUMAN';
  const nextGuesser = nextHider === 'HUMAN' ? 'COMPUTER' : 'HUMAN';

  return {
    ...ctx,
    ...extras,
    state: 'ROUND_RESULT',
    phaseRemainingMs: 0,
    validRoundsCompleted: ctx.validRoundsCompleted + 1,
    totalAwardedPoints: ctx.totalAwardedPoints + resolution.totalPointsAwarded,
    humanScore: nextHumanScore,
    computerScore: nextComputerScore,
    ruleHiderOwner: nextHider,
    ruleGuesserOwner: nextGuesser,
    matchWinner,
    seriesHumanWins,
    seriesComputerWins,
    seriesWinner,
    lastRoundWinner: resolution.winner,
  };
}

function startRound(
  ctx: GameContext,
  options: {
    resetMatch?: boolean;
    resetSeries?: boolean;
    initialHider?: PlayerRole;
    computerHideSetup?: ComputerHideSetupPayload;
  } = {}
): GameContext {
  const resetMatch = options.resetMatch ?? false;
  const resetSeries = options.resetSeries ?? false;

  const hider: PlayerRole = resetMatch
    ? (options.initialHider ?? (ctx.practiceMode ? 'COMPUTER' : 'HUMAN'))
    : ctx.practiceMode
      ? 'COMPUTER'
      : ctx.ruleHiderOwner;
  const guesser: PlayerRole = hider === 'HUMAN' ? 'COMPUTER' : 'HUMAN';

  const setup = options.computerHideSetup;
  const nextRoundNumber = resetMatch ? 1 : ctx.roundNumber;

  const base: GameContext = {
    ...ctx,
    ruleHiderOwner: hider,
    ruleGuesserOwner: guesser,
    roundNumber: nextRoundNumber,
    validRoundsCompleted: resetMatch ? 0 : ctx.validRoundsCompleted,
    totalAwardedPoints: resetMatch ? 0 : ctx.totalAwardedPoints,
    humanScore: resetMatch ? 0 : ctx.humanScore,
    computerScore: resetMatch ? 0 : ctx.computerScore,
    matchWinner: resetMatch ? null : ctx.matchWinner,
    seriesMatchIndex: resetSeries ? 0 : ctx.seriesMatchIndex,
    seriesHumanWins: resetSeries ? 0 : ctx.seriesHumanWins,
    seriesComputerWins: resetSeries ? 0 : ctx.seriesComputerWins,
    seriesWinner: resetSeries ? null : ctx.seriesWinner,
    pinchConfirmed: false,
    initialShownHand: null,
    hidingContactCount: 0,
    computerGuess: null,
    computerGuessProb: null,
    explanationFa: null,
    timelineEvents: [],
    revealedHand: null,
    lastRoundWinner: null,
    computerSecretGolHandIndex: setup?.secretGolHandIndex ?? null,
    commitPublic: setup?.commitPublic ?? null,
    commitSecret: setup?.commitSecret ?? null,
    teamGuessProgress: { ...EMPTY_TEAM_PROGRESS, openedEmptyHands: [] },
    lastHumanCalledHandIndex: null,
    lastHumanCallType: null,
    lastComputerHidesOutcomeReason: null,
    voidReason: null,
    voidMessageFa: null,
  };

  if (hider === 'COMPUTER') {
    return {
      ...base,
      state: 'COMPUTER_HIDES',
      phaseRemainingMs: setup?.shuffleDurationMs ?? ctx.config.computerShuffleDurationMs,
    };
  }

  return {
    ...base,
    state: 'SHOW_GOL',
    phaseRemainingMs: ctx.config.showDurationMs,
  };
}

function applyLevelConfiguration(
  ctx: GameContext,
  levelId: LevelId,
  teamHandsCount?: 4 | 6,
  customRuleSet?: RuleSetDefinition
): GameContext {
  const level = getLevelDefinition(levelId);
  const hands = teamHandsCount ?? ctx.teamHandsCount;
  const nextRules =
    customRuleSet ??
    buildRuleSetForLevel(levelId, {
      seriesMatchIndex: 0,
      targetScore: ctx.config.targetScore,
      teamHandsCount: hands,
    });

  return {
    ...ctx,
    levelId,
    teamHandsCount: hands,
    bestOfMatches: level.bestOfMatches,
    seriesMatchIndex: 0,
    seriesHumanWins: 0,
    seriesComputerWins: 0,
    seriesWinner: null,
    ruleSet: nextRules,
    config: {
      ...ctx.config,
      showDurationMs: nextRules.timeoutsMs.showGol,
      hideDurationMs: level.humanHideDurationMs,
      guessDisplayMs: nextRules.timeoutsMs.guessDisplay,
      revealDurationMs: nextRules.timeoutsMs.revealVerify,
      computerShuffleDurationMs: level.animation.shuffleDurationMs,
      humanGuessDurationMs: level.humanGuessDurationMs,
      minHandContactsRequired: level.minHandContactsRequired,
      targetScore: nextRules.targetScore,
    },
  };
}

export function transitionGameFsm(ctx: GameContext, event: GameEvent): GameContext {
  if (event.type === 'SET_TARGET_SCORE') {
    const clamped = Math.max(1, Math.min(50, Math.round(event.targetScore)));
    const updatedRules = loadRuleSet(ctx.ruleSet, { targetScore: clamped });
    return {
      ...ctx,
      ruleSet: updatedRules,
      config: { ...ctx.config, targetScore: clamped },
    };
  }

  if (event.type === 'SET_TEAM_HANDS_COUNT') {
    const updatedRules =
      ctx.ruleSet.id === 'traditional-team'
        ? loadRuleSet(ctx.ruleSet, { handsCount: event.handsCount })
        : ctx.ruleSet;
    return {
      ...ctx,
      teamHandsCount: event.handsCount,
      ruleSet: updatedRules,
    };
  }

  if (event.type === 'SET_CUSTOM_RULES') {
    const validated = loadRuleSet(event.ruleSet);
    return {
      ...ctx,
      ruleSet: validated,
      config: {
        ...ctx.config,
        showDurationMs: validated.timeoutsMs.showGol,
        hideDurationMs: validated.timeoutsMs.hiding,
        guessDisplayMs: validated.timeoutsMs.guessDisplay,
        revealDurationMs: validated.timeoutsMs.revealVerify,
        targetScore: validated.targetScore,
      },
    };
  }

  if (event.type === 'SELECT_LEVEL') {
    return applyLevelConfiguration(ctx, event.levelId, event.teamHandsCount, event.customRuleSet);
  }

  // Global tab-hidden pause handling
  if (event.type === 'TAB_HIDDEN') {
    if (
      ctx.state === 'SPLASH' ||
      ctx.state === 'PERMISSION' ||
      ctx.state === 'PAUSED' ||
      ctx.state === 'CAMERA_ERROR'
    ) {
      return ctx;
    }
    return {
      ...ctx,
      pausedFrom: ctx.state,
      state: 'PAUSED',
    };
  }

  if (event.type === 'TAB_VISIBLE') {
    if (ctx.state !== 'PAUSED' || !ctx.pausedFrom) {
      return ctx;
    }
    return {
      ...ctx,
      state: ctx.pausedFrom,
      pausedFrom: null,
    };
  }

  // Global recoverable camera loss handling
  if (event.type === 'CAMERA_LOST') {
    if (ctx.state === 'SPLASH' || ctx.state === 'CAMERA_ERROR' || ctx.practiceMode) {
      return ctx;
    }
    const baseState = ctx.state === 'PAUSED' ? (ctx.pausedFrom ?? 'MENU') : ctx.state;
    const isActiveCameraRoundPhase =
      baseState === 'SHOW_GOL' ||
      baseState === 'HIDING' ||
      baseState === 'COMPUTER_GUESS' ||
      baseState === 'REVEAL_VERIFY';

    return {
      ...ctx,
      state: 'CAMERA_ERROR',
      pausedFrom: null,
      cameraErrorReturnState: isActiveCameraRoundPhase ? 'ROUND_VOID' : baseState,
      voidReason: isActiveCameraRoundPhase ? 'CAMERA_INTERRUPTED' : ctx.voidReason,
      voidMessageFa: isActiveCameraRoundPhase ? VOID_MESSAGES_FA.CAMERA_INTERRUPTED : ctx.voidMessageFa,
      cameraErrorMessageFa:
        event.messageFa ?? 'دسترسی به دوربین قطع شد. لطفاً دوربین را بررسی و دوباره تلاش کنید.',
    };
  }

  switch (ctx.state) {
    case 'SPLASH': {
      if (event.type === 'GO_PERMISSION') {
        return { ...ctx, practiceMode: false, state: 'PERMISSION' };
      }
      if (event.type === 'ENTER_PRACTICE_MENU') {
        return { ...ctx, practiceMode: true, state: 'MENU' };
      }
      if (event.type === 'OPEN_BENCHMARK') {
        return { ...ctx, practiceMode: true, state: 'BENCHMARK' };
      }
      return ctx;
    }

    case 'PERMISSION': {
      if (event.type === 'CAMERA_READY') {
        return { ...ctx, practiceMode: false, state: ctx.calibrated ? 'MENU' : 'CALIBRATION' };
      }
      if (event.type === 'ENTER_PRACTICE_MENU') {
        return { ...ctx, practiceMode: true, state: 'MENU' };
      }
      return ctx;
    }

    case 'CALIBRATION': {
      if (event.type === 'CALIBRATION_COMPLETE') {
        return { ...ctx, calibrated: true, practiceMode: false, state: 'MENU' };
      }
      if (event.type === 'EXIT_TO_MENU' && (ctx.calibrated || ctx.practiceMode)) {
        return { ...ctx, state: 'MENU' };
      }
      return ctx;
    }

    case 'MENU': {
      if (event.type === 'OPEN_CALIBRATION') {
        return { ...ctx, state: 'CALIBRATION' };
      }
      if (event.type === 'OPEN_LEVEL_SELECT') {
        return { ...ctx, state: 'LEVEL_SELECT' };
      }
      if (event.type === 'OPEN_STATS') {
        return { ...ctx, state: 'STATS' };
      }
      if (event.type === 'OPEN_RULES') {
        return { ...ctx, state: 'RULES' };
      }
      if (event.type === 'OPEN_CONSENT') {
        return { ...ctx, state: 'CONSENT' };
      }
      if (event.type === 'OPEN_BENCHMARK') {
        return { ...ctx, state: 'BENCHMARK' };
      }
      if (event.type === 'START_GAME') {
        if (!ctx.calibrated && !ctx.practiceMode) {
          return { ...ctx, state: 'CALIBRATION' };
        }
        return startRound(ctx, {
          resetMatch: true,
          resetSeries: true,
          ...(event.initialHider !== undefined ? { initialHider: event.initialHider } : {}),
          ...(event.computerHideSetup !== undefined
            ? { computerHideSetup: event.computerHideSetup }
            : {}),
        });
      }
      return ctx;
    }

    case 'LEVEL_SELECT':
    case 'STATS':
    case 'RULES':
    case 'CONSENT':
    case 'BENCHMARK': {
      if (event.type === 'EXIT_TO_MENU') {
        return { ...ctx, state: 'MENU' };
      }
      if (event.type === 'OPEN_CONSENT' && ctx.state === 'STATS') {
        return { ...ctx, state: 'CONSENT' };
      }
      if (event.type === 'START_GAME' && ctx.state === 'LEVEL_SELECT') {
        if (!ctx.calibrated && !ctx.practiceMode) {
          return { ...ctx, state: 'CALIBRATION' };
        }
        return startRound(ctx, {
          resetMatch: true,
          resetSeries: true,
          ...(event.initialHider !== undefined ? { initialHider: event.initialHider } : {}),
          ...(event.computerHideSetup !== undefined
            ? { computerHideSetup: event.computerHideSetup }
            : {}),
        });
      }
      return ctx;
    }

    case 'SHOW_GOL': {
      if (event.type === 'EXIT_TO_MENU') {
        return { ...ctx, state: 'MENU', phaseRemainingMs: 0 };
      }
      if (event.type === 'PINCH_CONFIRMED') {
        return {
          ...ctx,
          state: 'HIDING',
          pinchConfirmed: true,
          initialShownHand: event.side,
          hidingContactCount: 0,
          phaseRemainingMs: ctx.config.hideDurationMs,
        };
      }
      if (event.type === 'TICK') {
        const lostMs = event.maxLostDurationMs ?? 0;
        if (lostMs > ctx.config.maxHandLostMs) {
          return transitionToVoid(ctx, 'HANDS_LOST');
        }
        const remaining = Math.max(0, ctx.phaseRemainingMs - Math.max(0, event.dtMs));
        if (remaining <= 0) {
          return transitionToVoid(ctx, 'SHOW_TIMEOUT');
        }
        return { ...ctx, phaseRemainingMs: remaining };
      }
      return ctx;
    }

    case 'HIDING': {
      if (event.type === 'EXIT_TO_MENU') {
        return { ...ctx, state: 'MENU', phaseRemainingMs: 0 };
      }
      if (event.type === 'FINISH_HIDING') {
        const totalContacts = Math.max(ctx.hidingContactCount, event.contactCount ?? 0);
        if (
          ctx.config.minHandContactsRequired > 0 &&
          totalContacts < ctx.config.minHandContactsRequired
        ) {
          return transitionToVoid(ctx, 'NO_HAND_CONTACT');
        }
        const payload = event.aiGuess ?? makeFallbackGuess();
        return {
          ...ctx,
          hidingContactCount: totalContacts,
          state: 'COMPUTER_GUESS',
          computerGuess: payload.guess,
          computerGuessProb: payload.probability,
          explanationFa: payload.explanationFa,
          timelineEvents: payload.timelineEvents ?? [],
          phaseRemainingMs: ctx.config.guessDisplayMs,
        };
      }
      if (event.type === 'TICK') {
        const lostMs = event.maxLostDurationMs ?? 0;
        if (lostMs > ctx.config.maxHandLostMs) {
          return transitionToVoid(ctx, 'HANDS_LOST');
        }
        const totalContacts = Math.max(ctx.hidingContactCount, event.contactCount ?? 0);
        const remaining = Math.max(0, ctx.phaseRemainingMs - Math.max(0, event.dtMs));
        if (remaining <= 0) {
          if (
            ctx.config.minHandContactsRequired > 0 &&
            totalContacts < ctx.config.minHandContactsRequired
          ) {
            return transitionToVoid(ctx, 'NO_HAND_CONTACT');
          }
          const payload = event.aiGuess ?? makeFallbackGuess(event.randomBit);
          return {
            ...ctx,
            hidingContactCount: totalContacts,
            state: 'COMPUTER_GUESS',
            computerGuess: payload.guess,
            computerGuessProb: payload.probability,
            explanationFa: payload.explanationFa,
            timelineEvents: payload.timelineEvents ?? [],
            phaseRemainingMs: ctx.config.guessDisplayMs,
          };
        }
        return {
          ...ctx,
          hidingContactCount: totalContacts,
          phaseRemainingMs: remaining,
        };
      }
      return ctx;
    }

    case 'COMPUTER_GUESS': {
      if (event.type === 'EXIT_TO_MENU') {
        return { ...ctx, state: 'MENU', phaseRemainingMs: 0 };
      }
      if (event.type === 'PROCEED_TO_REVEAL') {
        return {
          ...ctx,
          state: 'REVEAL_VERIFY',
          phaseRemainingMs: ctx.config.revealDurationMs,
        };
      }
      if (event.type === 'TICK') {
        const remaining = Math.max(0, ctx.phaseRemainingMs - Math.max(0, event.dtMs));
        if (remaining <= 0) {
          return {
            ...ctx,
            state: 'REVEAL_VERIFY',
            phaseRemainingMs: ctx.config.revealDurationMs,
          };
        }
        return { ...ctx, phaseRemainingMs: remaining };
      }
      return ctx;
    }

    case 'REVEAL_VERIFY': {
      if (event.type === 'EXIT_TO_MENU') {
        return { ...ctx, state: 'MENU', phaseRemainingMs: 0 };
      }
      if (event.type === 'REVEAL_VERIFIED') {
        if (event.outcome === 'HANDS_NOT_OPEN') {
          return ctx;
        }
        if (event.outcome === 'NEITHER') {
          return transitionToVoid(ctx, 'REVEAL_NEITHER');
        }
        if (event.outcome === 'BOTH') {
          return transitionToVoid(ctx, 'REVEAL_BOTH');
        }

        const computerWon = event.outcome === ctx.computerGuess;
        return applyRoundOutcomeToContext(
          ctx,
          'HUMAN',
          'COMPUTER',
          computerWon ? 'CORRECT_DIRECT' : 'WRONG',
          {
            revealedHand: event.outcome,
          }
        );
      }
      if (event.type === 'TICK') {
        const remaining = Math.max(0, ctx.phaseRemainingMs - Math.max(0, event.dtMs));
        if (remaining <= 0) {
          return transitionToVoid(ctx, 'REVEAL_TIMEOUT');
        }
        return { ...ctx, phaseRemainingMs: remaining };
      }
      return ctx;
    }

    case 'COMPUTER_HIDES': {
      if (event.type === 'EXIT_TO_MENU') {
        return { ...ctx, state: 'MENU', phaseRemainingMs: 0 };
      }
      if (event.type === 'SET_COMPUTER_COMMIT') {
        return {
          ...ctx,
          computerSecretGolHandIndex: event.setup.secretGolHandIndex,
          commitPublic: event.setup.commitPublic,
          commitSecret: event.setup.commitSecret,
          ...(event.setup.shuffleDurationMs !== undefined
            ? { phaseRemainingMs: event.setup.shuffleDurationMs }
            : {}),
        };
      }
      if (event.type === 'COMPUTER_SHUFFLE_DONE') {
        return {
          ...ctx,
          state: 'HUMAN_GUESS',
          phaseRemainingMs: ctx.config.humanGuessDurationMs,
        };
      }
      if (event.type === 'TICK') {
        const remaining = Math.max(0, ctx.phaseRemainingMs - Math.max(0, event.dtMs));
        if (remaining <= 0) {
          return {
            ...ctx,
            state: 'HUMAN_GUESS',
            phaseRemainingMs: ctx.config.humanGuessDurationMs,
          };
        }
        return { ...ctx, phaseRemainingMs: remaining };
      }
      return ctx;
    }

    case 'HUMAN_GUESS': {
      if (event.type === 'EXIT_TO_MENU') {
        return { ...ctx, state: 'MENU', phaseRemainingMs: 0 };
      }
      if (event.type === 'SET_COMPUTER_COMMIT') {
        return {
          ...ctx,
          computerSecretGolHandIndex: event.setup.secretGolHandIndex,
          commitPublic: event.setup.commitPublic,
          commitSecret: event.setup.commitSecret,
        };
      }
      if (event.type === 'HUMAN_GUESS_CALL') {
        const secretIdx = ctx.computerSecretGolHandIndex ?? 0;
        if (
          event.handIndex < 0 ||
          event.handIndex >= ctx.ruleSet.handsCount ||
          ctx.teamGuessProgress.openedEmptyHands.includes(event.handIndex)
        ) {
          return ctx;
        }

        const step = evaluateGuessCallStep(
          ctx.ruleSet,
          ctx.teamGuessProgress,
          secretIdx,
          event.handIndex,
          event.callType
        );

        if (!step.roundEnded || !step.outcome) {
          return {
            ...ctx,
            teamGuessProgress: step.nextProgress,
            lastHumanCalledHandIndex: event.handIndex,
            lastHumanCallType: event.callType,
          };
        }

        const outcome = step.outcome === 'VOID' ? 'WRONG' : step.outcome;
        return applyRoundOutcomeToContext(ctx, 'COMPUTER', 'HUMAN', outcome, {
          teamGuessProgress: step.nextProgress,
          lastHumanCalledHandIndex: event.handIndex,
          lastHumanCallType: event.callType,
          lastComputerHidesOutcomeReason: step.reason,
          revealedHand:
            ctx.ruleSet.handsCount === 2 ? (secretIdx === 0 ? 'LEFT' : 'RIGHT') : null,
        });
      }
      if (event.type === 'TICK') {
        const remaining = Math.max(0, ctx.phaseRemainingMs - Math.max(0, event.dtMs));
        if (remaining <= 0) {
          return applyRoundOutcomeToContext(ctx, 'COMPUTER', 'HUMAN', 'WRONG', {
            lastComputerHidesOutcomeReason: 'OUT_OF_GUESSES',
          });
        }
        return { ...ctx, phaseRemainingMs: remaining };
      }
      return ctx;
    }

    case 'ROUND_RESULT': {
      if (event.type === 'NEXT_ROUND') {
        if (ctx.matchWinner !== null) {
          // If best-of-3 series has another match left, advance seriesMatchIndex
          if (ctx.bestOfMatches === 3 && ctx.seriesWinner === null) {
            const nextMatchIdx = ctx.seriesMatchIndex + 1;
            const nextRules = buildRuleSetForLevel(ctx.levelId, {
              seriesMatchIndex: nextMatchIdx,
              targetScore: ctx.config.targetScore,
              teamHandsCount: ctx.teamHandsCount,
            });
            return startRound(
              {
                ...ctx,
                seriesMatchIndex: nextMatchIdx,
                ruleSet: nextRules,
              },
              {
                resetMatch: true,
                resetSeries: false,
                ...(event.computerHideSetup !== undefined
                  ? { computerHideSetup: event.computerHideSetup }
                  : {}),
              }
            );
          }

          return startRound(ctx, {
            resetMatch: true,
            resetSeries: true,
            ...(event.computerHideSetup !== undefined
              ? { computerHideSetup: event.computerHideSetup }
              : {}),
          });
        }

        return startRound(
          { ...ctx, roundNumber: ctx.roundNumber + 1 },
          {
            resetMatch: false,
            resetSeries: false,
            ...(event.computerHideSetup !== undefined
              ? { computerHideSetup: event.computerHideSetup }
              : {}),
          }
        );
      }
      if (event.type === 'RESET_MATCH') {
        const resetCtx = applyLevelConfiguration(ctx, ctx.levelId, ctx.teamHandsCount);
        return startRound(resetCtx, {
          resetMatch: true,
          resetSeries: true,
          ...(event.initialHider !== undefined ? { initialHider: event.initialHider } : {}),
          ...(event.computerHideSetup !== undefined
            ? { computerHideSetup: event.computerHideSetup }
            : {}),
        });
      }
      if (event.type === 'EXIT_TO_MENU') {
        return { ...ctx, state: 'MENU' };
      }
      return ctx;
    }

    case 'ROUND_VOID': {
      if (event.type === 'NEXT_ROUND') {
        return startRound(ctx, {
          resetMatch: false,
          resetSeries: false,
          ...(event.computerHideSetup !== undefined
            ? { computerHideSetup: event.computerHideSetup }
            : {}),
        });
      }
      if (event.type === 'RESET_MATCH') {
        return startRound(ctx, {
          resetMatch: true,
          resetSeries: true,
          ...(event.computerHideSetup !== undefined
            ? { computerHideSetup: event.computerHideSetup }
            : {}),
        });
      }
      if (event.type === 'EXIT_TO_MENU') {
        return { ...ctx, state: 'MENU' };
      }
      return ctx;
    }

    case 'PAUSED': {
      if (event.type === 'EXIT_TO_MENU') {
        return { ...ctx, state: 'MENU', pausedFrom: null, phaseRemainingMs: 0 };
      }
      return ctx;
    }

    case 'CAMERA_ERROR': {
      if (event.type === 'CAMERA_RECOVERED') {
        const target = ctx.cameraErrorReturnState ?? 'MENU';
        return {
          ...ctx,
          state: target,
          cameraErrorReturnState: null,
          cameraErrorMessageFa: null,
        };
      }
      if (event.type === 'RETRY_PERMISSION') {
        return {
          ...ctx,
          state: 'PERMISSION',
          cameraErrorReturnState: null,
          cameraErrorMessageFa: null,
        };
      }
      return ctx;
    }
  }
}
