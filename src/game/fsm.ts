import type { HandSide, RevealOutcome } from '../vision/types';
import {
  CLASSIC_1V1_RULES,
  loadRuleSet,
  resolveRoundByRules,
  type PlayerRole,
  type RuleSetDefinition,
} from './rules/engine';

export const ALL_FSM_STATES = [
  'SPLASH',
  'PERMISSION',
  'CALIBRATION',
  'MENU',
  'SHOW_GOL',
  'HIDING',
  'COMPUTER_GUESS',
  'REVEAL_VERIFY',
  'ROUND_RESULT',
  'ROUND_VOID',
  'PAUSED',
  'CAMERA_ERROR',
] as const;

export type GameStateName = (typeof ALL_FSM_STATES)[number];

export type VoidReason =
  | 'HANDS_LOST'
  | 'SHOW_TIMEOUT'
  | 'REVEAL_NEITHER'
  | 'REVEAL_BOTH'
  | 'REVEAL_TIMEOUT'
  | 'CAMERA_INTERRUPTED';

export interface FsmConfig {
  showDurationMs: number;
  hideDurationMs: number;
  guessDisplayMs: number;
  revealDurationMs: number;
  maxHandLostMs: number;
  targetScore: number;
}

export const DEFAULT_FSM_CONFIG: FsmConfig = {
  showDurationMs: CLASSIC_1V1_RULES.timeoutsMs.showGol,
  hideDurationMs: CLASSIC_1V1_RULES.timeoutsMs.hiding,
  guessDisplayMs: CLASSIC_1V1_RULES.timeoutsMs.guessDisplay,
  revealDurationMs: CLASSIC_1V1_RULES.timeoutsMs.revealVerify,
  maxHandLostMs: CLASSIC_1V1_RULES.voidConditions.maxHandLostMs,
  targetScore: CLASSIC_1V1_RULES.targetScore,
};

export interface AiGuessPayload {
  guess: HandSide;
  probability: number;
  explanationFa: string;
  timelineEvents?: string[];
}

export interface GameContext {
  state: GameStateName;
  calibrated: boolean;
  ruleSet: RuleSetDefinition;
  ruleHiderOwner: PlayerRole;
  ruleGuesserOwner: PlayerRole;
  roundNumber: number;
  validRoundsCompleted: number;
  humanScore: number;
  computerScore: number;
  matchWinner: PlayerRole | null;
  phaseRemainingMs: number;
  pinchConfirmed: boolean;
  initialShownHand: HandSide | null;
  computerGuess: HandSide | null;
  computerGuessProb: number | null;
  explanationFa: string | null;
  timelineEvents: string[];
  revealedHand: HandSide | null;
  lastRoundWinner: 'HUMAN' | 'COMPUTER' | null;
  voidReason: VoidReason | null;
  voidMessageFa: string | null;
  pausedFrom: Exclude<GameStateName, 'PAUSED' | 'CAMERA_ERROR'> | null;
  cameraErrorReturnState: Exclude<GameStateName, 'CAMERA_ERROR' | 'PAUSED'> | null;
  cameraErrorMessageFa: string | null;
  config: FsmConfig;
}

export type GameEvent =
  | { type: 'GO_PERMISSION' }
  | { type: 'CAMERA_READY' }
  | { type: 'CALIBRATION_COMPLETE' }
  | { type: 'OPEN_CALIBRATION' }
  | { type: 'SET_TARGET_SCORE'; targetScore: number }
  | { type: 'START_GAME' }
  | { type: 'PINCH_CONFIRMED'; side: HandSide }
  | {
      type: 'FINISH_HIDING';
      aiGuess?: AiGuessPayload;
    }
  | { type: 'PROCEED_TO_REVEAL' }
  | { type: 'REVEAL_VERIFIED'; outcome: RevealOutcome }
  | { type: 'NEXT_ROUND' }
  | { type: 'RESET_MATCH' }
  | { type: 'EXIT_TO_MENU' }
  | {
      type: 'TICK';
      dtMs: number;
      maxLostDurationMs?: number;
      randomBit?: 0 | 1;
      aiGuess?: AiGuessPayload;
    }
  | { type: 'TAB_HIDDEN' }
  | { type: 'TAB_VISIBLE' }
  | { type: 'CAMERA_LOST'; messageFa?: string }
  | { type: 'CAMERA_RECOVERED' }
  | { type: 'RETRY_PERMISSION' };

export const VOID_MESSAGES_FA: Record<VoidReason, string> = {
  HANDS_LOST: 'دست‌ها بیش از نیم ثانیه از کادر خارج شدند؛ دور باطل شد.',
  SHOW_TIMEOUT: 'زمان نمایش گل به پایان رسید؛ دور باطل شد.',
  REVEAL_NEITHER: 'گل در هیچ‌کدام از دست‌ها دیده نشد؛ دور باطل شد.',
  REVEAL_BOTH: 'گل در هر دو دست تشخیص داده شد؛ دور باطل شد.',
  REVEAL_TIMEOUT: 'زمان باز کردن هر دو دست به پایان رسید؛ دور باطل شد.',
  CAMERA_INTERRUPTED: 'ارتباط دوربین قطع و دوباره وصل شد؛ دور فعلی باطل و تکرار می‌شود.',
};

export function createInitialGameContext(
  config: Partial<FsmConfig> = {},
  ruleSet: RuleSetDefinition = CLASSIC_1V1_RULES
): GameContext {
  const mergedConfig: FsmConfig = {
    showDurationMs: ruleSet.timeoutsMs.showGol,
    hideDurationMs: ruleSet.timeoutsMs.hiding,
    guessDisplayMs: ruleSet.timeoutsMs.guessDisplay,
    revealDurationMs: ruleSet.timeoutsMs.revealVerify,
    maxHandLostMs: ruleSet.voidConditions.maxHandLostMs,
    targetScore: ruleSet.targetScore,
    ...config,
  };
  const effectiveRules = loadRuleSet(ruleSet, { targetScore: mergedConfig.targetScore });

  return {
    state: 'SPLASH',
    calibrated: false,
    ruleSet: effectiveRules,
    ruleHiderOwner: 'HUMAN',
    ruleGuesserOwner: 'COMPUTER',
    roundNumber: 1,
    validRoundsCompleted: 0,
    humanScore: 0,
    computerScore: 0,
    matchWinner: null,
    phaseRemainingMs: 0,
    pinchConfirmed: false,
    initialShownHand: null,
    computerGuess: null,
    computerGuessProb: null,
    explanationFa: null,
    timelineEvents: [],
    revealedHand: null,
    lastRoundWinner: null,
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

function startRound(ctx: GameContext, resetScores: boolean): GameContext {
  return {
    ...ctx,
    state: 'SHOW_GOL',
    ruleHiderOwner: resetScores ? 'HUMAN' : ctx.ruleHiderOwner,
    ruleGuesserOwner: resetScores ? 'COMPUTER' : ctx.ruleGuesserOwner,
    roundNumber: resetScores ? 1 : ctx.roundNumber,
    validRoundsCompleted: resetScores ? 0 : ctx.validRoundsCompleted,
    humanScore: resetScores ? 0 : ctx.humanScore,
    computerScore: resetScores ? 0 : ctx.computerScore,
    matchWinner: resetScores ? null : ctx.matchWinner,
    phaseRemainingMs: ctx.config.showDurationMs,
    pinchConfirmed: false,
    initialShownHand: null,
    computerGuess: null,
    computerGuessProb: null,
    explanationFa: null,
    timelineEvents: [],
    revealedHand: null,
    lastRoundWinner: null,
    voidReason: null,
    voidMessageFa: null,
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
    if (ctx.state === 'SPLASH' || ctx.state === 'CAMERA_ERROR') {
      return ctx;
    }
    const baseState = ctx.state === 'PAUSED' ? (ctx.pausedFrom ?? 'MENU') : ctx.state;
    const isActiveRoundPhase =
      baseState === 'SHOW_GOL' ||
      baseState === 'HIDING' ||
      baseState === 'COMPUTER_GUESS' ||
      baseState === 'REVEAL_VERIFY';

    return {
      ...ctx,
      state: 'CAMERA_ERROR',
      pausedFrom: null,
      cameraErrorReturnState: isActiveRoundPhase ? 'ROUND_VOID' : baseState,
      voidReason: isActiveRoundPhase ? 'CAMERA_INTERRUPTED' : ctx.voidReason,
      voidMessageFa: isActiveRoundPhase ? VOID_MESSAGES_FA.CAMERA_INTERRUPTED : ctx.voidMessageFa,
      cameraErrorMessageFa:
        event.messageFa ?? 'دسترسی به دوربین قطع شد. لطفاً دوربین را بررسی و دوباره تلاش کنید.',
    };
  }

  switch (ctx.state) {
    case 'SPLASH': {
      if (event.type === 'GO_PERMISSION') {
        return { ...ctx, state: 'PERMISSION' };
      }
      return ctx;
    }

    case 'PERMISSION': {
      if (event.type === 'CAMERA_READY') {
        return { ...ctx, state: ctx.calibrated ? 'MENU' : 'CALIBRATION' };
      }
      return ctx;
    }

    case 'CALIBRATION': {
      if (event.type === 'CALIBRATION_COMPLETE') {
        return { ...ctx, calibrated: true, state: 'MENU' };
      }
      if (event.type === 'EXIT_TO_MENU' && ctx.calibrated) {
        return { ...ctx, state: 'MENU' };
      }
      return ctx;
    }

    case 'MENU': {
      if (event.type === 'OPEN_CALIBRATION') {
        return { ...ctx, state: 'CALIBRATION' };
      }
      if (event.type === 'START_GAME') {
        if (!ctx.calibrated) {
          return { ...ctx, state: 'CALIBRATION' };
        }
        return startRound(ctx, true);
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
        const payload = event.aiGuess ?? makeFallbackGuess();
        return {
          ...ctx,
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
        const remaining = Math.max(0, ctx.phaseRemainingMs - Math.max(0, event.dtMs));
        if (remaining <= 0) {
          const payload = event.aiGuess ?? makeFallbackGuess(event.randomBit);
          return {
            ...ctx,
            state: 'COMPUTER_GUESS',
            computerGuess: payload.guess,
            computerGuessProb: payload.probability,
            explanationFa: payload.explanationFa,
            timelineEvents: payload.timelineEvents ?? [],
            phaseRemainingMs: ctx.config.guessDisplayMs,
          };
        }
        return { ...ctx, phaseRemainingMs: remaining };
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
        const resolution = resolveRoundByRules(ctx.ruleSet, {
          hider: 'HUMAN',
          guesser: 'COMPUTER',
          guessOutcome: computerWon ? 'CORRECT_DIRECT' : 'WRONG',
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

        return {
          ...ctx,
          state: 'ROUND_RESULT',
          phaseRemainingMs: 0,
          revealedHand: event.outcome,
          validRoundsCompleted: ctx.validRoundsCompleted + 1,
          humanScore: nextHumanScore,
          computerScore: nextComputerScore,
          ruleHiderOwner: resolution.nextHider,
          ruleGuesserOwner: resolution.nextGuesser,
          matchWinner,
          lastRoundWinner: resolution.winner,
        };
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

    case 'ROUND_RESULT': {
      if (event.type === 'NEXT_ROUND') {
        if (ctx.matchWinner !== null) {
          return startRound(ctx, true);
        }
        return startRound({ ...ctx, roundNumber: ctx.roundNumber + 1 }, false);
      }
      if (event.type === 'RESET_MATCH') {
        return startRound(ctx, true);
      }
      if (event.type === 'EXIT_TO_MENU') {
        return { ...ctx, state: 'MENU' };
      }
      return ctx;
    }

    case 'ROUND_VOID': {
      if (event.type === 'NEXT_ROUND') {
        return startRound(ctx, false);
      }
      if (event.type === 'RESET_MATCH') {
        return startRound(ctx, true);
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
