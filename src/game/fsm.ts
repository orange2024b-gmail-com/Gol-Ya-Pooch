import type { HandSide, RevealOutcome } from '../vision/types';

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
  showDurationMs: 12_000,
  hideDurationMs: 10_000,
  guessDisplayMs: 2_500,
  revealDurationMs: 10_000,
  maxHandLostMs: 500,
  targetScore: 7,
};

export interface GameContext {
  state: GameStateName;
  calibrated: boolean;
  roundNumber: number;
  validRoundsCompleted: number;
  humanScore: number;
  computerScore: number;
  phaseRemainingMs: number;
  pinchConfirmed: boolean;
  initialShownHand: HandSide | null;
  computerGuess: HandSide | null;
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
  | { type: 'START_GAME' }
  | { type: 'PINCH_CONFIRMED'; side: HandSide }
  | { type: 'FINISH_HIDING' }
  | { type: 'PROCEED_TO_REVEAL' }
  | { type: 'REVEAL_VERIFIED'; outcome: RevealOutcome }
  | { type: 'NEXT_ROUND' }
  | { type: 'RESET_MATCH' }
  | { type: 'EXIT_TO_MENU' }
  | { type: 'TICK'; dtMs: number; maxLostDurationMs?: number; randomBit?: 0 | 1 }
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

export function createInitialGameContext(config: Partial<FsmConfig> = {}): GameContext {
  return {
    state: 'SPLASH',
    calibrated: false,
    roundNumber: 1,
    validRoundsCompleted: 0,
    humanScore: 0,
    computerScore: 0,
    phaseRemainingMs: 0,
    pinchConfirmed: false,
    initialShownHand: null,
    computerGuess: null,
    revealedHand: null,
    lastRoundWinner: null,
    voidReason: null,
    voidMessageFa: null,
    pausedFrom: null,
    cameraErrorReturnState: null,
    cameraErrorMessageFa: null,
    config: { ...DEFAULT_FSM_CONFIG, ...config },
  };
}

function makePlaceholderGuess(randomBit?: 0 | 1): HandSide {
  if (randomBit === 0) return 'LEFT';
  if (randomBit === 1) return 'RIGHT';
  if (typeof globalThis.crypto !== 'undefined' && typeof globalThis.crypto.getRandomValues === 'function') {
    const arr = new Uint8Array(1);
    globalThis.crypto.getRandomValues(arr);
    return (arr[0]! & 1) === 0 ? 'LEFT' : 'RIGHT';
  }
  return Math.random() < 0.5 ? 'LEFT' : 'RIGHT';
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
    roundNumber: resetScores ? 1 : ctx.roundNumber,
    validRoundsCompleted: resetScores ? 0 : ctx.validRoundsCompleted,
    humanScore: resetScores ? 0 : ctx.humanScore,
    computerScore: resetScores ? 0 : ctx.computerScore,
    phaseRemainingMs: ctx.config.showDurationMs,
    pinchConfirmed: false,
    initialShownHand: null,
    computerGuess: null,
    revealedHand: null,
    lastRoundWinner: null,
    voidReason: null,
    voidMessageFa: null,
  };
}

export function transitionGameFsm(ctx: GameContext, event: GameEvent): GameContext {
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
        return {
          ...ctx,
          state: 'COMPUTER_GUESS',
          computerGuess: makePlaceholderGuess(),
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
          return {
            ...ctx,
            state: 'COMPUTER_GUESS',
            computerGuess: makePlaceholderGuess(event.randomBit),
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
        return {
          ...ctx,
          state: 'ROUND_RESULT',
          phaseRemainingMs: 0,
          revealedHand: event.outcome,
          validRoundsCompleted: ctx.validRoundsCompleted + 1,
          computerScore: ctx.computerScore + (computerWon ? 1 : 0),
          humanScore: ctx.humanScore + (computerWon ? 0 : 1),
          lastRoundWinner: computerWon ? 'COMPUTER' : 'HUMAN',
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
