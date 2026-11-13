import { clampProb, type BehaviorExpert, type RoundBehaviorRecord } from './types';

/**
 * Detects whether the player repeats the same hand immediately after fooling the computer
 * (humanWon === true). Neutral (0.5) when the previous round was a loss.
 */
export class RepeatAfterSuccessExpert implements BehaviorExpert {
  readonly id = 'repeat-after-success';
  readonly description = 'Repeat-after-success detector';
  private readonly priorStrength: number;

  constructor(priorStrength = 1.5) {
    this.priorStrength = priorStrength;
  }

  predictLeft(history: readonly RoundBehaviorRecord[]): number {
    if (history.length === 0) return 0.5;
    const last = history[history.length - 1]!;
    if (!last.humanWon) return 0.5;

    let totalSuccesses = 0;
    let repeats = 0;
    for (let i = 1; i < history.length; i++) {
      const prev = history[i - 1]!;
      const curr = history[i]!;
      if (prev.humanWon) {
        totalSuccesses += 1;
        if (curr.chosenHand === prev.chosenHand) repeats += 1;
      }
    }

    if (totalSuccesses === 0) {
      // Mild baseline win-stay prior before observations accumulate
      return clampProb(last.chosenHand === 'LEFT' ? 0.58 : 0.42);
    }

    const alpha = this.priorStrength * 0.5;
    const repeatProb = (repeats + alpha) / (totalSuccesses + this.priorStrength);
    return clampProb(last.chosenHand === 'LEFT' ? repeatProb : 1 - repeatProb);
  }
}

/**
 * Detects whether the player switches hands immediately after being caught by the computer
 * (humanWon === false). Neutral (0.5) when the previous round was a win.
 */
export class SwitchAfterCaughtExpert implements BehaviorExpert {
  readonly id = 'switch-after-caught';
  readonly description = 'Switch-after-being-caught detector';
  private readonly priorStrength: number;

  constructor(priorStrength = 1.5) {
    this.priorStrength = priorStrength;
  }

  predictLeft(history: readonly RoundBehaviorRecord[]): number {
    if (history.length === 0) return 0.5;
    const last = history[history.length - 1]!;
    if (last.humanWon) return 0.5;

    let totalCaught = 0;
    let switches = 0;
    for (let i = 1; i < history.length; i++) {
      const prev = history[i - 1]!;
      const curr = history[i]!;
      if (!prev.humanWon) {
        totalCaught += 1;
        if (curr.chosenHand !== prev.chosenHand) switches += 1;
      }
    }

    if (totalCaught === 0) {
      // Mild baseline lose-shift prior before observations accumulate
      return clampProb(last.chosenHand === 'LEFT' ? 0.38 : 0.62);
    }

    const alpha = this.priorStrength * 0.5;
    const switchProb = (switches + alpha) / (totalCaught + this.priorStrength);
    // If last was LEFT and they switch, next is RIGHT -> P(LEFT) = 1 - switchProb
    return clampProb(last.chosenHand === 'LEFT' ? 1 - switchProb : switchProb);
  }
}
