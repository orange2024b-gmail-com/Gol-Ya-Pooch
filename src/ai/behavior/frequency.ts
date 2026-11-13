import { clampProb, type BehaviorExpert, type RoundBehaviorRecord } from './types';

/**
 * Predicts P(LEFT) using Laplace-smoothed overall choice frequency.
 */
export class OverallFrequencyExpert implements BehaviorExpert {
  readonly id = 'freq-overall';
  readonly description = 'Overall hand frequency (Laplace smoothed)';
  private readonly priorStrength: number;

  constructor(priorStrength = 2) {
    this.priorStrength = priorStrength;
  }

  predictLeft(history: readonly RoundBehaviorRecord[]): number {
    if (history.length === 0) return 0.5;
    let leftCount = 0;
    for (const r of history) {
      if (r.chosenHand === 'LEFT') leftCount += 1;
    }
    const alpha = this.priorStrength * 0.5;
    return clampProb((leftCount + alpha) / (history.length + this.priorStrength));
  }
}

/**
 * Predicts P(LEFT) over a recent sliding window of rounds.
 */
export class RecentFrequencyExpert implements BehaviorExpert {
  readonly id: string;
  readonly description: string;
  private readonly windowSize: number;
  private readonly priorStrength: number;

  constructor(windowSize = 6, priorStrength = 2) {
    this.windowSize = Math.max(2, windowSize);
    this.priorStrength = priorStrength;
    this.id = `freq-recent-${this.windowSize}`;
    this.description = `Recent-${this.windowSize} hand frequency`;
  }

  predictLeft(history: readonly RoundBehaviorRecord[]): number {
    if (history.length === 0) return 0.5;
    const recent = history.slice(-this.windowSize);
    let leftCount = 0;
    for (const r of recent) {
      if (r.chosenHand === 'LEFT') leftCount += 1;
    }
    const alpha = this.priorStrength * 0.5;
    return clampProb((leftCount + alpha) / (recent.length + this.priorStrength));
  }
}
