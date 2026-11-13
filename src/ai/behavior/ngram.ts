import { clampProb, type BehaviorExpert, type RoundBehaviorRecord } from './types';

/**
 * n-gram predictor (n = 1..3) conditioned on the sequence of the player's last n final hands.
 */
export class NGramExpert implements BehaviorExpert {
  readonly id: string;
  readonly description: string;
  private readonly n: 1 | 2 | 3;
  private readonly priorStrength: number;

  constructor(n: 1 | 2 | 3, priorStrength = 1.5) {
    this.n = n;
    this.priorStrength = priorStrength;
    this.id = `ngram-${n}`;
    this.description = `${n}-gram hand transition predictor`;
  }

  predictLeft(history: readonly RoundBehaviorRecord[]): number {
    if (history.length <= this.n) return 0.5;

    const contextKey = history
      .slice(-this.n)
      .map((r) => r.chosenHand[0])
      .join('');

    let matches = 0;
    let leftAfter = 0;

    for (let i = this.n; i < history.length; i++) {
      let key = '';
      for (let k = this.n; k >= 1; k--) {
        key += history[i - k]!.chosenHand[0];
      }
      if (key === contextKey) {
        matches += 1;
        if (history[i]!.chosenHand === 'LEFT') {
          leftAfter += 1;
        }
      }
    }

    if (matches === 0) return 0.5;
    const alpha = this.priorStrength * 0.5;
    return clampProb((leftAfter + alpha) / (matches + this.priorStrength));
  }
}
