import { clampProb, type BehaviorExpert, type RoundBehaviorRecord } from './types';

/**
 * First-order Markov expert conditioned on the previous round's win/loss outcome
 * (win-stay vs. lose-shift probabilities).
 */
export class MarkovWinStayLoseShiftExpert implements BehaviorExpert {
  readonly id = 'markov-wsls';
  readonly description = 'Win-stay / lose-shift Markov predictor';
  private readonly priorStrength: number;

  constructor(priorStrength = 2) {
    this.priorStrength = priorStrength;
  }

  predictLeft(history: readonly RoundBehaviorRecord[]): number {
    if (history.length === 0) return 0.5;

    let winTotal = 0;
    let winStay = 0;
    let lossTotal = 0;
    let lossStay = 0;

    for (let i = 1; i < history.length; i++) {
      const prev = history[i - 1]!;
      const curr = history[i]!;
      const stayed = curr.chosenHand === prev.chosenHand;
      if (prev.humanWon) {
        winTotal += 1;
        if (stayed) winStay += 1;
      } else {
        lossTotal += 1;
        if (stayed) lossStay += 1;
      }
    }

    const last = history[history.length - 1]!;
    const alpha = this.priorStrength * 0.5;
    const stayProb = last.humanWon
      ? (winStay + alpha) / (winTotal + this.priorStrength)
      : (lossStay + alpha) / (lossTotal + this.priorStrength);

    return clampProb(last.chosenHand === 'LEFT' ? stayProb : 1 - stayProb);
  }
}
