import type { HandSide } from '../../vision/types';

export interface RoundBehaviorRecord {
  roundIndex: number;
  timestampMs: number;
  chosenHand: HandSide;
  computerGuess: HandSide;
  humanWon: boolean;
}

export interface BehaviorExpert {
  readonly id: string;
  readonly description: string;
  /** Predicts P(next chosen hand === 'LEFT') in [0.02, 0.98]. */
  predictLeft(history: readonly RoundBehaviorRecord[]): number;
}

export function clampProb(p: number): number {
  return Math.max(0.02, Math.min(0.98, p));
}
