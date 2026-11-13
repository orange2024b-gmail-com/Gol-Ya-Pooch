import type { HandSide } from '../vision/types';
import {
  clampProb,
  createDefaultBehaviorExperts,
  type BehaviorExpert,
  type RoundBehaviorRecord,
} from './behavior';

export interface HedgeEnsembleConfig {
  /** Learning rate eta for multiplicative weights update: w_i <- w_i * exp(-eta * loss_i). */
  learningRate: number;
  /** Minimum normalized weight floor so no expert is permanently zeroed out. */
  minWeightFloor: number;
}

export const DEFAULT_HEDGE_CONFIG: HedgeEnsembleConfig = {
  learningRate: 0.45,
  minWeightFloor: 0.01,
};

export interface ExpertPredictionSummary {
  id: string;
  weight: number;
  pLeft: number;
}

export interface EnsemblePrediction {
  pLeft: number;
  pRight: number;
  confidence: number;
  experts: ExpertPredictionSummary[];
}

export interface SerializedEnsembleState {
  weights: Record<string, number>;
  cumulativeLosses: Record<string, number>;
  roundsUpdated: number;
}

/**
 * Multiplicative-weights (Hedge) ensemble mixing behavioral experts over
 * player hiding history, updated after every verified round.
 */
export class HedgeEnsemble {
  private readonly experts: BehaviorExpert[];
  private readonly config: HedgeEnsembleConfig;
  private weights: number[];
  private cumulativeLosses: number[];
  private roundsUpdated = 0;

  constructor(
    experts: BehaviorExpert[] = createDefaultBehaviorExperts(),
    config: Partial<HedgeEnsembleConfig> = {}
  ) {
    if (experts.length === 0) {
      throw new Error('HedgeEnsemble requires at least one expert');
    }
    this.experts = experts;
    this.config = { ...DEFAULT_HEDGE_CONFIG, ...config };
    this.weights = new Array<number>(experts.length).fill(1 / experts.length);
    this.cumulativeLosses = new Array<number>(experts.length).fill(0);
  }

  private normalizeWeights(): void {
    const floor = Math.min(this.config.minWeightFloor, 0.5 / this.weights.length);
    let sum = 0;
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = Math.max(floor, this.weights[i] ?? floor);
      sum += this.weights[i]!;
    }
    if (sum <= 0) {
      const uniform = 1 / this.weights.length;
      this.weights.fill(uniform);
      return;
    }
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = this.weights[i]! / sum;
    }
  }

  predict(history: readonly RoundBehaviorRecord[]): EnsemblePrediction {
    const expertsSummary: ExpertPredictionSummary[] = [];
    let pLeftSum = 0;

    for (let i = 0; i < this.experts.length; i++) {
      const expert = this.experts[i]!;
      const w = this.weights[i]!;
      const p = expert.predictLeft(history);
      pLeftSum += w * p;
      expertsSummary.push({ id: expert.id, weight: w, pLeft: p });
    }

    const pLeft = clampProb(pLeftSum);
    return {
      pLeft,
      pRight: 1 - pLeft,
      confidence: Math.abs(pLeft - (1 - pLeft)),
      experts: expertsSummary,
    };
  }

  /**
   * Updates expert weights using squared (Brier) loss in [0, 1] against the verified outcome.
   * Should be called with the history *prior* to the current round.
   */
  update(historyBeforeRound: readonly RoundBehaviorRecord[], actualHand: HandSide): void {
    const target = actualHand === 'LEFT' ? 1 : 0;

    for (let i = 0; i < this.experts.length; i++) {
      const pred = this.experts[i]!.predictLeft(historyBeforeRound);
      const loss = (pred - target) * (pred - target); // bounded in [0, 1]
      this.cumulativeLosses[i] = (this.cumulativeLosses[i] ?? 0) + loss;
      this.weights[i] = this.weights[i]! * Math.exp(-this.config.learningRate * loss);
    }

    this.roundsUpdated += 1;
    this.normalizeWeights();
  }

  getWeights(): Record<string, number> {
    const out: Record<string, number> = {};
    for (let i = 0; i < this.experts.length; i++) {
      out[this.experts[i]!.id] = this.weights[i]!;
    }
    return out;
  }

  serialize(): SerializedEnsembleState {
    const weights: Record<string, number> = {};
    const cumulativeLosses: Record<string, number> = {};
    for (let i = 0; i < this.experts.length; i++) {
      const id = this.experts[i]!.id;
      weights[id] = this.weights[i]!;
      cumulativeLosses[id] = this.cumulativeLosses[i]!;
    }
    return {
      weights,
      cumulativeLosses,
      roundsUpdated: this.roundsUpdated,
    };
  }

  restore(state: SerializedEnsembleState): void {
    for (let i = 0; i < this.experts.length; i++) {
      const id = this.experts[i]!.id;
      const w = state.weights[id];
      const l = state.cumulativeLosses[id];
      this.weights[i] = typeof w === 'number' && Number.isFinite(w) && w > 0 ? w : 1 / this.experts.length;
      this.cumulativeLosses[i] = typeof l === 'number' && Number.isFinite(l) && l >= 0 ? l : 0;
    }
    this.roundsUpdated = state.roundsUpdated;
    this.normalizeWeights();
  }
}
