import type { FrameFeatures } from '../vision/features';
import type { HandSide } from '../vision/types';
import { EventDetector, type TimelineSummary, type TrackingEvent } from './eventDetector';

export interface BeliefFilterParams {
  /** Initial belief assigned to the hand confirmed in SHOW_GOL. */
  initCertainty: number;
  /** Per-second transfer diffusion rate when hands are clearly apart (~0). */
  apartTransferRatePerSec: number;
  /** Base per-second transfer diffusion rate when hands are in contact. */
  contactBaseRatePerSec: number;
  /** Additional diffusion rate scaling with contact duration (sec). */
  contactDurationSlopePerSec: number;
  /** Maximum symmetric contact diffusion probability per step. */
  maxContactDiffusionPerStep: number;
  /** Asymmetric handover boost when a hand partially opens during contact. */
  partialOpeningHandoverProb: number;
  /** Strong log-likelihood scale for gol sightings. */
  sightingLogOddsWeight: number;
  /** Weak log-likelihood weight for fist compactness asymmetry. */
  compactnessLogOddsWeight: number;
  /** Weak log-likelihood weight for finger curl asymmetry. */
  curlLogOddsWeight: number;
  /** Weak log-likelihood weight for compactness delta (squeezing) asymmetry. */
  compactnessDeltaLogOddsWeight: number;
  /** Attenuation factor for weak posture cues before any contact has occurred. */
  preContactWeakCueAttenuation: number;
  /** Minimum observation likelihood floor to avoid probability collapse. */
  minLikelihood: number;
}

export const DEFAULT_BELIEF_PARAMS: BeliefFilterParams = {
  initCertainty: 0.99,
  apartTransferRatePerSec: 0.002,
  contactBaseRatePerSec: 0.14,
  contactDurationSlopePerSec: 0.45,
  maxContactDiffusionPerStep: 0.22,
  partialOpeningHandoverProb: 0.66,
  sightingLogOddsWeight: 4.5,
  compactnessLogOddsWeight: 1.8,
  curlLogOddsWeight: 1.1,
  compactnessDeltaLogOddsWeight: 1.8,
  preContactWeakCueAttenuation: 0.05,
  minLikelihood: 0.05,
};

export interface BeliefState {
  timestampMs: number;
  pLeft: number;
  pRight: number;
  /** Confidence in [0, 1], computed as |P(LEFT) - P(RIGHT)|. */
  confidence: number;
  mostLikely: HandSide;
  inContact: boolean;
  lowQuality: boolean;
}

export interface BeliefFilterSnapshot extends BeliefState {
  initialSide: HandSide;
  hasHadContact: boolean;
  timeline: TrackingEvent[];
  summary: TimelineSummary;
  trace: BeliefState[];
}

function clamp(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function sigmoid(x: number): number {
  if (x >= 20) return 1 - 1e-9;
  if (x <= -20) return 1e-9;
  return 1 / (1 + Math.exp(-x));
}

/**
 * Discrete 2-state Bayesian HMM forward filter over {LEFT, RIGHT}
 * tracking gol location across hiding-phase frames.
 */
export class BeliefFilter {
  private readonly params: BeliefFilterParams;
  private readonly eventDetector = new EventDetector();

  private initialSide: HandSide = 'LEFT';
  private pLeft = 0.5;
  private pRight = 0.5;
  private lastTimestampMs: number | null = null;
  private hasHadContact = false;
  private trace: BeliefState[] = [];
  private lowQualityFrames = 0;

  constructor(params: Partial<BeliefFilterParams> = {}) {
    this.params = { ...DEFAULT_BELIEF_PARAMS, ...params };
  }

  init(initialSide: HandSide, startTimestampMs?: number): BeliefState {
    this.initialSide = initialSide;
    const cert = clamp(this.params.initCertainty, 0.5, 0.999);
    this.pLeft = initialSide === 'LEFT' ? cert : 1 - cert;
    this.pRight = 1 - this.pLeft;
    this.lastTimestampMs = startTimestampMs ?? null;
    this.hasHadContact = false;
    this.lowQualityFrames = 0;
    this.eventDetector.reset(startTimestampMs);

    const state: BeliefState = {
      timestampMs: startTimestampMs ?? 0,
      pLeft: this.pLeft,
      pRight: this.pRight,
      confidence: Math.abs(this.pLeft - this.pRight),
      mostLikely: this.pLeft >= this.pRight ? 'LEFT' : 'RIGHT',
      inContact: false,
      lowQuality: false,
    };
    this.trace = [state];
    return state;
  }

  /**
   * Computes the 2x2 HMM transition probabilities:
   *   tLtoR = P(state_t = RIGHT | state_{t-1} = LEFT)
   *   tRtoL = P(state_t = LEFT  | state_{t-1} = RIGHT)
   */
  private computeTransitionProbs(features: FrameFeatures, dtSec: number): {
    tLtoR: number;
    tRtoL: number;
  } {
    if (!features.interHand.inContact) {
      const pApart = clamp(this.params.apartTransferRatePerSec * dtSec, 0, 0.02);
      return { tLtoR: pApart, tRtoL: pApart };
    }

    const contactDurSec = features.interHand.contactDurationMs / 1000;
    const rate =
      this.params.contactBaseRatePerSec +
      this.params.contactDurationSlopePerSec * Math.min(2.5, contactDurSec);
    const baseDiff = clamp(rate * dtSec, 0, this.params.maxContactDiffusionPerStep);

    let tLtoR = baseDiff;
    let tRtoL = baseDiff;

    // Partial opening of hand X during contact boosts transfer away from X
    // (giver opens slightly) and slightly boosts transfer into the tighter hand.
    if (features.left.partialOpening) {
      const boost =
        this.params.partialOpeningHandoverProb *
        Math.max(0.35, features.left.partialOpeningStrength);
      tLtoR = clamp(tLtoR + boost, 0, 0.72);
    }
    if (features.right.partialOpening) {
      const boost =
        this.params.partialOpeningHandoverProb *
        Math.max(0.35, features.right.partialOpeningStrength);
      tRtoL = clamp(tRtoL + boost, 0, 0.72);
    }

    return { tLtoR, tRtoL };
  }

  /**
   * Computes observation likelihoods P(obs_t | state = LEFT) and P(obs_t | state = RIGHT).
   * Occluded or low-quality frames produce flattened [0.5, 0.5] likelihoods.
   */
  private computeObservationLikelihoods(features: FrameFeatures): {
    likLeft: number;
    likRight: number;
  } {
    const sightingDiff = features.left.golSightingScore - features.right.golSightingScore;
    const hasSighting = Math.abs(sightingDiff) >= 0.2;

    // Occluded or low-quality frames get flattened likelihoods unless a clear sighting occurred
    if (features.quality.lowQuality && !hasSighting) {
      return { likLeft: 0.5, likRight: 0.5 };
    }

    let logOdds = 0;

    // Strong observation: gol sighting in a hand ROI
    logOdds += this.params.sightingLogOddsWeight * sightingDiff;

    // Weak observations: fist compactness, finger curl, and squeeze delta asymmetries
    if (!features.quality.anyOccluded && features.interHand.bothDetected) {
      const weakScale = this.hasHadContact ? 1.0 : this.params.preContactWeakCueAttenuation;
      const compDiff = features.left.fistCompactness - features.right.fistCompactness;
      const curlDiff = features.left.totalCurl - features.right.totalCurl;
      const deltaDiff = features.left.compactnessDelta - features.right.compactnessDelta;

      logOdds +=
        weakScale *
        (this.params.compactnessLogOddsWeight * compDiff +
          this.params.curlLogOddsWeight * curlDiff +
          this.params.compactnessDeltaLogOddsWeight * deltaDiff);
    }

    const pLikLeft = clamp(
      sigmoid(logOdds),
      this.params.minLikelihood,
      1 - this.params.minLikelihood
    );
    return {
      likLeft: pLikLeft,
      likRight: 1 - pLikLeft,
    };
  }

  update(features: FrameFeatures): BeliefState {
    const ts = features.timestampMs;
    const dtSec =
      this.lastTimestampMs !== null
        ? clamp((ts - this.lastTimestampMs) / 1000, 1 / 120, 0.35)
        : 1 / 30;
    this.lastTimestampMs = ts;

    if (features.interHand.inContact) {
      this.hasHadContact = true;
    }
    if (features.quality.lowQuality) {
      this.lowQualityFrames += 1;
    }

    this.eventDetector.update(features);

    // 1. Predict step (HMM transition)
    const { tLtoR, tRtoL } = this.computeTransitionProbs(features, dtSec);
    const predLeft = this.pLeft * (1 - tLtoR) + this.pRight * tRtoL;
    const predRight = this.pRight * (1 - tRtoL) + this.pLeft * tLtoR;

    // 2. Update step (HMM observation likelihood)
    const { likLeft, likRight } = this.computeObservationLikelihoods(features);
    const unnormLeft = predLeft * likLeft;
    const unnormRight = predRight * likRight;
    const total = unnormLeft + unnormRight;

    if (total > 1e-12) {
      this.pLeft = clamp(unnormLeft / total, 0.001, 0.999);
      this.pRight = 1 - this.pLeft;
    }

    const state: BeliefState = {
      timestampMs: ts,
      pLeft: this.pLeft,
      pRight: this.pRight,
      confidence: Math.abs(this.pLeft - this.pRight),
      mostLikely: this.pLeft >= this.pRight ? 'LEFT' : 'RIGHT',
      inContact: features.interHand.inContact,
      lowQuality: features.quality.lowQuality,
    };
    this.trace.push(state);
    return state;
  }

  getBelief(): BeliefState {
    const latest = this.trace[this.trace.length - 1];
    if (latest) return latest;
    return {
      timestampMs: 0,
      pLeft: this.pLeft,
      pRight: this.pRight,
      confidence: Math.abs(this.pLeft - this.pRight),
      mostLikely: this.pLeft >= this.pRight ? 'LEFT' : 'RIGHT',
      inContact: false,
      lowQuality: false,
    };
  }

  getSnapshot(): BeliefFilterSnapshot {
    const current = this.getBelief();
    const summary = this.eventDetector.getSummary();
    return {
      ...current,
      initialSide: this.initialSide,
      hasHadContact: this.hasHadContact,
      timeline: summary.events,
      summary,
      trace: [...this.trace],
    };
  }

  /**
   * Overall tracking quality score in [0, 1] accounting for occluded/low-quality frame fraction.
   */
  getObservationQuality(): number {
    const total = Math.max(1, this.trace.length - 1);
    const badRatio = this.lowQualityFrames / total;
    return clamp(1 - badRatio, 0, 1);
  }
}
