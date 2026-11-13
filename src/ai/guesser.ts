import { sideLabelFa, toPersianDigits } from '../app/i18n';
import type { BeliefFilterSnapshot } from '../tracking/beliefFilter';
import type { TrackingEvent } from '../tracking/eventDetector';
import type { TemporalFusionWeights, TemporalPredictionResult } from '../tracking/temporalModel';
import type { HandSide } from '../vision/types';
import type { EnsemblePrediction } from './ensemble';

export interface FusionWeights {
  trackingWeight: number;
  behaviorWeight: number;
  bias: number;
}

export const DEFAULT_FUSION_WEIGHTS: FusionWeights = {
  trackingWeight: 1.0,
  behaviorWeight: 0.65,
  bias: 0,
};

export const DEFAULT_TEMPORAL_FUSION_WEIGHTS: TemporalFusionWeights = {
  hmmWeight: 0.286714,
  temporalWeight: 0.708812,
  bias: 0.053959,
};

export interface GuesserInput {
  belief: BeliefFilterSnapshot;
  observationQuality: number;
  behavior: EnsemblePrediction;
  weights?: Partial<FusionWeights>;
  temporal?: TemporalPredictionResult | null;
  temporalFusionWeights?: Partial<TemporalFusionWeights>;
}

export interface FinalGuessResult {
  guess: HandSide;
  probability: number;
  pLeft: number;
  pRight: number;
  trackingPLeft: number;
  hmmPLeft: number;
  temporalPLeft: number | null;
  temporalUsed: boolean;
  temporalInferenceMs: number | null;
  temporalFallbackReason: string | null;
  behaviorPLeft: number;
  trackingConfidence: number;
  observationQuality: number;
  effectiveTrackingWeight: number;
  effectiveBehaviorWeight: number;
  dominantSource: 'TRACKING' | 'BEHAVIOR' | 'COMBINED';
  explanationFa: string;
  timelineDescriptionsFa: string[];
}

function clampProb(p: number): number {
  return Math.max(0.005, Math.min(0.995, p));
}

export function logit(p: number): number {
  const cp = clampProb(p);
  return Math.log(cp / (1 - cp));
}

export function sigmoid(x: number): number {
  if (x >= 20) return 0.999999;
  if (x <= -20) return 0.000001;
  return 1 / (1 + Math.exp(-x));
}

function formatSecFa(sec: number): string {
  return toPersianDigits(Math.max(0, sec).toFixed(1));
}

export function formatTimelineEventsFa(events: readonly TrackingEvent[]): string[] {
  const lines: string[] = [];

  for (const ev of events) {
    switch (ev.type) {
      case 'CONTACT_START':
        lines.push(`ثانیه ${formatSecFa(ev.relativeSec)}: دست‌ها به هم رسیدند`);
        break;
      case 'PARTIAL_OPENING':
        lines.push(
          `ثانیه ${formatSecFa(ev.relativeSec)}: ${sideLabelFa(ev.side)} کمی باز شد${ev.duringContact ? ' (در حین تماس)' : ''}`
        );
        break;
      case 'GOL_SIGHTING':
        lines.push(`ثانیه ${formatSecFa(ev.relativeSec)}: گل برای لحظه‌ای در ${sideLabelFa(ev.side)} دیده شد`);
        break;
      case 'CONTACT_END':
        if (ev.durationMs >= 250) {
          lines.push(
            `ثانیه ${formatSecFa(ev.relativeSec)}: تماس دست‌ها پایان یافت (${toPersianDigits((ev.durationMs / 1000).toFixed(1))} ثانیه)`
          );
        }
        break;
      case 'OCCLUSION_START':
        lines.push(
          `ثانیه ${formatSecFa(ev.relativeSec)}: دید ${ev.side === 'BOTH' ? 'هر دو دست' : sideLabelFa(ev.side)} موقتاً محدود شد`
        );
        break;
      case 'OCCLUSION_END':
        break;
    }
  }

  return lines;
}

export function buildPersianExplanation(
  guess: HandSide,
  belief: BeliefFilterSnapshot,
  observationQuality: number,
  dominantSource: 'TRACKING' | 'BEHAVIOR' | 'COMBINED'
): string {
  const events = belief.timeline;
  const guessLabel = sideLabelFa(guess);

  // 1. Direct sighting in predicted hand
  const sighting = events.find(
    (e): e is Extract<TrackingEvent, { type: 'GOL_SIGHTING' }> =>
      e.type === 'GOL_SIGHTING' && e.side === guess
  );
  if (sighting) {
    return `در ثانیه ${formatSecFa(sighting.relativeSec)} گل برای لحظه‌ای در ${guessLabel} دیده شد.`;
  }

  // 2. Behavioral fallback when tracking confidence or quality is low
  if (dominantSource === 'BEHAVIOR' || (belief.confidence < 0.2 && observationQuality < 0.7)) {
    return `به دلیل دید محدود یا ابهام در حین مخفی‌سازی، بر اساس الگوی قبلی شما ${guessLabel} انتخاب شد.`;
  }

  // 3. Contact + partial opening event
  const contact = events.find(
    (e): e is Extract<TrackingEvent, { type: 'CONTACT_START' }> => e.type === 'CONTACT_START'
  );
  const partialDuringContact = events.find(
    (e): e is Extract<TrackingEvent, { type: 'PARTIAL_OPENING' }> =>
      e.type === 'PARTIAL_OPENING' && e.duringContact
  );
  const anyPartial =
    partialDuringContact ??
    events.find(
      (e): e is Extract<TrackingEvent, { type: 'PARTIAL_OPENING' }> =>
        e.type === 'PARTIAL_OPENING'
    );

  if (contact && anyPartial) {
    return `در ثانیه ${formatSecFa(contact.relativeSec)} دست‌ها به هم رسیدند و در ثانیه ${formatSecFa(anyPartial.relativeSec)} ${sideLabelFa(anyPartial.side)} کمی باز شد.`;
  }

  if (contact) {
    if (guess === belief.initialSide) {
      return `در ثانیه ${formatSecFa(contact.relativeSec)} تماس کوتاهی رخ داد، اما نشانه انتقال قوی نبود و ${guessLabel} سفت‌تر باقی ماند.`;
    }
    return `در ثانیه ${formatSecFa(contact.relativeSec)} دست‌ها به هم رسیدند و فشردگی بیشتر به نفع ${guessLabel} تغییر کرد.`;
  }

  if (anyPartial) {
    return `در ثانیه ${formatSecFa(anyPartial.relativeSec)} ${sideLabelFa(anyPartial.side)} کمی باز شد؛ احتمال بیشتر با ${guessLabel} است.`;
  }

  if (!belief.hasHadContact && guess === belief.initialSide) {
    return `دست‌ها در طول مخفی‌سازی به هم نرسیدند و گل در ${guessLabel} باقی ماند.`;
  }

  return `بر اساس مجموع نشانه‌های حرکتی و سابقه بازی، ${guessLabel} محتمل‌ترین گزینه تشخیص داده شد.`;
}

/**
 * Calibrated logistic fusion combining the HMM belief filter and the ONNX temporal model.
 * Falls back cleanly to `hmmPLeft` when the temporal prediction is unavailable.
 */
export function fuseHmmAndTemporal(
  hmmPLeft: number,
  temporal: TemporalPredictionResult | null | undefined,
  weights?: Partial<TemporalFusionWeights>
): { trackingPLeft: number; temporalUsed: boolean; temporalPLeft: number | null } {
  if (!temporal || !temporal.available || temporal.pLeft === null) {
    return {
      trackingPLeft: clampProb(hmmPLeft),
      temporalUsed: false,
      temporalPLeft: null,
    };
  }
  const w: TemporalFusionWeights = { ...DEFAULT_TEMPORAL_FUSION_WEIGHTS, ...weights };
  const fusedLogit =
    w.hmmWeight * logit(hmmPLeft) + w.temporalWeight * logit(temporal.pLeft) + w.bias;
  return {
    trackingPLeft: clampProb(sigmoid(fusedLogit)),
    temporalUsed: true,
    temporalPLeft: clampProb(temporal.pLeft),
  };
}

/**
 * Fuses tracking belief (HMM + optional ONNX temporal model) with the behavioral prior
 * via log-odds addition and produces the final guess, probability, and Persian explanation.
 */
export function makeFinalGuess(input: GuesserInput): FinalGuessResult {
  const weights: FusionWeights = { ...DEFAULT_FUSION_WEIGHTS, ...input.weights };
  const obsQuality = Math.max(0, Math.min(1, input.observationQuality));

  const hmmPLeft = clampProb(input.belief.pLeft);
  const { trackingPLeft, temporalUsed, temporalPLeft } = fuseHmmAndTemporal(
    hmmPLeft,
    input.temporal,
    input.temporalFusionWeights
  );

  const trackConf = Math.max(
    input.belief.confidence,
    temporalUsed ? Math.abs(trackingPLeft - (1 - trackingPLeft)) : 0
  );

  // Tracking dominates when clearly observed; behavior takes over when confidence/quality is low
  const effectiveTrackingWeight = weights.trackingWeight * Math.max(0.15, obsQuality);
  const effectiveBehaviorWeight =
    weights.behaviorWeight * (1 - 0.5 * trackConf * obsQuality);

  const trackLogit = logit(trackingPLeft);
  const behavLogit = logit(input.behavior.pLeft);

  const trackContrib = effectiveTrackingWeight * trackLogit;
  const behavContrib = effectiveBehaviorWeight * behavLogit;
  const fusedLogOdds = trackContrib + behavContrib + weights.bias;

  const pLeft = clampProb(sigmoid(fusedLogOdds));
  const pRight = 1 - pLeft;
  const guess: HandSide = pLeft >= pRight ? 'LEFT' : 'RIGHT';
  const probability = guess === 'LEFT' ? pLeft : pRight;

  const absTrack = Math.abs(trackContrib);
  const absBehav = Math.abs(behavContrib);
  let dominantSource: 'TRACKING' | 'BEHAVIOR' | 'COMBINED' = 'COMBINED';
  if (absTrack > absBehav * 1.35) {
    dominantSource = 'TRACKING';
  } else if (absBehav > absTrack * 1.35) {
    dominantSource = 'BEHAVIOR';
  }

  return {
    guess,
    probability,
    pLeft,
    pRight,
    trackingPLeft,
    hmmPLeft,
    temporalPLeft,
    temporalUsed,
    temporalInferenceMs: temporalUsed ? (input.temporal?.inferenceMs ?? null) : null,
    temporalFallbackReason: temporalUsed
      ? null
      : (input.temporal?.fallbackReason ?? 'HMM filter fallback'),
    behaviorPLeft: input.behavior.pLeft,
    trackingConfidence: trackConf,
    observationQuality: obsQuality,
    effectiveTrackingWeight,
    effectiveBehaviorWeight,
    dominantSource,
    explanationFa: buildPersianExplanation(guess, input.belief, obsQuality, dominantSource),
    timelineDescriptionsFa: formatTimelineEventsFa(input.belief.timeline),
  };
}
