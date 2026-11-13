import type { FrameFeatures } from '../vision/features';
import type { HandSide } from '../vision/types';

export const TEMPORAL_SEQ_LEN = 64;
export const TEMPORAL_SAMPLE_RATE_HZ = 10;
export const TEMPORAL_FEATURE_NAMES = [
  'initialSideSign',
  'leftDetected',
  'rightDetected',
  'leftCurl',
  'rightCurl',
  'leftCompactness',
  'rightCompactness',
  'leftCompactnessDelta',
  'rightCompactnessDelta',
  'leftFingertipDistNorm',
  'rightFingertipDistNorm',
  'leftWristSpeed',
  'rightWristSpeed',
  'leftJitter',
  'rightJitter',
  'leftPartialOpen',
  'rightPartialOpen',
  'leftGolSighting',
  'rightGolSighting',
  'inContact',
  'interHandDistNorm',
  'contactDurationNorm',
  'lowQuality',
  'progressNorm',
] as const;

export const TEMPORAL_FEATURE_DIM = TEMPORAL_FEATURE_NAMES.length;

const MAX_EVENT_INDICES = new Set<number>([15, 16, 17, 18, 19, 22]);

function clamp(v: number, min: number, max: number): number {
  if (!Number.isFinite(v)) return min;
  return Math.max(min, Math.min(max, v));
}

/**
 * Extracts a fixed-order 24-dimensional numeric feature vector from one FrameFeatures sample.
 * Must stay bit-for-bit aligned with `training/gol_training/features.py`.
 */
export function extractFrameFeatureVector(
  frame: FrameFeatures,
  initialSide: HandSide,
  progressNorm: number
): number[] {
  const initialSign = initialSide === 'LEFT' ? 1 : -1;
  return [
    initialSign,
    frame.left.detected ? 1 : 0,
    frame.right.detected ? 1 : 0,
    clamp(frame.left.totalCurl, 0, 1),
    clamp(frame.right.totalCurl, 0, 1),
    clamp(frame.left.fistCompactness, 0, 1),
    clamp(frame.right.fistCompactness, 0, 1),
    clamp(frame.left.compactnessDelta, -1, 1),
    clamp(frame.right.compactnessDelta, -1, 1),
    clamp(frame.left.meanFingertipDistNorm, 0, 3) / 3,
    clamp(frame.right.meanFingertipDistNorm, 0, 3) / 3,
    clamp(frame.left.wristVelocityNorm.speed, 0, 10) / 10,
    clamp(frame.right.wristVelocityNorm.speed, 0, 10) / 10,
    clamp(frame.left.jitter, 0, 5) / 5,
    clamp(frame.right.jitter, 0, 5) / 5,
    frame.left.partialOpening ? clamp(Math.max(0.35, frame.left.partialOpeningStrength), 0, 1) : 0,
    frame.right.partialOpening ? clamp(Math.max(0.35, frame.right.partialOpeningStrength), 0, 1) : 0,
    clamp(frame.left.golSightingScore, 0, 1),
    clamp(frame.right.golSightingScore, 0, 1),
    frame.interHand.inContact ? 1 : 0,
    clamp(frame.interHand.minLandmarkDistanceNorm, 0, 3) / 3,
    clamp(frame.interHand.contactDurationMs, 0, 5000) / 5000,
    frame.quality.lowQuality ? 1 : 0,
    clamp(progressNorm, 0, 1),
  ];
}

export interface ResampledTemporalSequence {
  /** Row-major flat Float32Array of shape [seqLen, TEMPORAL_FEATURE_DIM]. */
  data: Float32Array;
  /** 2D numeric matrix [seqLen][TEMPORAL_FEATURE_DIM] suitable for JSON round logging. */
  matrix: number[][];
  seqLen: number;
  featureDim: number;
  sampleRateHz: number;
  rawFrameCount: number;
  resampledSteps: number;
  durationMs: number;
}

/**
 * Resamples a variable-length hiding-phase FrameFeatures sequence to a fixed rate
 * (default 10 Hz), pads with zeros up to `seqLen` (default 64), or truncates to the
 * most recent `seqLen` steps when longer.
 */
export function resampleHidingFeatures(
  frames: readonly FrameFeatures[],
  initialSide: HandSide,
  options: { seqLen?: number; sampleRateHz?: number } = {}
): ResampledTemporalSequence {
  const seqLen = Math.max(1, Math.round(options.seqLen ?? TEMPORAL_SEQ_LEN));
  const sampleRateHz = Math.max(1, options.sampleRateHz ?? TEMPORAL_SAMPLE_RATE_HZ);
  const data = new Float32Array(seqLen * TEMPORAL_FEATURE_DIM);

  if (frames.length === 0) {
    return {
      data,
      matrix: Array.from({ length: seqLen }, () =>
        new Array<number>(TEMPORAL_FEATURE_DIM).fill(0)
      ),
      seqLen,
      featureDim: TEMPORAL_FEATURE_DIM,
      sampleRateHz,
      rawFrameCount: 0,
      resampledSteps: 0,
      durationMs: 0,
    };
  }

  const sorted = [...frames].sort((a, b) => a.timestampMs - b.timestampMs);
  const tStart = sorted[0]!.timestampMs;
  const tEnd = sorted[sorted.length - 1]!.timestampMs;
  const durationMs = Math.max(0, tEnd - tStart);
  const stepMs = 1000 / sampleRateHz;

  const totalBins = Math.max(1, Math.floor(durationMs / stepMs) + 1);
  const binOffset = totalBins > seqLen ? totalBins - seqLen : 0;
  const outSteps = Math.min(seqLen, totalBins);
  const effectiveStartMs = tStart + binOffset * stepMs;

  const rawVectors = sorted.map((f, idx) =>
    extractFrameFeatureVector(
      f,
      initialSide,
      sorted.length <= 1 ? 0 : idx / (sorted.length - 1)
    )
  );

  for (let b = 0; b < outSteps; b++) {
    const binStart = effectiveStartMs + b * stepMs;
    const binEnd = binStart + stepMs;
    const binMid = binStart + stepMs * 0.5;

    const inBinIndices: number[] = [];
    for (let i = 0; i < sorted.length; i++) {
      const ts = sorted[i]!.timestampMs;
      const match =
        b === outSteps - 1
          ? ts >= binStart - 1e-6 && ts <= binEnd + 1e-6
          : ts >= binStart - 1e-6 && ts < binEnd - 1e-6;
      if (match) {
        inBinIndices.push(i);
      }
    }

    if (inBinIndices.length === 0) {
      let bestIdx = 0;
      let bestDist = Infinity;
      for (let i = 0; i < sorted.length; i++) {
        const d = Math.abs(sorted[i]!.timestampMs - binMid);
        if (d < bestDist) {
          bestDist = d;
          bestIdx = i;
        }
      }
      inBinIndices.push(bestIdx);
    }

    const rowOffset = b * TEMPORAL_FEATURE_DIM;
    const binProgress = outSteps <= 1 ? 0 : b / (outSteps - 1);

    for (let fIdx = 0; fIdx < TEMPORAL_FEATURE_DIM; fIdx++) {
      if (fIdx === TEMPORAL_FEATURE_DIM - 1) {
        data[rowOffset + fIdx] = binProgress;
        continue;
      }

      if (MAX_EVENT_INDICES.has(fIdx)) {
        let maxVal = 0;
        for (const i of inBinIndices) {
          const val = rawVectors[i]![fIdx]!;
          if (val > maxVal) maxVal = val;
        }
        data[rowOffset + fIdx] = maxVal;
      } else {
        let sumVal = 0;
        for (const i of inBinIndices) {
          sumVal += rawVectors[i]![fIdx]!;
        }
        data[rowOffset + fIdx] = sumVal / inBinIndices.length;
      }
    }
  }

  const matrix: number[][] = [];
  for (let r = 0; r < seqLen; r++) {
    const row: number[] = [];
    const base = r * TEMPORAL_FEATURE_DIM;
    for (let c = 0; c < TEMPORAL_FEATURE_DIM; c++) {
      row.push(Number(data[base + c]!.toFixed(6)));
    }
    matrix.push(row);
  }

  return {
    data,
    matrix,
    seqLen,
    featureDim: TEMPORAL_FEATURE_DIM,
    sampleRateHz,
    rawFrameCount: sorted.length,
    resampledSteps: outSteps,
    durationMs,
  };
}
