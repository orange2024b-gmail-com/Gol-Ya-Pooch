import type { QualityReport, TrackedHand, VisionFrameHands } from './types';

export interface QualityThresholds {
  minMeanLuminance: number;
  maxMeanLuminance: number;
  minContrastStd: number;
  minBlurScore: number;
  minConfidence: number;
  minBBoxExtent: number;
  maxBBoxExtent: number;
  requiredStableDurationMs: number;
}

export const DEFAULT_QUALITY_THRESHOLDS: QualityThresholds = {
  minMeanLuminance: 48,
  maxMeanLuminance: 228,
  minContrastStd: 16,
  minBlurScore: 18,
  minConfidence: 0.7,
  minBBoxExtent: 0.13,
  maxBBoxExtent: 0.82,
  requiredStableDurationMs: 1000,
};

export interface LightingMetrics {
  meanLuminance: number;
  contrastStd: number;
  blurScore: number;
}

/**
 * Computes mean luminance, contrast std-dev, and a fast Laplacian-variance blur score
 * directly from an RGBA frame buffer using a strided subsample grid.
 */
export function computeLightingAndBlur(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  targetGridSize = 64
): LightingMetrics {
  if (width < 4 || height < 4 || rgba.length < width * height * 4) {
    return { meanLuminance: 0, contrastStd: 0, blurScore: 0 };
  }

  const cols = Math.min(width, targetGridSize);
  const rows = Math.min(height, targetGridSize);
  const stepX = Math.max(1, Math.floor(width / cols));
  const stepY = Math.max(1, Math.floor(height / rows));

  const gridW = Math.floor(width / stepX);
  const gridH = Math.floor(height / stepY);
  const luma = new Float32Array(gridW * gridH);

  let sum = 0;
  let sumSq = 0;
  let idx = 0;

  for (let gy = 0; gy < gridH; gy++) {
    const py = gy * stepY;
    for (let gx = 0; gx < gridW; gx++) {
      const px = gx * stepX;
      const offset = (py * width + px) * 4;
      const r = rgba[offset] ?? 0;
      const g = rgba[offset + 1] ?? 0;
      const b = rgba[offset + 2] ?? 0;
      const y = 0.299 * r + 0.587 * g + 0.114 * b;
      luma[idx++] = y;
      sum += y;
      sumSq += y * y;
    }
  }

  const count = idx || 1;
  const meanLuminance = sum / count;
  const variance = Math.max(0, sumSq / count - meanLuminance * meanLuminance);
  const contrastStd = Math.sqrt(variance);

  // Discrete 4-neighbor Laplacian variance across interior grid pixels
  let lapSum = 0;
  let lapSqSum = 0;
  let lapCount = 0;
  for (let gy = 1; gy < gridH - 1; gy++) {
    for (let gx = 1; gx < gridW - 1; gx++) {
      const c = luma[gy * gridW + gx]!;
      const up = luma[(gy - 1) * gridW + gx]!;
      const down = luma[(gy + 1) * gridW + gx]!;
      const left = luma[gy * gridW + (gx - 1)]!;
      const right = luma[gy * gridW + (gx + 1)]!;
      const lap = up + down + left + right - 4 * c;
      lapSum += lap;
      lapSqSum += lap * lap;
      lapCount += 1;
    }
  }

  let blurScore = 0;
  if (lapCount > 0) {
    const lapMean = lapSum / lapCount;
    blurScore = Math.max(0, lapSqSum / lapCount - lapMean * lapMean);
  }

  return { meanLuminance, contrastStd, blurScore };
}

export function evaluateHandDistance(
  left: TrackedHand,
  right: TrackedHand,
  thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS
): { distanceOk: boolean; distanceHintFa: string | null } {
  if (!left.detected || !right.detected || !left.bbox || !right.bbox) {
    return {
      distanceOk: false,
      distanceHintFa: 'هر دو دست باید کامل در کادر دوربین باشند',
    };
  }

  const leftExtent = Math.max(left.bbox.width, left.bbox.height);
  const rightExtent = Math.max(right.bbox.width, right.bbox.height);
  const minExtent = Math.min(leftExtent, rightExtent);
  const maxExtent = Math.max(leftExtent, rightExtent);

  if (minExtent < thresholds.minBBoxExtent) {
    return {
      distanceOk: false,
      distanceHintFa: 'دست‌ها کمی دور هستند؛ به دوربین نزدیک‌تر شوید',
    };
  }

  if (maxExtent > thresholds.maxBBoxExtent) {
    return {
      distanceOk: false,
      distanceHintFa: 'دست‌ها خیلی نزدیک هستند؛ کمی فاصله بگیرید',
    };
  }

  return { distanceOk: true, distanceHintFa: null };
}

export function assessFrameQuality(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  hands: VisionFrameHands,
  thresholds: QualityThresholds = DEFAULT_QUALITY_THRESHOLDS
): QualityReport {
  const { meanLuminance, contrastStd, blurScore } = computeLightingAndBlur(rgba, width, height);

  const lightingOk =
    meanLuminance >= thresholds.minMeanLuminance &&
    meanLuminance <= thresholds.maxMeanLuminance &&
    contrastStd >= thresholds.minContrastStd;

  const lightingHintFa = lightingOk ? null : 'بیشتر نور بدهید / پشت به پنجره ننشینید';
  const blurOk = blurScore >= thresholds.minBlurScore;

  const bothHandsDetected = hands.bothDetected;
  const minHandConfidence = bothHandsDetected
    ? Math.min(hands.left.confidence, hands.right.confidence)
    : 0;
  const confidenceOk = bothHandsDetected && minHandConfidence > thresholds.minConfidence;

  const { distanceOk, distanceHintFa } = evaluateHandDistance(hands.left, hands.right, thresholds);

  return {
    meanLuminance,
    contrastStd,
    blurScore,
    lightingOk,
    lightingHintFa,
    blurOk,
    bothHandsDetected,
    minHandConfidence,
    confidenceOk,
    distanceOk,
    distanceHintFa,
  };
}

/**
 * Tracks continuous duration during which both hands are detected with confidence > 0.7
 * (required >= 1000 ms for calibration).
 */
export class CalibrationStabilityTracker {
  private stableSinceMs: number | null = null;
  private readonly requiredMs: number;
  private readonly minConfidence: number;

  constructor(
    requiredMs = DEFAULT_QUALITY_THRESHOLDS.requiredStableDurationMs,
    minConfidence = DEFAULT_QUALITY_THRESHOLDS.minConfidence
  ) {
    this.requiredMs = requiredMs;
    this.minConfidence = minConfidence;
  }

  reset(): void {
    this.stableSinceMs = null;
  }

  update(
    hands: VisionFrameHands,
    timestampMs: number
  ): { stableMs: number; progress: number; passed: boolean } {
    const valid =
      hands.bothDetected &&
      hands.left.confidence > this.minConfidence &&
      hands.right.confidence > this.minConfidence;

    if (!valid) {
      this.stableSinceMs = null;
      return { stableMs: 0, progress: 0, passed: false };
    }

    if (this.stableSinceMs === null) {
      this.stableSinceMs = timestampMs;
    }

    const stableMs = Math.max(0, timestampMs - this.stableSinceMs);
    const progress = Math.min(1, stableMs / this.requiredMs);
    return {
      stableMs,
      progress,
      passed: stableMs >= this.requiredMs,
    };
  }
}
