import { computePalmSize } from './handTracker';
import type {
  BlobDetectionResult,
  GolColorModel,
  HandSide,
  Landmark,
  OpenPalmObservation,
  PinchConfirmationState,
  PinchObservation,
  RevealVerificationResult,
  TrackedHand,
} from './types';

export interface GolDetectorConfig {
  minPinchDistNorm: number;
  maxPinchDistNorm: number;
  pinchRoiScale: number;
  palmRoiScale: number;
  maxLabMahalanobisSq: number;
  maxHueDiffDeg: number;
  maxSatDiff: number;
  maxValDiff: number;
  minBlobMatchRatio: number;
  minBlobCompactness: number;
  confirmConsecutiveFrames: number;
  lightingShiftLuminanceDelta: number;
}

export const DEFAULT_GOL_CONFIG: GolDetectorConfig = {
  minPinchDistNorm: 0.04,
  maxPinchDistNorm: 0.58,
  pinchRoiScale: 0.38,
  palmRoiScale: 0.42,
  maxLabMahalanobisSq: 11.5,
  maxHueDiffDeg: 32,
  maxSatDiff: 0.36,
  maxValDiff: 0.42,
  minBlobMatchRatio: 0.14,
  minBlobCompactness: 0.32,
  confirmConsecutiveFrames: 15,
  lightingShiftLuminanceDelta: 48,
};

export function rgbToHsv(r: number, g: number, b: number): [number, number, number] {
  const rn = r / 255;
  const gn = g / 255;
  const bn = b / 255;
  const max = Math.max(rn, gn, bn);
  const min = Math.min(rn, gn, bn);
  const delta = max - min;

  let h = 0;
  if (delta > 1e-6) {
    if (max === rn) {
      h = 60 * (((gn - bn) / delta) % 6);
    } else if (max === gn) {
      h = 60 * ((bn - rn) / delta + 2);
    } else {
      h = 60 * ((rn - gn) / delta + 4);
    }
    if (h < 0) h += 360;
  }

  const s = max <= 1e-6 ? 0 : delta / max;
  return [h, s, max];
}

function srgbChannelToLinear(c: number): number {
  const cn = c / 255;
  return cn <= 0.04045 ? cn / 12.92 : Math.pow((cn + 0.055) / 1.055, 2.4);
}

function labPivot(t: number): number {
  return t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
}

/**
 * Converts sRGB [0..255] to CIE L*a*b* (D65 illuminant).
 */
export function rgbToLab(r: number, g: number, b: number): [number, number, number] {
  const rl = srgbChannelToLinear(r);
  const gl = srgbChannelToLinear(g);
  const bl = srgbChannelToLinear(b);

  // sRGB -> XYZ (D65)
  const x = (0.4124564 * rl + 0.3575761 * gl + 0.1804375 * bl) / 0.95047;
  const y = (0.2126729 * rl + 0.7151522 * gl + 0.0721750 * bl) / 1.0;
  const z = (0.0193339 * rl + 0.1191920 * gl + 0.9503041 * bl) / 1.08883;

  const fx = labPivot(x);
  const fy = labPivot(y);
  const fz = labPivot(z);

  const l = 116 * fy - 16;
  const a = 500 * (fx - fy);
  const bStar = 200 * (fy - fz);
  return [l, a, bStar];
}

function invertSymmetric3x3(
  m00: number,
  m01: number,
  m02: number,
  m11: number,
  m12: number,
  m22: number
): [number, number, number, number, number, number, number, number, number] {
  const c00 = m11 * m22 - m12 * m12;
  const c01 = m02 * m12 - m01 * m22;
  const c02 = m01 * m12 - m02 * m11;
  const c11 = m00 * m22 - m02 * m02;
  const c12 = m01 * m02 - m00 * m12;
  const c22 = m00 * m11 - m01 * m01;

  const det = m00 * c00 + m01 * c01 + m02 * c02;
  if (Math.abs(det) < 1e-6) {
    return [1 / 64, 0, 0, 0, 1 / 64, 0, 0, 0, 1 / 64];
  }
  const invDet = 1 / det;
  return [
    c00 * invDet,
    c01 * invDet,
    c02 * invDet,
    c01 * invDet,
    c11 * invDet,
    c12 * invDet,
    c02 * invDet,
    c12 * invDet,
    c22 * invDet,
  ];
}

export function hueDistanceDeg(h1: number, h2: number): number {
  const diff = Math.abs(h1 - h2) % 360;
  return diff > 180 ? 360 - diff : diff;
}

/**
 * Calibrates the gol color model from pixels inside an on-screen calibration circle.
 * Samples HSV (circular hue mean) and CIE Lab (mean and regularized inverse covariance).
 */
export function calibrateGolColorFromRgba(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  centerNorm: { x: number; y: number },
  radiusNorm: number,
  referenceLuminance: number,
  timestampMs: number
): GolColorModel | null {
  if (width < 4 || height < 4 || rgba.length < width * height * 4) return null;

  const cx = centerNorm.x * width;
  const cy = centerNorm.y * height;
  const radiusPx = Math.max(3, radiusNorm * Math.min(width, height));
  const rSq = radiusPx * radiusPx;

  const x0 = Math.max(0, Math.floor(cx - radiusPx));
  const x1 = Math.min(width - 1, Math.ceil(cx + radiusPx));
  const y0 = Math.max(0, Math.floor(cy - radiusPx));
  const y1 = Math.min(height - 1, Math.ceil(cy + radiusPx));

  const hsvSamples: Array<[number, number, number]> = [];
  const labSamples: Array<[number, number, number]> = [];

  for (let py = y0; py <= y1; py++) {
    const dy = py - cy;
    for (let px = x0; px <= x1; px++) {
      const dx = px - cx;
      if (dx * dx + dy * dy > rSq) continue;
      const offset = (py * width + px) * 4;
      const r = rgba[offset] ?? 0;
      const g = rgba[offset + 1] ?? 0;
      const b = rgba[offset + 2] ?? 0;
      hsvSamples.push(rgbToHsv(r, g, b));
      labSamples.push(rgbToLab(r, g, b));
    }
  }

  const n = hsvSamples.length;
  if (n < 9) return null;

  let sinSum = 0;
  let cosSum = 0;
  let sSum = 0;
  let vSum = 0;
  let lSum = 0;
  let aSum = 0;
  let bSum = 0;

  for (let i = 0; i < n; i++) {
    const [h, s, v] = hsvSamples[i]!;
    const [l, a, b] = labSamples[i]!;
    const rad = (h * Math.PI) / 180;
    sinSum += Math.sin(rad);
    cosSum += Math.cos(rad);
    sSum += s;
    vSum += v;
    lSum += l;
    aSum += a;
    bSum += b;
  }

  let hMean = (Math.atan2(sinSum / n, cosSum / n) * 180) / Math.PI;
  if (hMean < 0) hMean += 360;
  const sMean = sSum / n;
  const vMean = vSum / n;

  const lMean = lSum / n;
  const aMean = aSum / n;
  const bMean = bSum / n;

  let hVar = 0;
  let sVar = 0;
  let vVar = 0;
  let c00 = 0;
  let c01 = 0;
  let c02 = 0;
  let c11 = 0;
  let c12 = 0;
  let c22 = 0;

  for (let i = 0; i < n; i++) {
    const [h, s, v] = hsvSamples[i]!;
    const [l, a, b] = labSamples[i]!;
    const hd = hueDistanceDeg(h, hMean);
    hVar += hd * hd;
    sVar += (s - sMean) * (s - sMean);
    vVar += (v - vMean) * (v - vMean);

    const dl = l - lMean;
    const da = a - aMean;
    const db = b - bMean;
    c00 += dl * dl;
    c01 += dl * da;
    c02 += dl * db;
    c11 += da * da;
    c12 += da * db;
    c22 += db * db;
  }

  // Regularize covariance so flat/uniform synthetic swatches stay well-conditioned
  const regL = 36;
  const regAB = 25;
  c00 = c00 / n + regL;
  c01 = c01 / n;
  c02 = c02 / n;
  c11 = c11 / n + regAB;
  c12 = c12 / n;
  c22 = c22 / n + regAB;

  return {
    calibratedAtMs: timestampMs,
    referenceLuminance,
    hsvMean: [hMean, sMean, vMean],
    hsvStd: [Math.sqrt(hVar / n), Math.sqrt(sVar / n), Math.sqrt(vVar / n)],
    labMean: [lMean, aMean, bMean],
    labCovarianceInv: invertSymmetric3x3(c00, c01, c02, c11, c12, c22),
    sampleCount: n,
  };
}

export function detectLightingShift(
  currentLuminance: number,
  model: GolColorModel,
  config: GolDetectorConfig = DEFAULT_GOL_CONFIG
): boolean {
  return Math.abs(currentLuminance - model.referenceLuminance) > config.lightingShiftLuminanceDelta;
}

export function doesPixelMatchGol(
  r: number,
  g: number,
  b: number,
  model: GolColorModel,
  config: GolDetectorConfig = DEFAULT_GOL_CONFIG
): boolean {
  const [l, a, bStar] = rgbToLab(r, g, b);
  const dl = l - model.labMean[0];
  const da = a - model.labMean[1];
  const db = bStar - model.labMean[2];
  const inv = model.labCovarianceInv;

  const mahalanobisSq =
    dl * (inv[0] * dl + inv[1] * da + inv[2] * db) +
    da * (inv[3] * dl + inv[4] * da + inv[5] * db) +
    db * (inv[6] * dl + inv[7] * da + inv[8] * db);

  if (mahalanobisSq > config.maxLabMahalanobisSq) {
    return false;
  }

  const [h, s, v] = rgbToHsv(r, g, b);
  const sTol = Math.max(config.maxSatDiff, model.hsvStd[1] * 2.5);
  const vTol = Math.max(config.maxValDiff, model.hsvStd[2] * 2.5);

  if (Math.abs(s - model.hsvMean[1]) > sTol || Math.abs(v - model.hsvMean[2]) > vTol) {
    return false;
  }

  // Hue is unreliable at very low saturation (achromatic colors)
  if (model.hsvMean[1] > 0.18 && s > 0.12) {
    const hTol = Math.max(config.maxHueDiffDeg, model.hsvStd[0] * 2.5);
    if (hueDistanceDeg(h, model.hsvMean[0]) > hTol) {
      return false;
    }
  }

  return true;
}

/**
 * Checks a circular ROI for a compact blob matching the calibrated gol color model.
 * Uses typed arrays and requires both sufficient match ratio and spatial compactness.
 */
export function detectGolBlobInRoi(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  roiCenterNorm: { x: number; y: number },
  roiRadiusNorm: number,
  model: GolColorModel,
  config: GolDetectorConfig = DEFAULT_GOL_CONFIG
): BlobDetectionResult {
  if (width < 4 || height < 4 || rgba.length < width * height * 4) {
    return {
      matchRatio: 0,
      compactness: 0,
      matchedPixels: 0,
      totalPixels: 0,
      detected: false,
    };
  }

  const cx = roiCenterNorm.x * width;
  const cy = roiCenterNorm.y * height;
  const radiusPx = Math.max(3, roiRadiusNorm * Math.min(width, height));
  const rSq = radiusPx * radiusPx;

  const x0 = Math.max(0, Math.floor(cx - radiusPx));
  const x1 = Math.min(width - 1, Math.ceil(cx + radiusPx));
  const y0 = Math.max(0, Math.floor(cy - radiusPx));
  const y1 = Math.min(height - 1, Math.ceil(cy + radiusPx));

  let totalPixels = 0;
  let matchedPixels = 0;
  let sumX = 0;
  let sumY = 0;
  let sumX2 = 0;
  let sumY2 = 0;
  let minMx = x1;
  let maxMx = x0;
  let minMy = y1;
  let maxMy = y0;

  for (let py = y0; py <= y1; py++) {
    const dy = py - cy;
    for (let px = x0; px <= x1; px++) {
      const dx = px - cx;
      if (dx * dx + dy * dy > rSq) continue;
      totalPixels += 1;

      const offset = (py * width + px) * 4;
      const r = rgba[offset] ?? 0;
      const g = rgba[offset + 1] ?? 0;
      const b = rgba[offset + 2] ?? 0;
      if (doesPixelMatchGol(r, g, b, model, config)) {
        matchedPixels += 1;
        sumX += px;
        sumY += py;
        sumX2 += px * px;
        sumY2 += py * py;
        if (px < minMx) minMx = px;
        if (px > maxMx) maxMx = px;
        if (py < minMy) minMy = py;
        if (py > maxMy) maxMy = py;
      }
    }
  }

  if (totalPixels === 0 || matchedPixels < 4) {
    return {
      matchRatio: 0,
      compactness: 0,
      matchedPixels,
      totalPixels,
      detected: false,
    };
  }

  const matchRatio = matchedPixels / totalPixels;

  // Spatial compactness: compare bounding-box fill ratio and normalized variance concentration
  const bboxW = Math.max(1, maxMx - minMx + 1);
  const bboxH = Math.max(1, maxMy - minMy + 1);
  const bboxFill = Math.min(1, matchedPixels / (bboxW * bboxH));

  const meanX = sumX / matchedPixels;
  const meanY = sumY / matchedPixels;
  const varX = Math.max(0, sumX2 / matchedPixels - meanX * meanX);
  const varY = Math.max(0, sumY2 / matchedPixels - meanY * meanY);
  const spreadNorm = Math.sqrt(varX + varY) / Math.max(1, radiusPx);
  const concentration = 1 / (1 + spreadNorm * 1.4);

  const compactness = 0.65 * bboxFill + 0.35 * concentration;
  const detected =
    matchRatio >= config.minBlobMatchRatio && compactness >= config.minBlobCompactness;

  return {
    matchRatio,
    compactness,
    matchedPixels,
    totalPixels,
    detected,
  };
}

export function evaluatePinchForHand(
  hand: TrackedHand,
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  model: GolColorModel,
  config: GolDetectorConfig = DEFAULT_GOL_CONFIG
): PinchObservation | null {
  const lms = hand.smoothedLandmarks;
  if (!hand.detected || !lms || lms.length < 21) return null;

  const thumbTip = lms[4]!;
  const indexTip = lms[8]!;
  const palmSize = Math.max(0.02, computePalmSize(lms));
  const pinchDistNorm = Math.hypot(thumbTip.x - indexTip.x, thumbTip.y - indexTip.y) / palmSize;

  const pinchPostureOk =
    pinchDistNorm >= config.minPinchDistNorm && pinchDistNorm <= config.maxPinchDistNorm;

  const roiCenter = {
    x: (thumbTip.x + indexTip.x) * 0.5,
    y: (thumbTip.y + indexTip.y) * 0.5,
  };
  const roiRadiusNorm = Math.max(0.025, palmSize * config.pinchRoiScale);

  const blob = detectGolBlobInRoi(rgba, width, height, roiCenter, roiRadiusNorm, model, config);
  const golPresentInFrame = pinchPostureOk && blob.detected;

  return {
    side: hand.side,
    pinchPostureOk,
    pinchDistanceNorm: pinchDistNorm,
    roiCenter,
    roiRadiusNorm,
    blob,
    golPresentInFrame,
  };
}

/**
 * Tracks N consecutive positive frames (default 15, ~0.5 s at 30 FPS) to confirm gol sighting.
 */
export class PinchConfirmationTracker {
  private consecutiveFrames = 0;
  private activeSide: HandSide | null = null;
  private confirmedSide: HandSide | null = null;
  private readonly requiredFrames: number;

  constructor(requiredFrames = DEFAULT_GOL_CONFIG.confirmConsecutiveFrames) {
    this.requiredFrames = Math.max(1, requiredFrames);
  }

  reset(): void {
    this.consecutiveFrames = 0;
    this.activeSide = null;
    this.confirmedSide = null;
  }

  update(observations: readonly PinchObservation[]): PinchConfirmationState {
    if (this.confirmedSide !== null) {
      return {
        confirmed: true,
        confirmedSide: this.confirmedSide,
        activeSide: this.confirmedSide,
        consecutiveFrames: this.requiredFrames,
        requiredFrames: this.requiredFrames,
        progress: 1,
      };
    }

    const positive = observations.filter((obs) => obs.golPresentInFrame);
    if (positive.length === 1) {
      const winner = positive[0]!;
      if (this.activeSide === winner.side || this.activeSide === null) {
        this.activeSide = winner.side;
        this.consecutiveFrames += 1;
      } else {
        this.activeSide = winner.side;
        this.consecutiveFrames = 1;
      }
    } else {
      // Neither hand or ambiguous both hands -> reset consecutive count
      this.consecutiveFrames = 0;
      this.activeSide = null;
    }

    if (this.consecutiveFrames >= this.requiredFrames && this.activeSide !== null) {
      this.confirmedSide = this.activeSide;
    }

    return {
      confirmed: this.confirmedSide !== null,
      confirmedSide: this.confirmedSide,
      activeSide: this.activeSide,
      consecutiveFrames: Math.min(this.consecutiveFrames, this.requiredFrames),
      requiredFrames: this.requiredFrames,
      progress: Math.min(1, this.consecutiveFrames / this.requiredFrames),
    };
  }
}

function dist2d(a: Landmark, b: Landmark): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/**
 * Checks whether a hand is held as an open palm facing the camera.
 */
export function evaluateOpenPalmPosture(landmarks: readonly Landmark[]): {
  fingersExtended: boolean;
  palmFacingCamera: boolean;
  isOpenPalm: boolean;
  palmCenter: { x: number; y: number };
  palmRadiusNorm: number;
} {
  if (landmarks.length < 21) {
    return {
      fingersExtended: false,
      palmFacingCamera: false,
      isOpenPalm: false,
      palmCenter: { x: 0.5, y: 0.5 },
      palmRadiusNorm: 0.05,
    };
  }

  const wrist = landmarks[0]!;
  const thumbMcp = landmarks[2]!;
  const thumbTip = landmarks[4]!;
  const indexMcp = landmarks[5]!;
  const indexPip = landmarks[6]!;
  const indexTip = landmarks[8]!;
  const middleMcp = landmarks[9]!;
  const middlePip = landmarks[10]!;
  const middleTip = landmarks[12]!;
  const ringMcp = landmarks[13]!;
  const ringPip = landmarks[14]!;
  const ringTip = landmarks[16]!;
  const pinkyMcp = landmarks[17]!;
  const pinkyPip = landmarks[18]!;
  const pinkyTip = landmarks[20]!;

  const palmSize = Math.max(0.02, computePalmSize(landmarks));

  const indexExt = dist2d(wrist, indexTip) > dist2d(wrist, indexPip) * 1.15;
  const middleExt = dist2d(wrist, middleTip) > dist2d(wrist, middlePip) * 1.15;
  const ringExt = dist2d(wrist, ringTip) > dist2d(wrist, ringPip) * 1.15;
  const pinkyExt = dist2d(wrist, pinkyTip) > dist2d(wrist, pinkyPip) * 1.15;
  const thumbExt = dist2d(pinkyMcp, thumbTip) > dist2d(pinkyMcp, thumbMcp) * 1.05;

  const extendedCount =
    Number(indexExt) + Number(middleExt) + Number(ringExt) + Number(pinkyExt) + Number(thumbExt);
  const fingersExtended = extendedCount >= 4 && indexExt && middleExt;

  // Palm facing camera check: palm spans wide (indexMcp to pinkyMcp) and fingers spread outward
  // without extreme foreshortening or curled-in z.
  const palmWidthNorm = dist2d(indexMcp, pinkyMcp) / palmSize;
  const avgTipSpread =
    (dist2d(indexTip, middleTip) + dist2d(middleTip, ringTip) + dist2d(ringTip, pinkyTip)) /
    (3 * palmSize);
  const palmFacingCamera = palmWidthNorm >= 0.35 && avgTipSpread >= 0.14;

  const palmCenter = {
    x: (wrist.x + indexMcp.x + middleMcp.x + ringMcp.x + pinkyMcp.x) / 5,
    y: (wrist.y + indexMcp.y + middleMcp.y + ringMcp.y + pinkyMcp.y) / 5,
  };

  return {
    fingersExtended,
    palmFacingCamera,
    isOpenPalm: fingersExtended && palmFacingCamera,
    palmCenter,
    palmRadiusNorm: Math.max(0.035, palmSize * DEFAULT_GOL_CONFIG.palmRoiScale),
  };
}

export function verifyRevealHands(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  leftHand: TrackedHand,
  rightHand: TrackedHand,
  model: GolColorModel,
  config: GolDetectorConfig = DEFAULT_GOL_CONFIG
): RevealVerificationResult {
  const inspectHand = (hand: TrackedHand): OpenPalmObservation | null => {
    if (!hand.detected || !hand.smoothedLandmarks) return null;
    const posture = evaluateOpenPalmPosture(hand.smoothedLandmarks);
    const blob = detectGolBlobInRoi(
      rgba,
      width,
      height,
      posture.palmCenter,
      posture.palmRadiusNorm,
      model,
      config
    );
    return {
      side: hand.side,
      fingersExtended: posture.fingersExtended,
      palmFacingCamera: posture.palmFacingCamera,
      isOpenPalm: posture.isOpenPalm,
      palmCenter: posture.palmCenter,
      palmRadiusNorm: posture.palmRadiusNorm,
      blob,
      golDetected: posture.isOpenPalm && blob.detected,
    };
  };

  const left = inspectHand(leftHand);
  const right = inspectHand(rightHand);
  const bothOpen = Boolean(left?.isOpenPalm && right?.isOpenPalm);

  if (!bothOpen) {
    return { bothOpen: false, left, right, outcome: 'HANDS_NOT_OPEN' };
  }

  const leftHas = Boolean(left?.golDetected);
  const rightHas = Boolean(right?.golDetected);

  if (leftHas && rightHas) {
    return { bothOpen: true, left, right, outcome: 'BOTH' };
  }
  if (leftHas) {
    return { bothOpen: true, left, right, outcome: 'LEFT' };
  }
  if (rightHas) {
    return { bothOpen: true, left, right, outcome: 'RIGHT' };
  }
  return { bothOpen: true, left, right, outcome: 'NEITHER' };
}

/**
 * Scans multiple ROIs around a hand during the hiding phase (fingertips, inter-finger gaps, palm center)
 * to detect any brief sighting of the gol.
 */
export function scanHandRoisForGol(
  hand: TrackedHand,
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  model: GolColorModel,
  config: GolDetectorConfig = DEFAULT_GOL_CONFIG
): number {
  const lms = hand.smoothedLandmarks;
  if (!hand.detected || !lms || lms.length < 21) return 0;

  const palmSize = Math.max(0.02, computePalmSize(lms));
  const roiRadius = Math.max(0.025, palmSize * 0.28);

  const wrist = lms[0]!;
  const middleMcp = lms[9]!;
  const thumbTip = lms[4]!;
  const indexTip = lms[8]!;
  const middleTip = lms[12]!;

  const rois = [
    { x: (wrist.x + middleMcp.x) * 0.5, y: (wrist.y + middleMcp.y) * 0.5 },
    { x: (thumbTip.x + indexTip.x) * 0.5, y: (thumbTip.y + indexTip.y) * 0.5 },
    { x: (indexTip.x + middleTip.x) * 0.5, y: (indexTip.y + middleTip.y) * 0.5 },
  ];

  let bestScore = 0;
  for (const center of rois) {
    const blob = detectGolBlobInRoi(rgba, width, height, center, roiRadius, model, config);
    if (blob.detected) {
      bestScore = Math.max(bestScore, Math.min(1, blob.matchRatio * 2 * blob.compactness));
    }
  }
  return bestScore;
}
