import { computePalmSize } from './handTracker';
import type {
  HandSide,
  Landmark,
  QualityReport,
  TrackedHand,
  VisionFrameHands,
} from './types';

export interface FingerFlexionAngles {
  mcp: number;
  pip: number;
  dip: number;
  /** Normalized curl in [0, 1] where 0 is fully extended and 1 is fully curled. */
  curl: number;
}

export interface HandFingerAngles {
  thumb: FingerFlexionAngles;
  index: FingerFlexionAngles;
  middle: FingerFlexionAngles;
  ring: FingerFlexionAngles;
  pinky: FingerFlexionAngles;
}

export interface HandFrameFeatures {
  side: HandSide;
  detected: boolean;
  occluded: boolean;
  confidence: number;
  palmSize: number;
  fingerAngles: HandFingerAngles;
  /** Mean normalized curl across all 5 fingers in [0, 1]. */
  totalCurl: number;
  /** 2D convex-hull area of the 21 landmarks normalized by palmSize^2. */
  convexHullAreaNorm: number;
  /** Mean fingertip-to-palm-center distance normalized by palmSize. */
  meanFingertipDistNorm: number;
  /** Combined compactness scalar in [0, 1] (higher = tighter fist). */
  fistCompactness: number;
  /** Change in compactness vs sliding window baseline (>0 squeezing, <0 relaxing). */
  compactnessDelta: number;
  convexHullAreaDelta: number;
  meanFingertipDistDelta: number;
  /** Wrist velocity normalized by palmSize (palm-widths / sec). */
  wristVelocityNorm: { vx: number; vy: number; speed: number };
  /** Wrist acceleration normalized by palmSize (palm-widths / sec^2). */
  wristAccelerationNorm: { ax: number; ay: number; magnitude: number };
  /** High-frequency tremor / jitter proxy normalized by palmSize. */
  jitter: number;
  /** Estimated 3D palm normal unit vector. */
  palmNormal: { nx: number; ny: number; nz: number };
  /** |nz| in [0, 1], higher when palm or back of hand faces the camera squarely. */
  palmFacingScore: number;
  /** Depth proxy derived from landmark z and bounding-box scale. */
  depthProxy: { wristZ: number; meanZ: number; bboxSize: number };
  /** True on frames completing a brief curl drop + recovery while holding a fist. */
  partialOpening: boolean;
  /** Strength of the detected partial opening in [0, 1]. */
  partialOpeningStrength: number;
  /** Gol sighting score in [0, 1] from ROI blob detection. */
  golSightingScore: number;
}

export interface InterHandFeatures {
  bothDetected: boolean;
  minLandmarkDistance: number;
  minLandmarkDistanceNorm: number;
  inContact: boolean;
  contactDurationMs: number;
  relativeVelocityNorm: { vx: number; vy: number; speed: number };
}

export interface FrameQualityFeatures {
  blurScore: number;
  blurOk: boolean;
  meanLuminance: number;
  lightingOk: boolean;
  minConfidence: number;
  leftConfidence: number;
  rightConfidence: number;
  leftOccluded: boolean;
  rightOccluded: boolean;
  anyOccluded: boolean;
  lowQuality: boolean;
}

export interface FrameFeatures {
  timestampMs: number;
  left: HandFrameFeatures;
  right: HandFrameFeatures;
  interHand: InterHandFeatures;
  quality: FrameQualityFeatures;
}

export interface FeatureExtractorConfig {
  /** Sliding window size (frames) for compactness delta and jitter calculations. */
  windowFrames: number;
  /** Normalized inter-hand landmark distance threshold to enter contact. */
  contactEnterDistNorm: number;
  /** Normalized inter-hand landmark distance threshold to exit contact (hysteresis). */
  contactExitDistNorm: number;
  /** Minimum baseline curl to treat hand as closed/fisted before partial opening. */
  closedCurlThreshold: number;
  /** Minimum curl drop from baseline to initiate a partial-opening candidate. */
  partialOpenMinDrop: number;
  /** Minimum fraction of the drop that must recover to trigger a partial-opening event. */
  partialOpenRecoveryRatio: number;
  /** Maximum duration (ms) of a brief partial-opening dip+recovery cycle. */
  partialOpenMaxDurationMs: number;
  /** Minimum landmark confidence below which frame quality is flagged low. */
  minConfidenceThreshold: number;
}

export const DEFAULT_FEATURE_CONFIG: FeatureExtractorConfig = {
  windowFrames: 8,
  contactEnterDistNorm: 0.42,
  contactExitDistNorm: 0.52,
  closedCurlThreshold: 0.36,
  partialOpenMinDrop: 0.055,
  partialOpenRecoveryRatio: 0.45,
  partialOpenMaxDurationMs: 900,
  minConfidenceThreshold: 0.55,
};

function clamp01(v: number): number {
  return Math.max(0, Math.min(1, v));
}

/**
 * Computes the flexion angle (radians in [0, pi]) between consecutive bone vectors
 * (a -> b) and (b -> c). Straight finger = 0 rad; folded back = pi rad.
 */
export function computeJointFlexionRad(a: Landmark, b: Landmark, c: Landmark): number {
  const v1x = b.x - a.x;
  const v1y = b.y - a.y;
  const v1z = b.z - a.z;
  const v2x = c.x - b.x;
  const v2y = c.y - b.y;
  const v2z = c.z - b.z;

  const m1 = Math.hypot(v1x, v1y, v1z);
  const m2 = Math.hypot(v2x, v2y, v2z);
  if (m1 < 1e-6 || m2 < 1e-6) return 0;

  const cosTheta = Math.max(
    -1,
    Math.min(1, (v1x * v2x + v1y * v2y + v1z * v2z) / (m1 * m2))
  );
  return Math.acos(cosTheta);
}

function makeFingerAngles(
  a: Landmark,
  b: Landmark,
  c: Landmark,
  d: Landmark,
  e: Landmark
): FingerFlexionAngles {
  const mcp = computeJointFlexionRad(a, b, c);
  const pip = computeJointFlexionRad(b, c, d);
  const dip = computeJointFlexionRad(c, d, e);
  // 2.6 rad combined flexion roughly corresponds to a tight closed finger curl
  const curl = clamp01((mcp + pip + dip) / 2.6);
  return { mcp, pip, dip, curl };
}

export function computeHandFingerAngles(landmarks: readonly Landmark[]): {
  fingerAngles: HandFingerAngles;
  totalCurl: number;
} {
  const emptyJoint: FingerFlexionAngles = { mcp: 0, pip: 0, dip: 0, curl: 0 };
  if (landmarks.length < 21) {
    return {
      fingerAngles: {
        thumb: { ...emptyJoint },
        index: { ...emptyJoint },
        middle: { ...emptyJoint },
        ring: { ...emptyJoint },
        pinky: { ...emptyJoint },
      },
      totalCurl: 0,
    };
  }

  const wrist = landmarks[0]!;
  // Thumb: wrist(0) -> cmc(1) -> mcp(2) -> ip(3) -> tip(4)
  const thumbMcp = computeJointFlexionRad(landmarks[1]!, landmarks[2]!, landmarks[3]!);
  const thumbPip = computeJointFlexionRad(landmarks[2]!, landmarks[3]!, landmarks[4]!);
  const thumbDip = computeJointFlexionRad(wrist, landmarks[1]!, landmarks[2]!);
  const thumb: FingerFlexionAngles = {
    mcp: thumbMcp,
    pip: thumbPip,
    dip: thumbDip,
    curl: clamp01((thumbMcp + thumbPip + 0.5 * thumbDip) / 2.1),
  };

  const index = makeFingerAngles(wrist, landmarks[5]!, landmarks[6]!, landmarks[7]!, landmarks[8]!);
  const middle = makeFingerAngles(wrist, landmarks[9]!, landmarks[10]!, landmarks[11]!, landmarks[12]!);
  const ring = makeFingerAngles(wrist, landmarks[13]!, landmarks[14]!, landmarks[15]!, landmarks[16]!);
  const pinky = makeFingerAngles(wrist, landmarks[17]!, landmarks[18]!, landmarks[19]!, landmarks[20]!);

  const totalCurl =
    (thumb.curl + index.curl + middle.curl + ring.curl + pinky.curl) / 5;

  return {
    fingerAngles: { thumb, index, middle, ring, pinky },
    totalCurl,
  };
}

/**
 * Computes 2D convex hull area using Andrew's monotone chain algorithm.
 */
export function computeConvexHullArea2D(points: ReadonlyArray<{ x: number; y: number }>): number {
  if (points.length < 3) return 0;
  const sorted = [...points].sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));

  const cross = (o: { x: number; y: number }, a: { x: number; y: number }, b: { x: number; y: number }) =>
    (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);

  const lower: Array<{ x: number; y: number }> = [];
  for (const p of sorted) {
    while (lower.length >= 2 && cross(lower[lower.length - 2]!, lower[lower.length - 1]!, p) <= 0) {
      lower.pop();
    }
    lower.push(p);
  }

  const upper: Array<{ x: number; y: number }> = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const p = sorted[i]!;
    while (upper.length >= 2 && cross(upper[upper.length - 2]!, upper[upper.length - 1]!, p) <= 0) {
      upper.pop();
    }
    upper.push(p);
  }

  lower.pop();
  upper.pop();
  const hull = lower.concat(upper);
  if (hull.length < 3) return 0;

  let area2 = 0;
  for (let i = 0; i < hull.length; i++) {
    const p1 = hull[i]!;
    const p2 = hull[(i + 1) % hull.length]!;
    area2 += p1.x * p2.y - p2.x * p1.y;
  }
  return Math.abs(area2) * 0.5;
}

export function computeFistCompactness(landmarks: readonly Landmark[], palmSize: number): {
  convexHullAreaNorm: number;
  meanFingertipDistNorm: number;
  fistCompactness: number;
} {
  if (landmarks.length < 21) {
    return { convexHullAreaNorm: 0, meanFingertipDistNorm: 0, fistCompactness: 0 };
  }
  const ps = Math.max(0.02, palmSize);
  const hullArea = computeConvexHullArea2D(landmarks);
  const convexHullAreaNorm = hullArea / (ps * ps);

  const wrist = landmarks[0]!;
  const indexMcp = landmarks[5]!;
  const middleMcp = landmarks[9]!;
  const ringMcp = landmarks[13]!;
  const pinkyMcp = landmarks[17]!;

  const palmCenter = {
    x: (wrist.x + indexMcp.x + middleMcp.x + ringMcp.x + pinkyMcp.x) / 5,
    y: (wrist.y + indexMcp.y + middleMcp.y + ringMcp.y + pinkyMcp.y) / 5,
  };

  const tipIndices = [4, 8, 12, 16, 20] as const;
  let tipDistSum = 0;
  for (const idx of tipIndices) {
    const tip = landmarks[idx]!;
    tipDistSum += Math.hypot(tip.x - palmCenter.x, tip.y - palmCenter.y) / ps;
  }
  const meanFingertipDistNorm = tipDistSum / tipIndices.length;

  // Compactness maps smaller hull area & closer fingertips to [0, 1]
  const areaScore = 1 / (1 + Math.max(0, convexHullAreaNorm) * 1.15);
  const tipScore = 1 / (1 + Math.max(0, meanFingertipDistNorm) * 1.6);
  const fistCompactness = clamp01(0.45 * areaScore + 0.55 * tipScore);

  return {
    convexHullAreaNorm,
    meanFingertipDistNorm,
    fistCompactness,
  };
}

export function computePalmNormalAndDepth(
  landmarks: readonly Landmark[],
  bboxArea: number,
  bboxMaxDim: number
): {
  palmNormal: { nx: number; ny: number; nz: number };
  palmFacingScore: number;
  depthProxy: { wristZ: number; meanZ: number; bboxSize: number };
} {
  if (landmarks.length < 21) {
    return {
      palmNormal: { nx: 0, ny: 0, nz: 1 },
      palmFacingScore: 0,
      depthProxy: { wristZ: 0, meanZ: 0, bboxSize: 0 },
    };
  }
  const wrist = landmarks[0]!;
  const indexMcp = landmarks[5]!;
  const pinkyMcp = landmarks[17]!;

  const ax = indexMcp.x - wrist.x;
  const ay = indexMcp.y - wrist.y;
  const az = indexMcp.z - wrist.z;
  const bx = pinkyMcp.x - wrist.x;
  const by = pinkyMcp.y - wrist.y;
  const bz = pinkyMcp.z - wrist.z;

  const cx = ay * bz - az * by;
  const cy = az * bx - ax * bz;
  const cz = ax * by - ay * bx;
  const norm = Math.hypot(cx, cy, cz);

  const palmNormal =
    norm > 1e-6
      ? { nx: cx / norm, ny: cy / norm, nz: cz / norm }
      : { nx: 0, ny: 0, nz: 1 };

  let zSum = 0;
  for (const lm of landmarks) zSum += lm.z;

  return {
    palmNormal,
    palmFacingScore: clamp01(Math.abs(palmNormal.nz)),
    depthProxy: {
      wristZ: wrist.z,
      meanZ: zSum / landmarks.length,
      bboxSize: Math.max(bboxMaxDim, Math.sqrt(Math.max(0, bboxArea))),
    },
  };
}

export function computeMinLandmarkDistance(
  leftLandmarks: readonly Landmark[],
  rightLandmarks: readonly Landmark[]
): number {
  let minDist = Infinity;
  for (const la of leftLandmarks) {
    for (const lb of rightLandmarks) {
      const d = Math.hypot(la.x - lb.x, la.y - lb.y);
      if (d < minDist) minDist = d;
    }
  }
  return Number.isFinite(minDist) ? minDist : 1;
}

interface HandHistoryState {
  compactnessWindow: number[];
  hullWindow: number[];
  tipDistWindow: number[];
  velHistory: Array<{ vx: number; vy: number }>;
  lastTimestampMs: number | null;
  lastWristVelNorm: { vx: number; vy: number } | null;
  // Partial-opening state machine
  baselineCurl: number;
  inDip: boolean;
  dipStartMs: number | null;
  dipMinCurl: number;
  dipBaselineCurl: number;
}

function createEmptyHandHistory(): HandHistoryState {
  return {
    compactnessWindow: [],
    hullWindow: [],
    tipDistWindow: [],
    velHistory: [],
    lastTimestampMs: null,
    lastWristVelNorm: null,
    baselineCurl: 0,
    inDip: false,
    dipStartMs: null,
    dipMinCurl: 1,
    dipBaselineCurl: 0,
  };
}

function pushBounded(arr: number[], value: number, maxLen: number): void {
  arr.push(value);
  if (arr.length > maxLen) arr.shift();
}

function meanOf(arr: readonly number[]): number {
  if (arr.length === 0) return 0;
  let sum = 0;
  for (const v of arr) sum += v;
  return sum / arr.length;
}

function createZeroHandFeatures(side: HandSide, hand: TrackedHand, golSightingScore: number): HandFrameFeatures {
  return {
    side,
    detected: false,
    occluded: hand.occluded,
    confidence: hand.confidence,
    palmSize: 0,
    fingerAngles: {
      thumb: { mcp: 0, pip: 0, dip: 0, curl: 0 },
      index: { mcp: 0, pip: 0, dip: 0, curl: 0 },
      middle: { mcp: 0, pip: 0, dip: 0, curl: 0 },
      ring: { mcp: 0, pip: 0, dip: 0, curl: 0 },
      pinky: { mcp: 0, pip: 0, dip: 0, curl: 0 },
    },
    totalCurl: 0,
    convexHullAreaNorm: 0,
    meanFingertipDistNorm: 0,
    fistCompactness: 0,
    compactnessDelta: 0,
    convexHullAreaDelta: 0,
    meanFingertipDistDelta: 0,
    wristVelocityNorm: { vx: 0, vy: 0, speed: 0 },
    wristAccelerationNorm: { ax: 0, ay: 0, magnitude: 0 },
    jitter: 0,
    palmNormal: { nx: 0, ny: 0, nz: 1 },
    palmFacingScore: 0,
    depthProxy: { wristZ: 0, meanZ: 0, bboxSize: 0 },
    partialOpening: false,
    partialOpeningStrength: 0,
    golSightingScore,
  };
}

export interface FeatureExtractionInput {
  hands: VisionFrameHands;
  quality?: QualityReport | null;
  golSightings?: { left: number; right: number };
}

/**
 * Stateful per-frame feature extractor computing palm-normalized kinematics,
 * compactness deltas, jitter, inter-hand contact metrics, partial-opening events,
 * and frame quality indicators.
 *
 * Note: Muscle tension cannot be directly observed at webcam resolution;
 * compactness deltas, finger curl, and wrist jitter serve as geometric proxies.
 */
export class FeatureExtractor {
  private readonly config: FeatureExtractorConfig;
  private leftState = createEmptyHandHistory();
  private rightState = createEmptyHandHistory();
  private inContact = false;
  private contactStartMs: number | null = null;
  private lastContactTimestampMs: number | null = null;

  constructor(config: Partial<FeatureExtractorConfig> = {}) {
    this.config = { ...DEFAULT_FEATURE_CONFIG, ...config };
  }

  reset(): void {
    this.leftState = createEmptyHandHistory();
    this.rightState = createEmptyHandHistory();
    this.inContact = false;
    this.contactStartMs = null;
    this.lastContactTimestampMs = null;
  }

  private extractHand(
    hand: TrackedHand,
    state: HandHistoryState,
    timestampMs: number,
    golSightingScore: number
  ): HandFrameFeatures {
    const lms = hand.smoothedLandmarks ?? hand.rawLandmarks;
    if (!hand.detected || !lms || lms.length < 21) {
      state.inDip = false;
      state.dipStartMs = null;
      state.lastWristVelNorm = null;
      state.lastTimestampMs = timestampMs;
      return createZeroHandFeatures(hand.side, hand, golSightingScore);
    }

    const palmSize = Math.max(0.02, hand.palmSize > 0 ? hand.palmSize : computePalmSize(lms));
    const { fingerAngles, totalCurl } = computeHandFingerAngles(lms);
    const { convexHullAreaNorm, meanFingertipDistNorm, fistCompactness } = computeFistCompactness(
      lms,
      palmSize
    );

    const prevCompactnessMean =
      state.compactnessWindow.length > 0 ? meanOf(state.compactnessWindow) : fistCompactness;
    const prevHullMean =
      state.hullWindow.length > 0 ? meanOf(state.hullWindow) : convexHullAreaNorm;
    const prevTipMean =
      state.tipDistWindow.length > 0 ? meanOf(state.tipDistWindow) : meanFingertipDistNorm;

    const compactnessDelta = fistCompactness - prevCompactnessMean;
    const convexHullAreaDelta = convexHullAreaNorm - prevHullMean;
    const meanFingertipDistDelta = meanFingertipDistNorm - prevTipMean;

    pushBounded(state.compactnessWindow, fistCompactness, this.config.windowFrames);
    pushBounded(state.hullWindow, convexHullAreaNorm, this.config.windowFrames);
    pushBounded(state.tipDistWindow, meanFingertipDistNorm, this.config.windowFrames);

    // Wrist velocity, acceleration, and jitter normalized by palm size
    const vxNorm = hand.velocity.vx / palmSize;
    const vyNorm = hand.velocity.vy / palmSize;
    const speedNorm = Math.hypot(vxNorm, vyNorm);

    let axNorm = 0;
    let ayNorm = 0;
    if (state.lastWristVelNorm && state.lastTimestampMs !== null) {
      const dtSec = (timestampMs - state.lastTimestampMs) / 1000;
      if (dtSec > 1e-3 && dtSec <= 0.4) {
        axNorm = (vxNorm - state.lastWristVelNorm.vx) / dtSec;
        ayNorm = (vyNorm - state.lastWristVelNorm.vy) / dtSec;
      }
    }
    state.lastWristVelNorm = { vx: vxNorm, vy: vyNorm };
    state.lastTimestampMs = timestampMs;

    state.velHistory.push({ vx: vxNorm, vy: vyNorm });
    if (state.velHistory.length > this.config.windowFrames) {
      state.velHistory.shift();
    }

    // Jitter / tremor: RMS deviation of velocity from sliding window mean velocity
    let jitter = 0;
    if (state.velHistory.length >= 3) {
      let meanVx = 0;
      let meanVy = 0;
      for (const v of state.velHistory) {
        meanVx += v.vx;
        meanVy += v.vy;
      }
      meanVx /= state.velHistory.length;
      meanVy /= state.velHistory.length;
      let devSq = 0;
      for (const v of state.velHistory) {
        const dx = v.vx - meanVx;
        const dy = v.vy - meanVy;
        devSq += dx * dx + dy * dy;
      }
      jitter = Math.sqrt(devSq / state.velHistory.length);
    }

    const bboxMaxDim = hand.bbox ? Math.max(hand.bbox.width, hand.bbox.height) : palmSize * 1.4;
    const bboxArea = hand.bbox ? hand.bbox.area : bboxMaxDim * bboxMaxDim;
    const { palmNormal, palmFacingScore, depthProxy } = computePalmNormalAndDepth(
      lms,
      bboxArea,
      bboxMaxDim
    );

    // Partial-opening detector: brief curl drop below closed baseline then recovery
    let partialOpening = false;
    let partialOpeningStrength = 0;

    if (state.baselineCurl === 0) {
      state.baselineCurl = totalCurl;
    }

    if (!state.inDip) {
      const dropFromBaseline = state.baselineCurl - totalCurl;
      if (
        state.baselineCurl >= this.config.closedCurlThreshold &&
        dropFromBaseline >= this.config.partialOpenMinDrop
      ) {
        state.inDip = true;
        state.dipStartMs = timestampMs;
        state.dipMinCurl = totalCurl;
        state.dipBaselineCurl = state.baselineCurl;
      } else {
        // Track fist closing quickly upward, and adapt slowly downward so brief dips stand out
        const alpha = totalCurl > state.baselineCurl ? 0.35 : 0.04;
        state.baselineCurl = (1 - alpha) * state.baselineCurl + alpha * totalCurl;
      }
    } else {
      state.dipMinCurl = Math.min(state.dipMinCurl, totalCurl);
      const drop = state.dipBaselineCurl - state.dipMinCurl;
      const elapsed = state.dipStartMs !== null ? timestampMs - state.dipStartMs : 0;

      if (elapsed > this.config.partialOpenMaxDurationMs) {
        // Hand stayed open too long; reset baseline rather than treating as brief partial opening
        state.inDip = false;
        state.dipStartMs = null;
        state.baselineCurl = totalCurl;
      } else if (
        drop >= this.config.partialOpenMinDrop &&
        totalCurl >= state.dipMinCurl + drop * this.config.partialOpenRecoveryRatio
      ) {
        partialOpening = true;
        partialOpeningStrength = clamp01(drop / 0.16);
        state.inDip = false;
        state.dipStartMs = null;
        state.baselineCurl = Math.max(totalCurl, state.dipBaselineCurl * 0.9);
      }
    }

    return {
      side: hand.side,
      detected: true,
      occluded: false,
      confidence: hand.confidence,
      palmSize,
      fingerAngles,
      totalCurl,
      convexHullAreaNorm,
      meanFingertipDistNorm,
      fistCompactness,
      compactnessDelta,
      convexHullAreaDelta,
      meanFingertipDistDelta,
      wristVelocityNorm: { vx: vxNorm, vy: vyNorm, speed: speedNorm },
      wristAccelerationNorm: {
        ax: axNorm,
        ay: ayNorm,
        magnitude: Math.hypot(axNorm, ayNorm),
      },
      jitter,
      palmNormal,
      palmFacingScore,
      depthProxy,
      partialOpening,
      partialOpeningStrength,
      golSightingScore: clamp01(golSightingScore),
    };
  }

  update(input: FeatureExtractionInput): FrameFeatures {
    const { hands, quality } = input;
    const ts = hands.timestampMs;

    const left = this.extractHand(
      hands.left,
      this.leftState,
      ts,
      input.golSightings?.left ?? 0
    );
    const right = this.extractHand(
      hands.right,
      this.rightState,
      ts,
      input.golSightings?.right ?? 0
    );

    const leftLms = hands.left.smoothedLandmarks ?? hands.left.rawLandmarks;
    const rightLms = hands.right.smoothedLandmarks ?? hands.right.rawLandmarks;

    let minLandmarkDistance = 1;
    let minLandmarkDistanceNorm = 10;
    let relVx = 0;
    let relVy = 0;
    let relSpeed = 0;

    if (
      hands.bothDetected &&
      leftLms &&
      rightLms &&
      leftLms.length >= 21 &&
      rightLms.length >= 21
    ) {
      minLandmarkDistance = computeMinLandmarkDistance(leftLms, rightLms);
      const meanPalm = Math.max(0.02, (left.palmSize + right.palmSize) * 0.5);
      minLandmarkDistanceNorm = minLandmarkDistance / meanPalm;

      relVx = (hands.left.velocity.vx - hands.right.velocity.vx) / meanPalm;
      relVy = (hands.left.velocity.vy - hands.right.velocity.vy) / meanPalm;
      relSpeed = Math.hypot(relVx, relVy);

      if (!this.inContact && minLandmarkDistanceNorm <= this.config.contactEnterDistNorm) {
        this.inContact = true;
        this.contactStartMs = ts;
      } else if (this.inContact && minLandmarkDistanceNorm > this.config.contactExitDistNorm) {
        this.inContact = false;
        this.contactStartMs = null;
      }
      this.lastContactTimestampMs = ts;
    } else {
      // Preserve brief contact state during momentary 1-2 frame occlusion, reset if lost longer
      if (
        this.inContact &&
        this.lastContactTimestampMs !== null &&
        ts - this.lastContactTimestampMs > 250
      ) {
        this.inContact = false;
        this.contactStartMs = null;
      }
    }

    const contactDurationMs =
      this.inContact && this.contactStartMs !== null ? Math.max(0, ts - this.contactStartMs) : 0;

    const leftConfidence = hands.left.detected ? hands.left.confidence : 0;
    const rightConfidence = hands.right.detected ? hands.right.confidence : 0;
    const minConfidence = Math.min(leftConfidence, rightConfidence);
    const leftOccluded = hands.left.occluded || !hands.left.detected;
    const rightOccluded = hands.right.occluded || !hands.right.detected;
    const anyOccluded = leftOccluded || rightOccluded;

    const blurScore = quality?.blurScore ?? 100;
    const blurOk = quality?.blurOk ?? true;
    const meanLuminance = quality?.meanLuminance ?? 128;
    const lightingOk = quality?.lightingOk ?? true;

    const lowQuality =
      anyOccluded ||
      !blurOk ||
      !lightingOk ||
      minConfidence < this.config.minConfidenceThreshold;

    return {
      timestampMs: ts,
      left,
      right,
      interHand: {
        bothDetected: hands.bothDetected,
        minLandmarkDistance,
        minLandmarkDistanceNorm,
        inContact: this.inContact,
        contactDurationMs,
        relativeVelocityNorm: { vx: relVx, vy: relVy, speed: relSpeed },
      },
      quality: {
        blurScore,
        blurOk,
        meanLuminance,
        lightingOk,
        minConfidence,
        leftConfidence,
        rightConfidence,
        leftOccluded,
        rightOccluded,
        anyOccluded,
        lowQuality,
      },
    };
  }
}
