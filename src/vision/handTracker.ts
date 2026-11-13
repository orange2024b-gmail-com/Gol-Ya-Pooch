import {
  FilesetResolver,
  HandLandmarker,
  type HandLandmarkerResult,
} from '@mediapipe/tasks-vision';
import { LandmarkOneEuroFilter } from './oneEuro';
import type {
  HandSide,
  Landmark,
  NormalizedBBox,
  RawHandDetection,
  TrackedHand,
  VisionFrameHands,
} from './types';

export function computeBBox(landmarks: readonly Landmark[]): NormalizedBBox {
  let minX = Infinity;
  let minY = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;

  for (const lm of landmarks) {
    if (lm.x < minX) minX = lm.x;
    if (lm.y < minY) minY = lm.y;
    if (lm.x > maxX) maxX = lm.x;
    if (lm.y > maxY) maxY = lm.y;
  }

  const width = Math.max(0, maxX - minX);
  const height = Math.max(0, maxY - minY);
  return {
    minX,
    minY,
    maxX,
    maxY,
    width,
    height,
    area: width * height,
  };
}

export function computePalmSize(landmarks: readonly Landmark[]): number {
  const wrist = landmarks[0];
  const middleMcp = landmarks[9];
  if (!wrist || !middleMcp) return 0.1;
  const dist = Math.hypot(middleMcp.x - wrist.x, middleMcp.y - wrist.y);
  if (dist > 1e-4) return dist;
  const indexMcp = landmarks[5];
  if (!indexMcp) return 0.1;
  return Math.max(1e-4, Math.hypot(indexMcp.x - wrist.x, indexMcp.y - wrist.y));
}

/**
 * Solves minimum-cost rectangular assignment using the Hungarian (Kuhn-Munkres) algorithm.
 * Returns an array of length rows where result[r] is the assigned column index or -1.
 */
export function hungarianAssign(costMatrix: readonly (readonly number[])[]): number[] {
  const rows = costMatrix.length;
  if (rows === 0) return [];
  const cols = costMatrix[0]?.length ?? 0;
  if (cols === 0) return new Array<number>(rows).fill(-1);

  const transposed = rows > cols;
  const n = transposed ? cols : rows;
  const m = transposed ? rows : cols;

  const cost = (i: number, j: number): number => {
    if (transposed) {
      return costMatrix[j - 1]?.[i - 1] ?? 1e9;
    }
    return costMatrix[i - 1]?.[j - 1] ?? 1e9;
  };

  const u = new Float64Array(n + 1);
  const v = new Float64Array(m + 1);
  const p = new Int32Array(m + 1);
  const way = new Int32Array(m + 1);

  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Float64Array(m + 1).fill(Infinity);
    const used = new Uint8Array(m + 1);

    do {
      used[j0] = 1;
      const i0 = p[j0]!;
      let delta = Infinity;
      let j1 = 0;

      for (let j = 1; j <= m; j++) {
        if (!used[j]) {
          const cur = cost(i0, j) - u[i0]! - v[j]!;
          if (cur < minv[j]!) {
            minv[j] = cur;
            way[j] = j0;
          }
          if (minv[j]! < delta) {
            delta = minv[j]!;
            j1 = j;
          }
        }
      }

      for (let j = 0; j <= m; j++) {
        if (used[j]) {
          u[p[j]!] = u[p[j]!]! + delta;
          v[j] = v[j]! - delta;
        } else {
          minv[j] = minv[j]! - delta;
        }
      }
      j0 = j1;
    } while (p[j0] !== 0);

    do {
      const j1 = way[j0]!;
      p[j0] = p[j1]!;
      j0 = j1;
    } while (j0 !== 0);
  }

  const rowToCol = new Array<number>(rows).fill(-1);
  for (let j = 1; j <= m; j++) {
    const matchedRow = p[j]!;
    if (matchedRow > 0) {
      if (transposed) {
        rowToCol[j - 1] = matchedRow - 1;
      } else {
        rowToCol[matchedRow - 1] = j - 1;
      }
    }
  }
  return rowToCol;
}

interface InternalHandTrack {
  side: HandSide;
  initialized: boolean;
  lastWrist: { x: number; y: number } | null;
  lastRawWrist: { x: number; y: number } | null;
  velocity: { vx: number; vy: number };
  lastSeenTimestampMs: number | null;
  lostSinceMs: number | null;
  lostFrames: number;
  lostDurationMs: number;
  confidence: number;
  rawLandmarks: Landmark[] | null;
  smoothedLandmarks: Landmark[] | null;
  bbox: NormalizedBBox | null;
  palmSize: number;
  mediaPipeHandedness: HandSide | null;
  filter: LandmarkOneEuroFilter;
}

export interface HandIdentityTrackerOptions {
  /** Weak tie-breaking weight for MediaPipe handedness ONLY when spatial prediction is unavailable or tied. */
  handednessPriorWeight: number;
  /** Maximum normalized distance allowed for associating a detection to an existing track. */
  maxAssociationDist: number;
  /** Reset One-Euro filter if hand was lost for more than this many frames. */
  maxLostFramesBeforeFilterReset: number;
  /** Velocity decay per missed frame during occlusion. */
  occlusionVelocityDecay: number;
}

const DEFAULT_TRACKER_OPTIONS: HandIdentityTrackerOptions = {
  handednessPriorWeight: 0.001,
  maxAssociationDist: 0.42,
  maxLostFramesBeforeFilterReset: 20,
  occlusionVelocityDecay: 0.92,
};

function createEmptyTrack(side: HandSide): InternalHandTrack {
  return {
    side,
    initialized: false,
    lastWrist: null,
    lastRawWrist: null,
    velocity: { vx: 0, vy: 0 },
    lastSeenTimestampMs: null,
    lostSinceMs: null,
    lostFrames: 0,
    lostDurationMs: 0,
    confidence: 0,
    rawLandmarks: null,
    smoothedLandmarks: null,
    bbox: null,
    palmSize: 0,
    mediaPipeHandedness: null,
    filter: new LandmarkOneEuroFilter(),
  };
}

function toPublicTrackedHand(track: InternalHandTrack, detected: boolean): TrackedHand {
  return {
    side: track.side,
    detected,
    occluded: !detected && track.initialized,
    lostFrames: track.lostFrames,
    lostDurationMs: track.lostDurationMs,
    confidence: detected ? track.confidence : 0,
    rawLandmarks: detected ? track.rawLandmarks : null,
    smoothedLandmarks: detected ? track.smoothedLandmarks : null,
    wrist: track.lastWrist ? { ...track.lastWrist } : null,
    velocity: { ...track.velocity },
    bbox: detected ? track.bbox : null,
    palmSize: detected ? track.palmSize : 0,
    mediaPipeHandedness: detected ? track.mediaPipeHandedness : null,
  };
}

/**
 * Maintains stable LEFT / RIGHT hand identities across frames using
 * velocity-predicted spatial continuity (Hungarian assignment on wrist position)
 * with MediaPipe handedness as a weak tie-breaking prior.
 */
export class HandIdentityTracker {
  private readonly options: HandIdentityTrackerOptions;
  private leftTrack: InternalHandTrack = createEmptyTrack('LEFT');
  private rightTrack: InternalHandTrack = createEmptyTrack('RIGHT');

  constructor(options: Partial<HandIdentityTrackerOptions> = {}) {
    this.options = { ...DEFAULT_TRACKER_OPTIONS, ...options };
  }

  reset(): void {
    this.leftTrack = createEmptyTrack('LEFT');
    this.rightTrack = createEmptyTrack('RIGHT');
  }

  private predictWrist(track: InternalHandTrack, timestampMs: number): { x: number; y: number } | null {
    if (!track.lastRawWrist || track.lastSeenTimestampMs === null) return null;
    const dtSec = Math.min(Math.max((timestampMs - track.lastSeenTimestampMs) / 1000, 0), 0.35);
    const decay = Math.pow(this.options.occlusionVelocityDecay, track.lostFrames);
    return {
      x: track.lastRawWrist.x + track.velocity.vx * dtSec * decay,
      y: track.lastRawWrist.y + track.velocity.vy * dtSec * decay,
    };
  }

  private applyMissedFrame(track: InternalHandTrack, timestampMs: number): void {
    if (!track.initialized) {
      track.lostFrames = 0;
      track.lostDurationMs = 0;
      return;
    }
    if (track.lostSinceMs === null) {
      track.lostSinceMs = timestampMs;
    }
    track.lostFrames += 1;
    track.lostDurationMs = Math.max(0, timestampMs - track.lostSinceMs);
  }

  private applyDetection(
    track: InternalHandTrack,
    det: RawHandDetection,
    timestampMs: number
  ): void {
    const rawCopy = det.landmarks.map((lm) => ({ ...lm }));
    if (track.lostFrames > this.options.maxLostFramesBeforeFilterReset) {
      track.filter.reset();
    }
    const smoothed = track.filter.filter(rawCopy, timestampMs);
    const rawWristLm = rawCopy[0] ?? { x: 0.5, y: 0.5, z: 0 };
    const newRawWrist = { x: rawWristLm.x, y: rawWristLm.y };
    const smoothedWristLm = smoothed[0] ?? rawWristLm;
    const newSmoothedWrist = { x: smoothedWristLm.x, y: smoothedWristLm.y };

    if (track.lastRawWrist && track.lastSeenTimestampMs !== null) {
      const dtSec = Math.max((timestampMs - track.lastSeenTimestampMs) / 1000, 1 / 120);
      if (dtSec <= 0.35) {
        const rawVx = (newRawWrist.x - track.lastRawWrist.x) / dtSec;
        const rawVy = (newRawWrist.y - track.lastRawWrist.y) / dtSec;
        const clamp = (v: number) => Math.max(-3.5, Math.min(3.5, v));
        track.velocity = {
          vx: 0.35 * track.velocity.vx + 0.65 * clamp(rawVx),
          vy: 0.35 * track.velocity.vy + 0.65 * clamp(rawVy),
        };
      } else {
        track.velocity = { vx: 0, vy: 0 };
      }
    }

    track.initialized = true;
    track.lastRawWrist = newRawWrist;
    track.lastWrist = newSmoothedWrist;
    track.lastSeenTimestampMs = timestampMs;
    track.lostSinceMs = null;
    track.lostFrames = 0;
    track.lostDurationMs = 0;
    track.confidence = det.handednessScore;
    track.rawLandmarks = rawCopy;
    track.smoothedLandmarks = smoothed;
    track.bbox = computeBBox(smoothed);
    track.palmSize = computePalmSize(smoothed);
    track.mediaPipeHandedness = det.handedness;
  }

  update(detections: readonly RawHandDetection[], timestampMs: number): VisionFrameHands {
    const dets = detections.slice(0, 2);
    const tracks: [InternalHandTrack, InternalHandTrack] = [this.leftTrack, this.rightTrack];
    const matchedDetForTrack = new Array<number>(2).fill(-1);

    if (dets.length > 0) {
      const bothUninitialized = !this.leftTrack.initialized && !this.rightTrack.initialized;

      if (bothUninitialized) {
        if (dets.length === 1) {
          const d0 = dets[0]!;
          const wristX = d0.landmarks[0]?.x ?? 0.5;
          const side: HandSide =
            d0.handednessScore >= 0.55 ? d0.handedness : wristX < 0.5 ? 'LEFT' : 'RIGHT';
          matchedDetForTrack[side === 'LEFT' ? 0 : 1] = 0;
        } else {
          const d0 = dets[0]!;
          const d1 = dets[1]!;
          if (d0.handedness !== d1.handedness) {
            matchedDetForTrack[d0.handedness === 'LEFT' ? 0 : 1] = 0;
            matchedDetForTrack[d1.handedness === 'LEFT' ? 0 : 1] = 1;
          } else {
            const x0 = d0.landmarks[0]?.x ?? 0;
            const x1 = d1.landmarks[0]?.x ?? 1;
            if (x0 <= x1) {
              matchedDetForTrack[0] = 0;
              matchedDetForTrack[1] = 1;
            } else {
              matchedDetForTrack[0] = 1;
              matchedDetForTrack[1] = 0;
            }
          }
        }
      } else {
        // Build cost matrix with rows = [LEFT, RIGHT], cols = detections
        const costMatrix: number[][] = [[], []];
        for (let tIdx = 0; tIdx < 2; tIdx++) {
          const track = tracks[tIdx]!;
          const predicted = this.predictWrist(track, timestampMs);
          for (let dIdx = 0; dIdx < dets.length; dIdx++) {
            const det = dets[dIdx]!;
            const detWrist = det.landmarks[0] ?? { x: 0.5, y: 0.5, z: 0 };
            const priorPenalty =
              det.handedness === track.side ? 0 : this.options.handednessPriorWeight;
            if (predicted) {
              const dist = Math.hypot(detWrist.x - predicted.x, detWrist.y - predicted.y);
              costMatrix[tIdx]!.push(dist + priorPenalty);
            } else {
              // Uninitialized complementary track gets a high base cost so the initialized track
              // keeps its nearby detection, while still allowing a far detection to initialize this track.
              costMatrix[tIdx]!.push(0.45 + priorPenalty);
            }
          }
        }

        const assignment = hungarianAssign(costMatrix);
        const usedDets = new Set<number>();

        for (let tIdx = 0; tIdx < 2; tIdx++) {
          const dIdx = assignment[tIdx] ?? -1;
          if (dIdx < 0) continue;
          const track = tracks[tIdx]!;
          const det = dets[dIdx]!;
          const predicted = this.predictWrist(track, timestampMs);
          const detWrist = det.landmarks[0] ?? { x: 0.5, y: 0.5, z: 0 };
          const spatialDist = predicted
            ? Math.hypot(detWrist.x - predicted.x, detWrist.y - predicted.y)
            : 0;

          if (
            !track.initialized ||
            spatialDist <= this.options.maxAssociationDist ||
            dets.length === 2
          ) {
            matchedDetForTrack[tIdx] = dIdx;
            usedDets.add(dIdx);
          }
        }

        // If a single detection was too far from the initialized track and the other track is uninitialized,
        // initialize the other track with it.
        if (dets.length === 1 && usedDets.size === 0) {
          const uninitIdx = !this.leftTrack.initialized
            ? 0
            : !this.rightTrack.initialized
              ? 1
              : -1;
          if (uninitIdx !== -1) {
            matchedDetForTrack[uninitIdx] = 0;
          }
        }
      }
    }

    const leftDetected = matchedDetForTrack[0]! >= 0;
    const rightDetected = matchedDetForTrack[1]! >= 0;

    if (leftDetected) {
      this.applyDetection(this.leftTrack, dets[matchedDetForTrack[0]!]!, timestampMs);
    } else {
      this.applyMissedFrame(this.leftTrack, timestampMs);
    }

    if (rightDetected) {
      this.applyDetection(this.rightTrack, dets[matchedDetForTrack[1]!]!, timestampMs);
    } else {
      this.applyMissedFrame(this.rightTrack, timestampMs);
    }

    const left = toPublicTrackedHand(this.leftTrack, leftDetected);
    const right = toPublicTrackedHand(this.rightTrack, rightDetected);
    const maxLostDurationMs = Math.max(
      leftDetected ? 0 : left.lostDurationMs,
      rightDetected ? 0 : right.lostDurationMs
    );

    return {
      timestampMs,
      left,
      right,
      maxLostDurationMs,
      bothDetected: leftDetected && rightDetected,
    };
  }
}

export function extractRawDetections(result: HandLandmarkerResult): RawHandDetection[] {
  const out: RawHandDetection[] = [];
  const landmarksList = result.landmarks ?? [];
  const handednessList = result.handedness ?? result.handednesses ?? [];

  for (let i = 0; i < landmarksList.length; i++) {
    const lms = landmarksList[i];
    if (!lms || lms.length < 21) continue;
    const category = handednessList[i]?.[0];
    const label = category?.categoryName?.toUpperCase() === 'LEFT' ? 'LEFT' : 'RIGHT';
    const score = typeof category?.score === 'number' ? category.score : 0.5;
    out.push({
      landmarks: lms.map((lm) => ({
        x: lm.x,
        y: lm.y,
        z: lm.z,
        ...(typeof lm.visibility === 'number' ? { visibility: lm.visibility } : {}),
      })),
      handedness: label,
      handednessScore: score,
    });
  }
  return out;
}

export class MediaPipeHandTracker {
  private handLandmarker: HandLandmarker | null = null;
  private readonly identityTracker: HandIdentityTracker;

  constructor(options: Partial<HandIdentityTrackerOptions> = {}) {
    this.identityTracker = new HandIdentityTracker(options);
  }

  get isReady(): boolean {
    return this.handLandmarker !== null;
  }

  resetTracking(): void {
    this.identityTracker.reset();
  }

  async init(baseUrl = import.meta.env.BASE_URL): Promise<void> {
    if (this.handLandmarker) return;
    const normalizedBase = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
    const wasmPath = `${normalizedBase}mediapipe/wasm`;
    const modelPath = `${normalizedBase}mediapipe/hand_landmarker.task`;

    const vision = await FilesetResolver.forVisionTasks(wasmPath);
    try {
      this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: modelPath,
          delegate: 'GPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    } catch {
      this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
        baseOptions: {
          modelAssetPath: modelPath,
          delegate: 'CPU',
        },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence: 0.5,
        minTrackingConfidence: 0.5,
      });
    }
  }

  detectFrame(
    video: HTMLVideoElement | HTMLCanvasElement,
    timestampMs: number
  ): { rawDetections: RawHandDetection[]; hands: VisionFrameHands } {
    if (!this.handLandmarker) {
      throw new Error('MediaPipeHandTracker has not been initialized');
    }
    const result = this.handLandmarker.detectForVideo(video, timestampMs);
    const rawDetections = extractRawDetections(result);
    const hands = this.identityTracker.update(rawDetections, timestampMs);
    return { rawDetections, hands };
  }

  close(): void {
    this.handLandmarker?.close();
    this.handLandmarker = null;
  }
}
