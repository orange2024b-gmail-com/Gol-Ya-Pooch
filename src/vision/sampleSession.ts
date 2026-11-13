import { HandIdentityTracker } from './handTracker';
import { SessionRecorder } from './recorder';
import type { Landmark, RecordedSession } from './types';

function makeQuickPose(
  wrist: { x: number; y: number },
  posture: 'open' | 'pinch' | 'fist',
  mirrorX = false
): Landmark[] {
  const dir = mirrorX ? -1 : 1;
  const s = 0.16;
  const pt = (dx: number, dy: number): Landmark => ({
    x: wrist.x + dir * dx * s,
    y: wrist.y + dy * s,
    z: 0,
    visibility: 0.98,
  });

  if (posture === 'pinch') {
    return [
      pt(0, 0), pt(-0.2, -0.18), pt(-0.32, -0.42), pt(-0.34, -0.68), pt(-0.24, -0.92),
      pt(-0.18, -0.82), pt(-0.2, -1.08), pt(-0.2, -1.12), pt(-0.18, -1.04),
      pt(0, -0.88), pt(0.02, -0.68), pt(0.02, -0.52), pt(0.02, -0.42),
      pt(0.2, -0.82), pt(0.2, -0.62), pt(0.18, -0.48), pt(0.16, -0.38),
      pt(0.38, -0.7), pt(0.36, -0.52), pt(0.34, -0.4), pt(0.32, -0.32),
    ];
  }
  if (posture === 'open') {
    return [
      pt(0, 0), pt(-0.22, -0.18), pt(-0.38, -0.38), pt(-0.52, -0.56), pt(-0.64, -0.74),
      pt(-0.22, -0.82), pt(-0.26, -1.16), pt(-0.28, -1.42), pt(-0.3, -1.66),
      pt(0, -0.88), pt(0, -1.26), pt(0, -1.54), pt(0, -1.78),
      pt(0.22, -0.82), pt(0.25, -1.18), pt(0.27, -1.44), pt(0.29, -1.66),
      pt(0.42, -0.7), pt(0.47, -0.98), pt(0.51, -1.18), pt(0.55, -1.36),
    ];
  }
  return [
    pt(0, 0), pt(-0.2, -0.18), pt(-0.3, -0.38), pt(-0.22, -0.54), pt(-0.12, -0.58),
    pt(-0.18, -0.8), pt(-0.16, -0.58), pt(-0.14, -0.42), pt(-0.12, -0.32),
    pt(0, -0.84), pt(0, -0.56), pt(0, -0.4), pt(0, -0.3),
    pt(0.18, -0.8), pt(0.16, -0.56), pt(0.14, -0.4), pt(0.12, -0.3),
    pt(0.35, -0.68), pt(0.32, -0.5), pt(0.28, -0.38), pt(0.24, -0.28),
  ];
}

/**
 * Generates a built-in numeric-only recorded session (show pinch -> crossing hide -> reveal)
 * so users and automated tests can inspect record/replay mode without a physical camera.
 */
export function createBuiltInSampleSession(): RecordedSession {
  const tracker = new HandIdentityTracker();
  const recorder = new SessionRecorder();
  recorder.start({
    source: 'synthetic',
    notes: 'Built-in Phase 1 sample session: 16-frame pinch, 1-cross hide, left-hand reveal',
  });

  // Frames 0..17: Show gol in LEFT hand pinch (18 frames >= N=15)
  for (let f = 0; f < 18; f++) {
    const ts = f * 33;
    const rawDetections = [
      { landmarks: makeQuickPose({ x: 0.3, y: 0.7 }, 'pinch'), handedness: 'LEFT' as const, handednessScore: 0.94 },
      { landmarks: makeQuickPose({ x: 0.7, y: 0.7 }, 'open', true), handedness: 'RIGHT' as const, handednessScore: 0.92 },
    ];
    const hands = tracker.update(rawDetections, ts);
    recorder.recordFrame({
      timestampMs: ts,
      rawDetections,
      hands,
      quality: {
        meanLuminance: 132,
        contrastStd: 34,
        blurScore: 110,
        lightingOk: true,
        lightingHintFa: null,
        blurOk: true,
        bothHandsDetected: true,
        minHandConfidence: 0.92,
        confidenceOk: true,
        distanceOk: true,
        distanceHintFa: null,
      },
      pinch: [
        {
          side: 'LEFT',
          pinchPostureOk: true,
          pinchDistanceNorm: 0.14,
          roiCenter: { x: 0.27, y: 0.54 },
          roiRadiusNorm: 0.05,
          blob: { matchRatio: 0.42, compactness: 0.78, matchedPixels: 48, totalPixels: 114, detected: true },
          golPresentInFrame: true,
        },
      ],
    });
  }

  // Frames 18..47: Hands cross while hiding (MediaPipe handedness flips at crossing)
  for (let f = 18; f < 48; f++) {
    const ts = f * 33;
    const t = (f - 18) / 29;
    const lx = 0.3 + 0.4 * t;
    const rx = 0.7 - 0.4 * t;
    const crossed = lx > rx;
    const rawDetections = [
      {
        landmarks: makeQuickPose({ x: lx, y: 0.68 - 0.05 * Math.sin(t * Math.PI) }, 'fist'),
        handedness: crossed ? ('RIGHT' as const) : ('LEFT' as const),
        handednessScore: 0.89,
      },
      {
        landmarks: makeQuickPose({ x: rx, y: 0.68 + 0.05 * Math.sin(t * Math.PI) }, 'fist', true),
        handedness: crossed ? ('LEFT' as const) : ('RIGHT' as const),
        handednessScore: 0.89,
      },
    ];
    const hands = tracker.update(rawDetections, ts);
    recorder.recordFrame({ timestampMs: ts, rawDetections, hands });
  }

  // Frame 48: Open-palm reveal with gol in LEFT
  {
    const ts = 48 * 33;
    const rawDetections = [
      { landmarks: makeQuickPose({ x: 0.7, y: 0.7 }, 'open'), handedness: 'RIGHT' as const, handednessScore: 0.93 },
      { landmarks: makeQuickPose({ x: 0.3, y: 0.7 }, 'open', true), handedness: 'LEFT' as const, handednessScore: 0.93 },
    ];
    const hands = tracker.update(rawDetections, ts);
    recorder.recordFrame({
      timestampMs: ts,
      rawDetections,
      hands,
      reveal: {
        bothOpen: true,
        outcome: 'LEFT',
        leftGolDetected: true,
        rightGolDetected: false,
      },
    });
  }

  return recorder.stop();
}
