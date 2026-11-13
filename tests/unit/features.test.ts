import { describe, expect, it } from 'vitest';
import {
  FeatureExtractor,
  computeConvexHullArea2D,
  computeFistCompactness,
  computeHandFingerAngles,
  computeJointFlexionRad,
} from '../../src/vision/features';
import { HandIdentityTracker, computePalmSize } from '../../src/vision/handTracker';
import { SessionRecorder, parseRecordedSession, replayRecordedSession } from '../../src/vision/recorder';
import type { Landmark, VisionFrameHands } from '../../src/vision/types';
import { makeSyntheticHandLandmarks } from '../vision/fixtures';

function makeHandsFrame(
  timestampMs: number,
  leftLms: Landmark[] | null,
  rightLms: Landmark[] | null,
  options: {
    leftVel?: { vx: number; vy: number };
    rightVel?: { vx: number; vy: number };
    leftConf?: number;
    rightConf?: number;
  } = {}
): VisionFrameHands {
  const leftDet = leftLms !== null;
  const rightDet = rightLms !== null;
  return {
    timestampMs,
    left: {
      side: 'LEFT',
      detected: leftDet,
      occluded: !leftDet,
      lostFrames: leftDet ? 0 : 1,
      lostDurationMs: leftDet ? 0 : 33,
      confidence: leftDet ? (options.leftConf ?? 0.92) : 0,
      rawLandmarks: leftLms,
      smoothedLandmarks: leftLms,
      wrist: leftLms?.[0] ? { x: leftLms[0].x, y: leftLms[0].y } : null,
      velocity: options.leftVel ?? { vx: 0, vy: 0 },
      bbox: leftLms
        ? { minX: 0.2, minY: 0.4, maxX: 0.4, maxY: 0.7, width: 0.2, height: 0.3, area: 0.06 }
        : null,
      palmSize: leftLms ? computePalmSize(leftLms) : 0,
      mediaPipeHandedness: leftDet ? 'LEFT' : null,
    },
    right: {
      side: 'RIGHT',
      detected: rightDet,
      occluded: !rightDet,
      lostFrames: rightDet ? 0 : 1,
      lostDurationMs: rightDet ? 0 : 33,
      confidence: rightDet ? (options.rightConf ?? 0.92) : 0,
      rawLandmarks: rightLms,
      smoothedLandmarks: rightLms,
      wrist: rightLms?.[0] ? { x: rightLms[0].x, y: rightLms[0].y } : null,
      velocity: options.rightVel ?? { vx: 0, vy: 0 },
      bbox: rightLms
        ? { minX: 0.6, minY: 0.4, maxX: 0.8, maxY: 0.7, width: 0.2, height: 0.3, area: 0.06 }
        : null,
      palmSize: rightLms ? computePalmSize(rightLms) : 0,
      mediaPipeHandedness: rightDet ? 'RIGHT' : null,
    },
    maxLostDurationMs: !leftDet || !rightDet ? 33 : 0,
    bothDetected: leftDet && rightDet,
  };
}

function interpolateLandmarks(a: readonly Landmark[], b: readonly Landmark[], t: number): Landmark[] {
  return a.map((la, i) => {
    const lb = b[i]!;
    return {
      x: la.x + (lb.x - la.x) * t,
      y: la.y + (lb.y - la.y) * t,
      z: la.z + (lb.z - la.z) * t,
      visibility: 0.98,
    };
  });
}

describe('Per-frame feature extraction (src/vision/features.ts)', () => {
  it('computes joint flexion angles accurately for straight, 90-deg, and folded segments', () => {
    const straight = computeJointFlexionRad(
      { x: 0, y: 0, z: 0 },
      { x: 0, y: -1, z: 0 },
      { x: 0, y: -2, z: 0 }
    );
    const rightAngle = computeJointFlexionRad(
      { x: 0, y: 0, z: 0 },
      { x: 0, y: -1, z: 0 },
      { x: 1, y: -1, z: 0 }
    );
    const folded = computeJointFlexionRad(
      { x: 0, y: 0, z: 0 },
      { x: 0, y: -1, z: 0 },
      { x: 0, y: 0, z: 0 }
    );

    expect(straight).toBeCloseTo(0, 5);
    expect(rightAngle).toBeCloseTo(Math.PI / 2, 5);
    expect(folded).toBeCloseTo(Math.PI, 5);
  });

  it('distinguishes open palm from closed fist and remains palm-size invariant', () => {
    const openSmall = makeSyntheticHandLandmarks({ wrist: { x: 0.3, y: 0.7 }, scale: 0.12, posture: 'open' });
    const fistSmall = makeSyntheticHandLandmarks({ wrist: { x: 0.3, y: 0.7 }, scale: 0.12, posture: 'fist' });
    const fistLarge = makeSyntheticHandLandmarks({ wrist: { x: 0.3, y: 0.7 }, scale: 0.26, posture: 'fist' });

    const openAngles = computeHandFingerAngles(openSmall);
    const fistAngles = computeHandFingerAngles(fistSmall);
    expect(openAngles.totalCurl).toBeLessThan(0.15);
    expect(fistAngles.totalCurl).toBeGreaterThan(0.55);

    const openComp = computeFistCompactness(openSmall, computePalmSize(openSmall));
    const fistCompSmall = computeFistCompactness(fistSmall, computePalmSize(fistSmall));
    const fistCompLarge = computeFistCompactness(fistLarge, computePalmSize(fistLarge));

    expect(fistCompSmall.fistCompactness).toBeGreaterThan(openComp.fistCompactness + 0.2);
    expect(fistCompSmall.convexHullAreaNorm).toBeLessThan(openComp.convexHullAreaNorm);
    expect(fistCompSmall.fistCompactness).toBeCloseTo(fistCompLarge.fistCompactness, 3);

    expect(
      computeConvexHullArea2D([
        { x: 0, y: 0 },
        { x: 2, y: 0 },
        { x: 2, y: 3 },
        { x: 0, y: 3 },
        { x: 1, y: 1 },
      ])
    ).toBeCloseTo(6, 5);
  });

  it('tracks compactness delta, wrist kinematics, jitter, contact duration, and partial-opening events', () => {
    const extractor = new FeatureExtractor();
    const leftFist = makeSyntheticHandLandmarks({ wrist: { x: 0.3, y: 0.7 }, posture: 'fist' });
    const leftOpen = makeSyntheticHandLandmarks({ wrist: { x: 0.3, y: 0.7 }, posture: 'open' });
    const rightFistFar = makeSyntheticHandLandmarks({
      wrist: { x: 0.75, y: 0.7 },
      posture: 'fist',
      mirrorX: true,
    });
    const rightFistNear = makeSyntheticHandLandmarks({
      wrist: { x: 0.34, y: 0.7 },
      posture: 'fist',
      mirrorX: true,
    });

    // Establish closed baseline for 6 frames apart with alternating micro-jitter
    let lastFeat: ReturnType<FeatureExtractor['update']> | null = null;
    for (let i = 0; i < 6; i++) {
      const vx = i % 2 === 0 ? 0.08 : -0.08;
      lastFeat = extractor.update({
        hands: makeHandsFrame(i * 33, leftFist, rightFistFar, {
          leftVel: { vx, vy: 0.02 },
          rightVel: { vx: 0, vy: 0 },
        }),
        golSightings: { left: 0.02, right: 0 },
      });
    }
    expect(lastFeat).not.toBeNull();
    expect(lastFeat!.interHand.inContact).toBe(false);
    expect(lastFeat!.interHand.contactDurationMs).toBe(0);
    expect(lastFeat!.left.jitter).toBeGreaterThan(0);

    // Hands enter contact and stay in contact for several frames
    for (let i = 6; i < 12; i++) {
      lastFeat = extractor.update({
        hands: makeHandsFrame(i * 33, leftFist, rightFistNear, {
          leftVel: { vx: 0.15, vy: 0 },
          rightVel: { vx: -0.1, vy: 0 },
        }),
      });
    }
    expect(lastFeat!.interHand.inContact).toBe(true);
    expect(lastFeat!.interHand.contactDurationMs).toBeGreaterThanOrEqual(5 * 33);
    expect(lastFeat!.interHand.relativeVelocityNorm.speed).toBeGreaterThan(0);

    // Brief partial opening on LEFT hand (relax toward open, then squeeze back to fist)
    const halfOpen = interpolateLandmarks(leftFist, leftOpen, 0.55);
    const relaxFeat = extractor.update({
      hands: makeHandsFrame(12 * 33, halfOpen, rightFistNear),
    });
    expect(relaxFeat.left.compactnessDelta).toBeLessThan(0);

    extractor.update({
      hands: makeHandsFrame(13 * 33, interpolateLandmarks(leftFist, leftOpen, 0.65), rightFistNear),
    });

    const recoverFeat = extractor.update({
      hands: makeHandsFrame(14 * 33, leftFist, rightFistNear),
    });
    expect(recoverFeat.left.partialOpening).toBe(true);
    expect(recoverFeat.left.partialOpeningStrength).toBeGreaterThan(0.2);
    expect(recoverFeat.left.compactnessDelta).toBeGreaterThan(0);
  });

  it('flags occlusion & low quality and integrates cleanly with SessionRecorder replay', () => {
    const tracker = new HandIdentityTracker();
    const recorder = new SessionRecorder();
    const extractor = new FeatureExtractor();
    recorder.start({ source: 'synthetic', notes: 'feature-logging-test' });

    for (let f = 0; f < 4; f++) {
      const ts = f * 33;
      const raw =
        f === 2
          ? [{ landmarks: makeSyntheticHandLandmarks({ wrist: { x: 0.7, y: 0.7 }, posture: 'fist', mirrorX: true }), handedness: 'RIGHT' as const, handednessScore: 0.9 }]
          : [
              { landmarks: makeSyntheticHandLandmarks({ wrist: { x: 0.3, y: 0.7 }, posture: 'fist' }), handedness: 'LEFT' as const, handednessScore: 0.9 },
              { landmarks: makeSyntheticHandLandmarks({ wrist: { x: 0.7, y: 0.7 }, posture: 'fist', mirrorX: true }), handedness: 'RIGHT' as const, handednessScore: 0.9 },
            ];
      const hands = tracker.update(raw, ts);
      const features = extractor.update({
        hands,
        quality: {
          meanLuminance: 120,
          contrastStd: 30,
          blurScore: f === 3 ? 8 : 90,
          lightingOk: true,
          lightingHintFa: null,
          blurOk: f !== 3,
          bothHandsDetected: hands.bothDetected,
          minHandConfidence: 0.9,
          confidenceOk: hands.bothDetected,
          distanceOk: true,
          distanceHintFa: null,
        },
      });
      recorder.recordFrame({ timestampMs: ts, rawDetections: raw, hands, features });
    }

    const parsed = parseRecordedSession(recorder.exportJson(false));
    const replay = replayRecordedSession(parsed);
    expect(replay.frames[2]!.features.quality.leftOccluded).toBe(true);
    expect(replay.frames[2]!.features.quality.lowQuality).toBe(true);
    expect(replay.frames[3]!.features.quality.blurOk).toBe(false);
    expect(replay.frames[3]!.features.quality.lowQuality).toBe(true);
  });
});
