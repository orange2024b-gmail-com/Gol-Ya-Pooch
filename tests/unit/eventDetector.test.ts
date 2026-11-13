import { describe, expect, it } from 'vitest';
import { EventDetector } from '../../src/tracking/eventDetector';
import type { FrameFeatures, HandFrameFeatures } from '../../src/vision/features';
import type { HandSide } from '../../src/vision/types';

function makeHandFeat(
  side: HandSide,
  overrides: Partial<HandFrameFeatures> = {}
): HandFrameFeatures {
  return {
    side,
    detected: true,
    occluded: false,
    confidence: 0.92,
    palmSize: 0.16,
    fingerAngles: {
      thumb: { mcp: 0.6, pip: 0.6, dip: 0.4, curl: 0.7 },
      index: { mcp: 1.0, pip: 0.8, dip: 0.4, curl: 0.75 },
      middle: { mcp: 1.0, pip: 0.8, dip: 0.4, curl: 0.75 },
      ring: { mcp: 1.0, pip: 0.8, dip: 0.4, curl: 0.75 },
      pinky: { mcp: 1.0, pip: 0.8, dip: 0.4, curl: 0.75 },
    },
    totalCurl: 0.74,
    convexHullAreaNorm: 0.55,
    meanFingertipDistNorm: 0.42,
    fistCompactness: 0.72,
    compactnessDelta: 0,
    convexHullAreaDelta: 0,
    meanFingertipDistDelta: 0,
    wristVelocityNorm: { vx: 0, vy: 0, speed: 0 },
    wristAccelerationNorm: { ax: 0, ay: 0, magnitude: 0 },
    jitter: 0.02,
    palmNormal: { nx: 0, ny: 0, nz: 1 },
    palmFacingScore: 0.9,
    depthProxy: { wristZ: 0, meanZ: 0, bboxSize: 0.2 },
    partialOpening: false,
    partialOpeningStrength: 0,
    golSightingScore: 0,
    ...overrides,
  };
}

export function makeSyntheticFeatures(
  timestampMs: number,
  options: {
    inContact?: boolean;
    contactDurationMs?: number;
    minLandmarkDistanceNorm?: number;
    left?: Partial<HandFrameFeatures>;
    right?: Partial<HandFrameFeatures>;
    anyOccluded?: boolean;
    blurOk?: boolean;
  } = {}
): FrameFeatures {
  const anyOccluded = options.anyOccluded ?? false;
  const blurOk = options.blurOk ?? true;
  return {
    timestampMs,
    left: makeHandFeat('LEFT', options.left),
    right: makeHandFeat('RIGHT', options.right),
    interHand: {
      bothDetected: !anyOccluded,
      minLandmarkDistance: (options.minLandmarkDistanceNorm ?? (options.inContact ? 0.25 : 1.5)) * 0.16,
      minLandmarkDistanceNorm: options.minLandmarkDistanceNorm ?? (options.inContact ? 0.25 : 1.5),
      inContact: options.inContact ?? false,
      contactDurationMs: options.contactDurationMs ?? 0,
      relativeVelocityNorm: { vx: 0, vy: 0, speed: 0 },
    },
    quality: {
      blurScore: blurOk ? 90 : 5,
      blurOk,
      meanLuminance: 130,
      lightingOk: true,
      minConfidence: anyOccluded ? 0 : 0.92,
      leftConfidence: 0.92,
      rightConfidence: anyOccluded ? 0 : 0.92,
      leftOccluded: false,
      rightOccluded: anyOccluded,
      anyOccluded,
      lowQuality: anyOccluded || !blurOk,
    },
  };
}

describe('EventDetector (src/tracking/eventDetector.ts)', () => {
  it('emits structured contact, partial-opening, gol-sighting, and occlusion events with relativeSec', () => {
    const detector = new EventDetector();
    detector.reset(1000);

    // Apart
    expect(detector.update(makeSyntheticFeatures(1000))).toEqual([]);

    // Contact starts at +2.0s (ts=3000)
    const cStart = detector.update(
      makeSyntheticFeatures(3000, { inContact: true, contactDurationMs: 0 })
    );
    expect(cStart).toHaveLength(1);
    expect(cStart[0]!.type).toBe('CONTACT_START');
    expect(cStart[0]!.relativeSec).toBeCloseTo(2.0, 3);

    // Partial opening during contact at +2.2s
    const po = detector.update(
      makeSyntheticFeatures(3200, {
        inContact: true,
        contactDurationMs: 200,
        right: { partialOpening: true, partialOpeningStrength: 0.65 },
      })
    );
    expect(po).toHaveLength(1);
    expect(po[0]).toMatchObject({
      type: 'PARTIAL_OPENING',
      side: 'RIGHT',
      duringContact: true,
    });

    // Contact ends at +2.5s
    const cEnd = detector.update(makeSyntheticFeatures(3500, { inContact: false }));
    expect(cEnd).toHaveLength(1);
    expect(cEnd[0]).toMatchObject({
      type: 'CONTACT_END',
      durationMs: 500,
      partialOpeningsDuringContact: ['RIGHT'],
    });

    // Debounced gol sighting
    const s1 = detector.update(makeSyntheticFeatures(3600, { right: { golSightingScore: 0.8 } }));
    const s2 = detector.update(makeSyntheticFeatures(3660, { right: { golSightingScore: 0.85 } }));
    const s3 = detector.update(makeSyntheticFeatures(3950, { right: { golSightingScore: 0.8 } }));
    expect(s1.map((e) => e.type)).toEqual(['GOL_SIGHTING']);
    expect(s2).toEqual([]);
    expect(s3.map((e) => e.type)).toEqual(['GOL_SIGHTING']);

    // Occlusion start and end
    const oStart = detector.update(makeSyntheticFeatures(4000, { anyOccluded: true }));
    const oEnd = detector.update(makeSyntheticFeatures(4200, { anyOccluded: false }));
    expect(oStart[0]).toMatchObject({ type: 'OCCLUSION_START', side: 'RIGHT' });
    expect(oEnd[0]).toMatchObject({ type: 'OCCLUSION_END', durationMs: 200 });

    const summary = detector.getSummary();
    expect(summary.contactCount).toBe(1);
    expect(summary.partialOpeningCount).toBe(1);
    expect(summary.sightingCount).toBe(2);
    expect(summary.occludedFrames).toBe(1);
    expect(summary.events).toHaveLength(7);
  });
});
