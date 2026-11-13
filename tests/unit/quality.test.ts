import { describe, expect, it } from 'vitest';
import { HandIdentityTracker } from '../../src/vision/handTracker';
import {
  CalibrationStabilityTracker,
  assessFrameQuality,
  computeLightingAndBlur,
} from '../../src/vision/quality';
import { makeDetection, makeSyntheticRgbaFrame } from '../vision/fixtures';

describe('Frame quality & calibration stability checks', () => {
  it('flags dark / low-contrast frames with Persian lighting advice', () => {
    const dark = new Uint8ClampedArray(160 * 120 * 4);
    for (let i = 0; i < dark.length; i += 4) {
      dark[i] = 15;
      dark[i + 1] = 15;
      dark[i + 2] = 15;
      dark[i + 3] = 255;
    }

    const tracker = new HandIdentityTracker();
    const hands = tracker.update(
      [makeDetection({ x: 0.3, y: 0.65 }, 'LEFT'), makeDetection({ x: 0.7, y: 0.65 }, 'RIGHT')],
      0
    );

    const report = assessFrameQuality(dark, 160, 120, hands);
    expect(report.lightingOk).toBe(false);
    expect(report.lightingHintFa).toBe('بیشتر نور بدهید / پشت به پنجره ننشینید');
  });

  it('passes lighting, distance, and confidence on a well-lit textured frame', () => {
    const lit = makeSyntheticRgbaFrame(160, 120, [135, 130, 125]);
    const metrics = computeLightingAndBlur(lit, 160, 120);
    expect(metrics.meanLuminance).toBeGreaterThan(100);
    expect(metrics.blurScore).toBeGreaterThan(20);

    const tracker = new HandIdentityTracker();
    const hands = tracker.update(
      [
        makeDetection({ x: 0.3, y: 0.7 }, 'LEFT', 0.92),
        makeDetection({ x: 0.7, y: 0.7 }, 'RIGHT', 0.91),
      ],
      100
    );

    const report = assessFrameQuality(lit, 160, 120, hands);
    expect(report.lightingOk).toBe(true);
    expect(report.confidenceOk).toBe(true);
    expect(report.distanceOk).toBe(true);
  });

  it('requires >= 1000 ms continuous detection of both hands above 0.7 confidence', () => {
    const tracker = new HandIdentityTracker();
    const stability = new CalibrationStabilityTracker(1000, 0.7);

    const h0 = tracker.update(
      [
        makeDetection({ x: 0.3, y: 0.7 }, 'LEFT', 0.85),
        makeDetection({ x: 0.7, y: 0.7 }, 'RIGHT', 0.88),
      ],
      0
    );
    expect(stability.update(h0, 0).passed).toBe(false);

    const h500 = tracker.update(
      [
        makeDetection({ x: 0.3, y: 0.7 }, 'LEFT', 0.85),
        makeDetection({ x: 0.7, y: 0.7 }, 'RIGHT', 0.88),
      ],
      500
    );
    expect(stability.update(h500, 500).passed).toBe(false);

    // Dip in confidence resets timer
    const hDip = tracker.update(
      [
        makeDetection({ x: 0.3, y: 0.7 }, 'LEFT', 0.55),
        makeDetection({ x: 0.7, y: 0.7 }, 'RIGHT', 0.88),
      ],
      700
    );
    expect(stability.update(hDip, 700).stableMs).toBe(0);

    const hRestart = tracker.update(
      [
        makeDetection({ x: 0.3, y: 0.7 }, 'LEFT', 0.9),
        makeDetection({ x: 0.7, y: 0.7 }, 'RIGHT', 0.9),
      ],
      800
    );
    expect(stability.update(hRestart, 800).passed).toBe(false);

    const hPass = tracker.update(
      [
        makeDetection({ x: 0.3, y: 0.7 }, 'LEFT', 0.9),
        makeDetection({ x: 0.7, y: 0.7 }, 'RIGHT', 0.9),
      ],
      1850
    );
    const res = stability.update(hPass, 1850);
    expect(res.passed).toBe(true);
    expect(res.stableMs).toBe(1050);
  });
});
