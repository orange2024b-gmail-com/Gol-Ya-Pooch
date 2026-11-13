import { describe, expect, it } from 'vitest';
import {
  calibrateGolColorFromRgba,
  evaluatePinchForHand,
} from '../../src/vision/golDetector';
import { HandIdentityTracker } from '../../src/vision/handTracker';
import { SessionRecorder, replayRecordedSession } from '../../src/vision/recorder';
import type { HandSide } from '../../src/vision/types';
import {
  drawSyntheticCircle,
  makeSyntheticHandLandmarks,
  makeSyntheticRgbaFrame,
} from './fixtures';

describe('Vision regression: 40 synthetic/recorded pinch confirmation attempts', () => {
  it('replays 40 pinch attempts under varied colors & lighting and achieves >= 95% confirmation', () => {
    const w = 160;
    const h = 120;
    const attempts = 40;
    let confirmedCount = 0;

    const beadColors: Array<[number, number, number]> = [
      [220, 35, 45], // bright red bead
      [30, 190, 75], // bright green bead
      [235, 195, 30], // yellow bead
      [35, 120, 225], // blue bead
    ];
    const backgrounds: Array<[number, number, number]> = [
      [115, 100, 88],
      [155, 138, 122],
      [95, 85, 76],
      [180, 165, 150],
    ];

    for (let a = 0; a < attempts; a++) {
      const beadRgb = beadColors[a % beadColors.length]!;
      const bgRgb = backgrounds[Math.floor(a / 10) % backgrounds.length]!;
      const side: HandSide = a % 2 === 0 ? 'LEFT' : 'RIGHT';
      const wrist = side === 'LEFT' ? { x: 0.32, y: 0.72 } : { x: 0.68, y: 0.72 };

      const calFrame = makeSyntheticRgbaFrame(w, h, bgRgb);
      drawSyntheticCircle(calFrame, w, h, { x: 0.5, y: 0.5 }, 0.08, beadRgb, 6);
      const model = calibrateGolColorFromRgba(calFrame, w, h, { x: 0.5, y: 0.5 }, 0.06, 120, 0)!;

      const tracker = new HandIdentityTracker();
      const recorder = new SessionRecorder();
      recorder.start({ source: 'synthetic', golColorModel: model, notes: `pinch-${a + 1}` });

      const baseLms = makeSyntheticHandLandmarks({
        wrist,
        posture: 'pinch',
        mirrorX: side === 'RIGHT',
      });
      const pinchCenter = {
        x: (baseLms[4]!.x + baseLms[8]!.x) / 2,
        y: (baseLms[4]!.y + baseLms[8]!.y) / 2,
      };

      const pinchFrame = makeSyntheticRgbaFrame(w, h, bgRgb);
      drawSyntheticCircle(pinchFrame, w, h, pinchCenter, 0.036, beadRgb, 7);

      // 18 frames of steady pinch with slight natural hand micro-jitter
      for (let f = 0; f < 18; f++) {
        const jitterX = ((f % 3) - 1) * 0.002;
        const jitterY = (((f + 1) % 3) - 1) * 0.002;
        const lms = baseLms.map((lm) => ({
          ...lm,
          x: lm.x + jitterX,
          y: lm.y + jitterY,
        }));
        const rawDetections = [{ landmarks: lms, handedness: side, handednessScore: 0.91 }];
        const ts = f * 33;
        const hands = tracker.update(rawDetections, ts);
        const targetHand = side === 'LEFT' ? hands.left : hands.right;
        const obs = evaluatePinchForHand(targetHand, pinchFrame, w, h, model);

        recorder.recordFrame({
          timestampMs: ts,
          rawDetections,
          hands,
          pinch: obs ? [obs] : [],
        });
      }

      const session = recorder.stop();
      const replay = replayRecordedSession(session, { requiredPinchFrames: 15 });
      if (replay.pinchConfirmed && replay.confirmedSide === side) {
        confirmedCount += 1;
      }
    }

    const rate = confirmedCount / attempts;
    expect(confirmedCount).toBe(40);
    expect(rate).toBeGreaterThanOrEqual(0.95);
  });
});
