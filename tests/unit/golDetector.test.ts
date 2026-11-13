import { describe, expect, it } from 'vitest';
import {
  PinchConfirmationTracker,
  calibrateGolColorFromRgba,
  detectGolBlobInRoi,
  detectLightingShift,
  evaluatePinchForHand,
  rgbToHsv,
  rgbToLab,
  verifyRevealHands,
} from '../../src/vision/golDetector';
import { HandIdentityTracker } from '../../src/vision/handTracker';
import {
  drawSyntheticCircle,
  makeSyntheticHandLandmarks,
  makeSyntheticRgbaFrame,
} from '../vision/fixtures';

describe('Gol color calibration, pinch detection, and open-palm reveal verification', () => {
  it('converts RGB to HSV and Lab accurately', () => {
    const [h, s, v] = rgbToHsv(220, 40, 40);
    expect(h).toBeCloseTo(0, 1);
    expect(s).toBeGreaterThan(0.75);
    expect(v).toBeGreaterThan(0.8);

    const [l, a] = rgbToLab(220, 40, 40);
    expect(l).toBeGreaterThan(40);
    expect(a).toBeGreaterThan(45);
  });

  it('calibrates gol color inside circle and detects compact blob while rejecting background', () => {
    const w = 160;
    const h = 120;
    const calFrame = makeSyntheticRgbaFrame(w, h, [140, 125, 110]);
    drawSyntheticCircle(calFrame, w, h, { x: 0.5, y: 0.5 }, 0.08, [220, 35, 45]);

    const model = calibrateGolColorFromRgba(calFrame, w, h, { x: 0.5, y: 0.5 }, 0.06, 130, 100);
    expect(model).not.toBeNull();
    expect(model!.sampleCount).toBeGreaterThan(20);

    // Positive ROI with gol bead
    const posBlob = detectGolBlobInRoi(calFrame, w, h, { x: 0.5, y: 0.5 }, 0.09, model!);
    expect(posBlob.detected).toBe(true);
    expect(posBlob.compactness).toBeGreaterThan(0.4);

    // Empty ROI with skin/background only
    const negBlob = detectGolBlobInRoi(calFrame, w, h, { x: 0.2, y: 0.2 }, 0.09, model!);
    expect(negBlob.detected).toBe(false);

    expect(detectLightingShift(190, model!)).toBe(true);
    expect(detectLightingShift(140, model!)).toBe(false);
  });

  it('confirms gol only after N=15 consecutive positive pinch frames', () => {
    const w = 160;
    const h = 120;
    const frame = makeSyntheticRgbaFrame(w, h, [140, 125, 110]);
    drawSyntheticCircle(frame, w, h, { x: 0.5, y: 0.5 }, 0.08, [25, 195, 70]);

    const model = calibrateGolColorFromRgba(frame, w, h, { x: 0.5, y: 0.5 }, 0.06, 130, 0)!;
    const tracker = new HandIdentityTracker();
    const pinchTracker = new PinchConfirmationTracker(15);

    const pinchLms = makeSyntheticHandLandmarks({ wrist: { x: 0.3, y: 0.72 }, posture: 'pinch' });
    const thumbTip = pinchLms[4]!;
    const indexTip = pinchLms[8]!;
    const pinchCenter = {
      x: (thumbTip.x + indexTip.x) / 2,
      y: (thumbTip.y + indexTip.y) / 2,
    };

    const pinchFrame = makeSyntheticRgbaFrame(w, h, [140, 125, 110]);
    drawSyntheticCircle(pinchFrame, w, h, pinchCenter, 0.035, [25, 195, 70]);

    const emptyFrame = makeSyntheticRgbaFrame(w, h, [140, 125, 110]);

    // 14 frames positive -> not confirmed yet
    for (let i = 0; i < 14; i++) {
      const hands = tracker.update(
        [
          { landmarks: pinchLms, handedness: 'LEFT', handednessScore: 0.9 },
          {
            landmarks: makeSyntheticHandLandmarks({ wrist: { x: 0.72, y: 0.72 }, posture: 'open' }),
            handedness: 'RIGHT',
            handednessScore: 0.9,
          },
        ],
        i * 33
      );
      const obsL = evaluatePinchForHand(hands.left, pinchFrame, w, h, model);
      const state = pinchTracker.update(obsL ? [obsL] : []);
      expect(state.confirmed).toBe(false);
    }

    // 1 missed frame resets count
    {
      const hands = tracker.update(
        [{ landmarks: pinchLms, handedness: 'LEFT', handednessScore: 0.9 }],
        14 * 33
      );
      const obsL = evaluatePinchForHand(hands.left, emptyFrame, w, h, model);
      const state = pinchTracker.update(obsL ? [obsL] : []);
      expect(state.consecutiveFrames).toBe(0);
    }

    // 15 consecutive positive frames -> confirmed
    let finalState = pinchTracker.update([]);
    for (let i = 15; i < 30; i++) {
      const hands = tracker.update(
        [{ landmarks: pinchLms, handedness: 'LEFT', handednessScore: 0.9 }],
        i * 33
      );
      const obsL = evaluatePinchForHand(hands.left, pinchFrame, w, h, model);
      finalState = pinchTracker.update(obsL ? [obsL] : []);
    }

    expect(finalState.confirmed).toBe(true);
    expect(finalState.confirmedSide).toBe('LEFT');
  });

  it('verifies reveal outcomes (LEFT, RIGHT, NEITHER, BOTH, HANDS_NOT_OPEN)', () => {
    const w = 160;
    const h = 120;
    const calFrame = makeSyntheticRgbaFrame(w, h, [140, 125, 110]);
    drawSyntheticCircle(calFrame, w, h, { x: 0.5, y: 0.5 }, 0.08, [220, 35, 45]);
    const model = calibrateGolColorFromRgba(calFrame, w, h, { x: 0.5, y: 0.5 }, 0.06, 130, 0)!;

    const tracker = new HandIdentityTracker();
    const leftOpenLms = makeSyntheticHandLandmarks({ wrist: { x: 0.3, y: 0.72 }, posture: 'open' });
    const rightOpenLms = makeSyntheticHandLandmarks({
      wrist: { x: 0.7, y: 0.72 },
      posture: 'open',
      mirrorX: true,
    });
    const rightFistLms = makeSyntheticHandLandmarks({
      wrist: { x: 0.7, y: 0.72 },
      posture: 'fist',
      mirrorX: true,
    });

    // Not open
    const notOpenHands = tracker.update(
      [
        { landmarks: leftOpenLms, handedness: 'LEFT', handednessScore: 0.9 },
        { landmarks: rightFistLms, handedness: 'RIGHT', handednessScore: 0.9 },
      ],
      0
    );
    expect(verifyRevealHands(calFrame, w, h, notOpenHands.left, notOpenHands.right, model).outcome).toBe(
      'HANDS_NOT_OPEN'
    );

    // Use a fresh tracker for open-palm verification so instantaneous pose switch in unit test
    // doesn't get smeared across a 33ms step by the One-Euro filter.
    const openTracker = new HandIdentityTracker();
    const openHands = openTracker.update(
      [
        { landmarks: leftOpenLms, handedness: 'LEFT', handednessScore: 0.9 },
        { landmarks: rightOpenLms, handedness: 'RIGHT', handednessScore: 0.9 },
      ],
      33
    );

    const leftPalmCenter = {
      x: (leftOpenLms[0]!.x + leftOpenLms[5]!.x + leftOpenLms[9]!.x + leftOpenLms[13]!.x + leftOpenLms[17]!.x) / 5,
      y: (leftOpenLms[0]!.y + leftOpenLms[5]!.y + leftOpenLms[9]!.y + leftOpenLms[13]!.y + leftOpenLms[17]!.y) / 5,
    };
    const rightPalmCenter = {
      x: (rightOpenLms[0]!.x + rightOpenLms[5]!.x + rightOpenLms[9]!.x + rightOpenLms[13]!.x + rightOpenLms[17]!.x) / 5,
      y: (rightOpenLms[0]!.y + rightOpenLms[5]!.y + rightOpenLms[9]!.y + rightOpenLms[13]!.y + rightOpenLms[17]!.y) / 5,
    };

    // NEITHER
    const neitherFrame = makeSyntheticRgbaFrame(w, h, [140, 125, 110]);
    expect(verifyRevealHands(neitherFrame, w, h, openHands.left, openHands.right, model).outcome).toBe(
      'NEITHER'
    );

    // LEFT
    const leftFrame = makeSyntheticRgbaFrame(w, h, [140, 125, 110]);
    drawSyntheticCircle(leftFrame, w, h, leftPalmCenter, 0.04, [220, 35, 45]);
    expect(verifyRevealHands(leftFrame, w, h, openHands.left, openHands.right, model).outcome).toBe(
      'LEFT'
    );

    // RIGHT
    const rightFrame = makeSyntheticRgbaFrame(w, h, [140, 125, 110]);
    drawSyntheticCircle(rightFrame, w, h, rightPalmCenter, 0.04, [220, 35, 45]);
    expect(verifyRevealHands(rightFrame, w, h, openHands.left, openHands.right, model).outcome).toBe(
      'RIGHT'
    );

    // BOTH
    const bothFrame = makeSyntheticRgbaFrame(w, h, [140, 125, 110]);
    drawSyntheticCircle(bothFrame, w, h, leftPalmCenter, 0.04, [220, 35, 45]);
    drawSyntheticCircle(bothFrame, w, h, rightPalmCenter, 0.04, [220, 35, 45]);
    expect(verifyRevealHands(bothFrame, w, h, openHands.left, openHands.right, model).outcome).toBe(
      'BOTH'
    );
  });
});
