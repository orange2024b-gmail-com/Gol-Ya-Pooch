import { describe, expect, it } from 'vitest';
import { SessionRecorder, parseRecordedSession, replayRecordedSession } from '../../src/vision/recorder';
import { HandIdentityTracker } from '../../src/vision/handTracker';
import { makeDetection } from './fixtures';

interface CrossingScenario {
  name: string;
  totalFrames: number;
  swapMediaPipeLabelsAtCross: boolean;
  occludeLeftFrames?: [number, number];
  verticalOffset: number;
  doubleCross?: boolean;
}

function buildCrossingSession(scenario: CrossingScenario) {
  const tracker = new HandIdentityTracker();
  const recorder = new SessionRecorder();
  recorder.start({ source: 'synthetic', notes: scenario.name });

  const expectedLeftX: number[] = [];
  const expectedRightX: number[] = [];

  for (let f = 0; f < scenario.totalFrames; f++) {
    const t = f / (scenario.totalFrames - 1);
    const phase = scenario.doubleCross ? Math.sin(t * Math.PI) : t;
    // Left starts at x=0.25 and crosses to x=0.75 (or returns if doubleCross)
    const lx = 0.25 + 0.5 * phase;
    const rx = 0.75 - 0.5 * phase;
    const ly = 0.65 - scenario.verticalOffset * Math.sin(t * Math.PI);
    const ry = 0.65 + scenario.verticalOffset * Math.sin(t * Math.PI);

    expectedLeftX.push(lx);
    expectedRightX.push(rx);

    const crossed = lx > rx;
    const leftLabel = scenario.swapMediaPipeLabelsAtCross && crossed ? 'RIGHT' : 'LEFT';
    const rightLabel = scenario.swapMediaPipeLabelsAtCross && crossed ? 'LEFT' : 'RIGHT';

    const leftOccluded =
      scenario.occludeLeftFrames !== undefined &&
      f >= scenario.occludeLeftFrames[0] &&
      f <= scenario.occludeLeftFrames[1];

    const rawDetections = [];
    if (!leftOccluded) {
      rawDetections.push(makeDetection({ x: lx, y: ly }, leftLabel, 0.88));
    }
    rawDetections.push(makeDetection({ x: rx, y: ry }, rightLabel, 0.88));

    const ts = f * 33;
    const hands = tracker.update(rawDetections, ts);
    recorder.recordFrame({ timestampMs: ts, rawDetections, hands });
  }

  const json = recorder.exportJson(false);
  return { session: parseRecordedSession(json), expectedLeftX, expectedRightX };
}

describe('Vision regression: 20 recorded hand-crossing sessions', () => {
  const scenarios: CrossingScenario[] = Array.from({ length: 20 }, (_, idx) => ({
    name: `crossing-session-${idx + 1}`,
    totalFrames: 45 + (idx % 4) * 10,
    swapMediaPipeLabelsAtCross: true,
    verticalOffset: 0.03 + (idx % 5) * 0.025,
    doubleCross: idx % 3 === 0,
    ...(idx % 2 === 1 ? { occludeLeftFrames: [20, 22] as [number, number] } : {}),
  }));

  it.each(scenarios)(
    'maintains zero left/right identity swaps in $name',
    (scenario) => {
      const { session, expectedLeftX, expectedRightX } = buildCrossingSession(scenario);
      const replay = replayRecordedSession(session);

      expect(replay.totalFrames).toBe(scenario.totalFrames);

      const swaps: Array<{ frame: number; side: string; wristX: number; expL: number; expR: number }> = [];
      for (let i = 0; i < replay.frames.length; i++) {
        const frame = replay.frames[i]!;
        if (frame.hands.left.detected && frame.hands.left.rawLandmarks?.[0]) {
          const rx = frame.hands.left.rawLandmarks[0].x;
          const distToTrueLeft = Math.abs(rx - expectedLeftX[i]!);
          const distToTrueRight = Math.abs(rx - expectedRightX[i]!);
          if (distToTrueLeft > distToTrueRight + 0.02) {
            swaps.push({
              frame: i,
              side: 'LEFT',
              wristX: rx,
              expL: expectedLeftX[i]!,
              expR: expectedRightX[i]!,
            });
          }
        }
        if (frame.hands.right.detected && frame.hands.right.rawLandmarks?.[0]) {
          const rx = frame.hands.right.rawLandmarks[0].x;
          const distToTrueRight = Math.abs(rx - expectedRightX[i]!);
          const distToTrueLeft = Math.abs(rx - expectedLeftX[i]!);
          if (distToTrueRight > distToTrueLeft + 0.02) {
            swaps.push({
              frame: i,
              side: 'RIGHT',
              wristX: rx,
              expL: expectedLeftX[i]!,
              expR: expectedRightX[i]!,
            });
          }
        }
      }

      expect(swaps).toEqual([]);
    }
  );
});
