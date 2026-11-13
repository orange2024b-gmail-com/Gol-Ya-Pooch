import { describe, expect, it } from 'vitest';
import type { RoundBehaviorRecord } from '../../src/ai/behavior/types';
import { HedgeEnsemble } from '../../src/ai/ensemble';
import { makeFinalGuess } from '../../src/ai/guesser';
import { BeliefFilter } from '../../src/tracking/beliefFilter';
import { FeatureExtractor } from '../../src/vision/features';
import { HandIdentityTracker } from '../../src/vision/handTracker';
import { SessionRecorder, replayRecordedSession } from '../../src/vision/recorder';
import type { HandSide, Landmark, RecordedSession } from '../../src/vision/types';
import { makeSyntheticHandLandmarks } from './fixtures';

interface SyntheticRoundSpec {
  profileId: 'profile-a' | 'profile-b' | 'profile-c';
  roundIdx: number;
  initialHand: HandSide;
  transfers: 0 | 1 | 2;
  fakeContacts: number;
  briefSightingOnFinal?: boolean;
  briefOcclusion?: boolean;
}

interface BuiltRound {
  spec: SyntheticRoundSpec;
  expectedFinalHand: HandSide;
  session: RecordedSession;
}

function interpolateLms(a: readonly Landmark[], b: readonly Landmark[], t: number): Landmark[] {
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

function makeTightenedFist(wrist: { x: number; y: number }, mirrorX: boolean, tightness: number): Landmark[] {
  const fist = makeSyntheticHandLandmarks({ wrist, posture: 'fist', mirrorX });
  const open = makeSyntheticHandLandmarks({ wrist, posture: 'open', mirrorX });

  if (tightness < 0.8) {
    // Partial opening: interpolate between open and fist
    const t = Math.max(0.35, Math.min(1.0, tightness));
    return interpolateLms(open, fist, t);
  }

  // Tight fist squeeze: contract fingers slightly tighter inward toward palm center (landmarks 0,5,9,13,17)
  const palmCx = (fist[0]!.x + fist[5]!.x + fist[9]!.x + fist[13]!.x + fist[17]!.x) / 5;
  const palmCy = (fist[0]!.y + fist[5]!.y + fist[9]!.y + fist[13]!.y + fist[17]!.y) / 5;
  // tightness in [0.80, 1.0]: 1.0 contracts inward, 0.82 relaxes outward slightly
  const radialScale = 1.28 - 0.40 * tightness; // 0.84 -> 0.944, 1.0 -> 0.88

  return fist.map((lm, idx) => {
    if (idx === 0 || idx === 1 || idx === 5 || idx === 9 || idx === 13 || idx === 17) {
      return lm;
    }
    return {
      x: palmCx + (lm.x - palmCx) * radialScale,
      y: palmCy + (lm.y - palmCy) * radialScale,
      z: lm.z,
      visibility: lm.visibility ?? 0.98,
    };
  });
}

function buildRecordedEvaluationRound(spec: SyntheticRoundSpec): BuiltRound {
  const tracker = new HandIdentityTracker();
  const extractor = new FeatureExtractor();
  const recorder = new SessionRecorder();
  recorder.start({
    source: 'synthetic',
    notes: `${spec.profileId}-r${spec.roundIdx}-t${spec.transfers}`,
  });

  const quality = {
    meanLuminance: 134,
    contrastStd: 34,
    blurScore: 105,
    lightingOk: true,
    lightingHintFa: null,
    blurOk: true,
    bothHandsDetected: true,
    minHandConfidence: 0.92,
    confidenceOk: true,
    distanceOk: true,
    distanceHintFa: null,
  };

  // Phase 1: 16 frames of SHOW_GOL pinch on initialHand
  for (let f = 0; f < 16; f++) {
    const ts = f * 33;
    const rawDetections = [
      {
        landmarks: makeSyntheticHandLandmarks({
          wrist: { x: 0.28, y: 0.7 },
          posture: spec.initialHand === 'LEFT' ? 'pinch' : 'open',
        }),
        handedness: 'LEFT' as const,
        handednessScore: 0.95,
      },
      {
        landmarks: makeSyntheticHandLandmarks({
          wrist: { x: 0.72, y: 0.7 },
          posture: spec.initialHand === 'RIGHT' ? 'pinch' : 'open',
          mirrorX: true,
        }),
        handedness: 'RIGHT' as const,
        handednessScore: 0.94,
      },
    ];
    const hands = tracker.update(rawDetections, ts);
    const features = extractor.update({ hands, quality });
    recorder.recordFrame({
      timestampMs: ts,
      rawDetections,
      hands,
      quality,
      features,
      pinch: [
        {
          side: spec.initialHand,
          pinchPostureOk: true,
          pinchDistanceNorm: 0.14,
          roiCenter: { x: spec.initialHand === 'LEFT' ? 0.26 : 0.74, y: 0.54 },
          roiRadiusNorm: 0.05,
          blob: { matchRatio: 0.45, compactness: 0.8, matchedPixels: 50, totalPixels: 110, detected: true },
          golPresentInFrame: true,
        },
      ],
    });
  }

  // Phase 2: HIDING sequence (frames 16..115, ~3.3s)
  // Build contact episodes: real transfers + optional fake contacts
  type ContactEpisode = {
    startFrame: number;
    endFrame: number;
    isRealTransfer: boolean;
    giver: HandSide;
  };
  const episodes: ContactEpisode[] = [];
  let holder: HandSide = spec.initialHand;

  const totalEvents = spec.transfers + spec.fakeContacts;
  // Distribute episodes evenly across hiding frames 22..105
  const realFlags: boolean[] = [];
  for (let i = 0; i < spec.transfers; i++) realFlags.push(true);
  for (let i = 0; i < spec.fakeContacts; i++) realFlags.push(false);

  for (let idx = 0; idx < totalEvents; idx++) {
    const startFrame = 22 + idx * 24;
    const isReal = realFlags[idx]!;
    const endFrame = startFrame + (isReal ? 17 : 8);
    episodes.push({ startFrame, endFrame, isRealTransfer: isReal, giver: holder });
    if (isReal) {
      holder = holder === 'LEFT' ? 'RIGHT' : 'LEFT';
    }
  }
  const expectedFinalHand = holder;

  let currentHolder: HandSide = spec.initialHand;
  for (let f = 16; f < 116; f++) {
    const ts = f * 33;
    const activeEp = episodes.find((e) => f >= e.startFrame && f <= e.endFrame);
    if (activeEp && f === activeEp.endFrame && activeEp.isRealTransfer) {
      currentHolder = currentHolder === 'LEFT' ? 'RIGHT' : 'LEFT';
    }

    let lx = 0.28;
    let rx = 0.72;
    let leftTight = currentHolder === 'LEFT' ? 0.99 : 0.88;
    let rightTight = currentHolder === 'RIGHT' ? 0.99 : 0.88;

    if (activeEp) {
      // Smoothstep approach & separation so hands decelerate naturally into contact
      const rampIn = Math.min(1, (f - activeEp.startFrame + 1) / 6);
      const rampOut = Math.min(1, (activeEp.endFrame - f + 1) / 6);
      const u = Math.min(rampIn, rampOut);
      const approach = u * u * (3 - 2 * u);
      lx = 0.28 + 0.19 * approach;
      rx = 0.72 - 0.19 * approach;

      if (activeEp.isRealTransfer) {
        const mid = Math.floor((activeEp.startFrame + activeEp.endFrame) / 2);
        if (f >= mid - 2 && f <= mid + 2) {
          // Giver briefly relaxes/half-opens to pass the gol
          if (activeEp.giver === 'LEFT') leftTight = 0.56;
          else rightTight = 0.56;
        }
        // Receiver tightens immediately after midpoint
        if (f > mid + 2) {
          const receiver: HandSide = activeEp.giver === 'LEFT' ? 'RIGHT' : 'LEFT';
          leftTight = receiver === 'LEFT' ? 1.0 : 0.84;
          rightTight = receiver === 'RIGHT' ? 1.0 : 0.84;
        }
      }
    }

    const occludeNow = Boolean(spec.briefOcclusion && f >= 78 && f <= 80);
    const rawDetections = [];
    if (!occludeNow) {
      rawDetections.push({
        landmarks: makeTightenedFist({ x: lx, y: 0.68 }, false, leftTight),
        handedness: 'LEFT' as const,
        handednessScore: 0.92,
      });
    }
    rawDetections.push({
      landmarks: makeTightenedFist({ x: rx, y: 0.68 }, true, rightTight),
      handedness: 'RIGHT' as const,
      handednessScore: 0.92,
    });

    const hands = tracker.update(rawDetections, ts);
    const sighting =
      spec.briefSightingOnFinal && f === 108
        ? {
            left: expectedFinalHand === 'LEFT' ? 0.85 : 0,
            right: expectedFinalHand === 'RIGHT' ? 0.85 : 0,
          }
        : { left: 0, right: 0 };

    const features = extractor.update({ hands, quality, golSightings: sighting });
    recorder.recordFrame({ timestampMs: ts, rawDetections, hands, quality, features });
  }

  // Reveal frame
  {
    const ts = 116 * 33;
    const rawDetections = [
      {
        landmarks: makeSyntheticHandLandmarks({ wrist: { x: 0.28, y: 0.7 }, posture: 'open' }),
        handedness: 'LEFT' as const,
        handednessScore: 0.94,
      },
      {
        landmarks: makeSyntheticHandLandmarks({ wrist: { x: 0.72, y: 0.7 }, posture: 'open', mirrorX: true }),
        handedness: 'RIGHT' as const,
        handednessScore: 0.94,
      },
    ];
    const hands = tracker.update(rawDetections, ts);
    const features = extractor.update({ hands, quality });
    recorder.recordFrame({
      timestampMs: ts,
      rawDetections,
      hands,
      quality,
      features,
      reveal: {
        bothOpen: true,
        outcome: expectedFinalHand,
        leftGolDetected: expectedFinalHand === 'LEFT',
        rightGolDetected: expectedFinalHand === 'RIGHT',
      },
    });
  }

  return {
    spec,
    expectedFinalHand,
    session: recorder.stop(),
  };
}

describe('Phase 2 Evaluation: 72-round synthetic multi-profile recorded dataset', () => {
  it('evaluates replay + belief filter + behavioral ensemble + guesser split by 0, 1, and 2+ transfers', () => {
    const specs: SyntheticRoundSpec[] = [];
    const profiles: Array<'profile-a' | 'profile-b' | 'profile-c'> = [
      'profile-a',
      'profile-b',
      'profile-c',
    ];
    const transferBuckets: Array<0 | 1 | 2> = [0, 1, 2];

    for (const profileId of profiles) {
      let rIdx = 0;
      for (const transfers of transferBuckets) {
        // 8 rounds per (profile, transferCount) => 3 * 3 * 8 = 72 recorded rounds
        for (let k = 0; k < 8; k++) {
          rIdx += 1;
          const initialHand: HandSide =
            profileId === 'profile-a'
              ? k % 4 === 0
                ? 'LEFT'
                : 'RIGHT'
              : (rIdx + k) % 2 === 0
                ? 'LEFT'
                : 'RIGHT';
          specs.push({
            profileId,
            roundIdx: rIdx,
            initialHand,
            transfers,
            fakeContacts: k % 2 === 1 ? 1 : 0,
            briefSightingOnFinal: k === 7,
            briefOcclusion: profileId === 'profile-c' && k % 3 === 0,
          });
        }
      }
    }

    expect(specs.length).toBe(72);

    const ensembles = {
      'profile-a': new HedgeEnsemble(),
      'profile-b': new HedgeEnsemble(),
      'profile-c': new HedgeEnsemble(),
    };
    const histories: Record<'profile-a' | 'profile-b' | 'profile-c', RoundBehaviorRecord[]> = {
      'profile-a': [],
      'profile-b': [],
      'profile-c': [],
    };

    const stats: Record<0 | 1 | 2, { total: number; correct: number }> = {
      0: { total: 0, correct: 0 },
      1: { total: 0, correct: 0 },
      2: { total: 0, correct: 0 },
    };

    for (const spec of specs) {
      const built = buildRecordedEvaluationRound(spec);
      const replay = replayRecordedSession(built.session, { requiredPinchFrames: 15 });
      expect(replay.pinchConfirmed).toBe(true);
      expect(replay.finalRevealOutcome).toBe(built.expectedFinalHand);

      const filter = new BeliefFilter();
      filter.init(replay.confirmedSide!, replay.frames[16]!.timestampMs);

      for (let f = 16; f < replay.frames.length - 1; f++) {
        filter.update(replay.frames[f]!.features);
      }

      const ensemble = ensembles[spec.profileId];
      const history = histories[spec.profileId];
      const behavPred = ensemble.predict(history);
      const guessRes = makeFinalGuess({
        belief: filter.getSnapshot(),
        observationQuality: filter.getObservationQuality(),
        behavior: behavPred,
      });

      const isCorrect = guessRes.guess === built.expectedFinalHand;
      stats[spec.transfers].total += 1;
      if (isCorrect) stats[spec.transfers].correct += 1;

      ensemble.update(history, built.expectedFinalHand);
      history.push({
        roundIndex: spec.roundIdx,
        timestampMs: spec.roundIdx * 5000,
        chosenHand: built.expectedFinalHand,
        computerGuess: guessRes.guess,
        humanWon: !isCorrect,
      });
    }

    const totalRounds = stats[0].total + stats[1].total + stats[2].total;
    const totalCorrect = stats[0].correct + stats[1].correct + stats[2].correct;
    const acc0 = stats[0].correct / stats[0].total;
    const acc1 = stats[1].correct / stats[1].total;
    const acc2 = stats[2].correct / stats[2].total;
    const accAll = totalCorrect / totalRounds;

    console.info(
      `[Phase 2 Synthetic Multi-Profile Eval] total=${totalRounds} (${totalCorrect}/${totalRounds} = ${(accAll * 100).toFixed(1)}%) | 0-transfer: ${stats[0].correct}/${stats[0].total} (${(acc0 * 100).toFixed(1)}%) | 1-transfer: ${stats[1].correct}/${stats[1].total} (${(acc1 * 100).toFixed(1)}%) | 2+-transfer: ${stats[2].correct}/${stats[2].total} (${(acc2 * 100).toFixed(1)}%)`
    );

    expect(totalRounds).toBeGreaterThanOrEqual(60);
    expect(acc0).toBeGreaterThanOrEqual(0.85);
    expect(acc1).toBeGreaterThanOrEqual(0.8);
    expect(acc2).toBeGreaterThanOrEqual(0.75);
    expect(accAll).toBeGreaterThanOrEqual(0.8);
  });
});
