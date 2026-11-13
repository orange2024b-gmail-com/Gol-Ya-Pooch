import { describe, expect, it } from 'vitest';
import type { EnsemblePrediction } from '../../src/ai/ensemble';
import { makeFinalGuess } from '../../src/ai/guesser';
import type { BeliefFilterSnapshot } from '../../src/tracking/beliefFilter';
import type { TrackingEvent } from '../../src/tracking/eventDetector';

function makeSnapshot(options: {
  pLeft: number;
  initialSide?: 'LEFT' | 'RIGHT';
  hasHadContact?: boolean;
  events?: TrackingEvent[];
}): BeliefFilterSnapshot {
  const pLeft = options.pLeft;
  const pRight = 1 - pLeft;
  const events = options.events ?? [];
  return {
    timestampMs: 5000,
    pLeft,
    pRight,
    confidence: Math.abs(pLeft - pRight),
    mostLikely: pLeft >= pRight ? 'LEFT' : 'RIGHT',
    inContact: false,
    lowQuality: false,
    initialSide: options.initialSide ?? 'LEFT',
    hasHadContact: options.hasHadContact ?? events.some((e) => e.type === 'CONTACT_START'),
    timeline: events,
    summary: {
      totalFrames: 150,
      contactCount: events.filter((e) => e.type === 'CONTACT_START').length,
      totalContactMs: 400,
      partialOpeningCount: events.filter((e) => e.type === 'PARTIAL_OPENING').length,
      sightingCount: events.filter((e) => e.type === 'GOL_SIGHTING').length,
      occludedFrames: 0,
      lowQualityFrames: 0,
      events,
    },
    trace: [],
  };
}

function makeBehavior(pLeft: number): EnsemblePrediction {
  return {
    pLeft,
    pRight: 1 - pLeft,
    confidence: Math.abs(pLeft - (1 - pLeft)),
    experts: [],
  };
}

describe('Final guesser & Persian explanations (src/ai/guesser.ts)', () => {
  it('lets tracking dominate when hiding phase is clearly observed', () => {
    const events: TrackingEvent[] = [
      { type: 'CONTACT_START', timestampMs: 3200, relativeSec: 3.2, minDistanceNorm: 0.25 },
      {
        type: 'PARTIAL_OPENING',
        timestampMs: 3500,
        relativeSec: 3.5,
        side: 'RIGHT',
        strength: 0.7,
        duringContact: true,
      },
      {
        type: 'CONTACT_END',
        timestampMs: 3900,
        relativeSec: 3.9,
        durationMs: 700,
        partialOpeningsDuringContact: ['RIGHT'],
      },
    ];

    const res = makeFinalGuess({
      belief: makeSnapshot({ pLeft: 0.12, initialSide: 'LEFT', events }),
      observationQuality: 0.96,
      behavior: makeBehavior(0.82), // behavior says LEFT, tracking strongly says RIGHT
    });

    expect(res.guess).toBe('RIGHT');
    expect(res.probability).toBeGreaterThan(0.75);
    expect(res.dominantSource).toBe('TRACKING');
    expect(res.explanationFa).toContain('۳.۲');
    expect(res.explanationFa).toContain('دست راست');
    expect(res.timelineDescriptionsFa.length).toBeGreaterThanOrEqual(2);
  });

  it('lets behavioral prior dominate when tracking confidence and quality are low', () => {
    const res = makeFinalGuess({
      belief: makeSnapshot({
        pLeft: 0.51,
        initialSide: 'LEFT',
        events: [{ type: 'OCCLUSION_START', timestampMs: 2000, relativeSec: 2.0, side: 'BOTH' }],
      }),
      observationQuality: 0.25,
      behavior: makeBehavior(0.14), // behavior strongly expects RIGHT
    });

    expect(res.guess).toBe('RIGHT');
    expect(res.dominantSource).toBe('BEHAVIOR');
    expect(res.probability).toBeGreaterThan(0.7);
    expect(res.explanationFa).toContain('الگوی قبلی');
    expect(res.explanationFa).toContain('دست راست');
  });

  it('explains gol sighting and no-contact rounds in Persian with Persian digits', () => {
    const sightingRes = makeFinalGuess({
      belief: makeSnapshot({
        pLeft: 0.94,
        events: [{ type: 'GOL_SIGHTING', timestampMs: 2400, relativeSec: 2.4, side: 'LEFT', score: 0.88 }],
      }),
      observationQuality: 1,
      behavior: makeBehavior(0.5),
    });
    expect(sightingRes.guess).toBe('LEFT');
    expect(sightingRes.explanationFa).toContain('۲.۴');
    expect(sightingRes.explanationFa).toContain('دیده شد');

    const noContactRes = makeFinalGuess({
      belief: makeSnapshot({ pLeft: 0.96, initialSide: 'LEFT', hasHadContact: false, events: [] }),
      observationQuality: 1,
      behavior: makeBehavior(0.5),
    });
    expect(noContactRes.guess).toBe('LEFT');
    expect(noContactRes.explanationFa).toContain('به هم نرسیدند');
  });

  it('respects configurable fusion weights', () => {
    const belief = makeSnapshot({ pLeft: 0.72, initialSide: 'LEFT', events: [] });
    const behavior = makeBehavior(0.15);

    const trackHeavy = makeFinalGuess({
      belief,
      observationQuality: 0.9,
      behavior,
      weights: { trackingWeight: 2.0, behaviorWeight: 0.2 },
    });
    const behavHeavy = makeFinalGuess({
      belief,
      observationQuality: 0.9,
      behavior,
      weights: { trackingWeight: 0.2, behaviorWeight: 2.0 },
    });

    expect(trackHeavy.guess).toBe('LEFT');
    expect(behavHeavy.guess).toBe('RIGHT');
  });
});
