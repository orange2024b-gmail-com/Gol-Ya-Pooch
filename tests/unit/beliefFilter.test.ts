import { describe, expect, it } from 'vitest';
import { BeliefFilter } from '../../src/tracking/beliefFilter';
import { makeSyntheticFeatures } from './eventDetector.test';

describe('Bayesian HMM BeliefFilter (src/tracking/beliefFilter.ts)', () => {
  it('1. no transfer: preserves high belief in initial hand even with mild static fist asymmetry', () => {
    const filter = new BeliefFilter();
    const init = filter.init('LEFT', 0);
    expect(init.pLeft).toBeGreaterThan(0.95);
    expect(init.confidence).toBeGreaterThan(0.9);

    for (let i = 1; i <= 60; i++) {
      // Hands stay apart; RIGHT happens to look slightly tighter statically
      filter.update(
        makeSyntheticFeatures(i * 33, {
          inContact: false,
          left: { fistCompactness: 0.68, totalCurl: 0.7 },
          right: { fistCompactness: 0.74, totalCurl: 0.75 },
        })
      );
    }

    const belief = filter.getBelief();
    expect(belief.mostLikely).toBe('LEFT');
    expect(belief.pLeft).toBeGreaterThan(0.9);
    expect(belief.pLeft + belief.pRight).toBeCloseTo(1, 6);
  });

  it('2. clean transfer: contact + partial opening + post-contact compactness/curl shift moves belief to RIGHT', () => {
    const filter = new BeliefFilter();
    filter.init('LEFT', 0);

    // Frames 1..15: apart with LEFT holding
    for (let i = 1; i <= 15; i++) {
      filter.update(
        makeSyntheticFeatures(i * 33, {
          inContact: false,
          left: { fistCompactness: 0.75, totalCurl: 0.76 },
          right: { fistCompactness: 0.7, totalCurl: 0.71 },
        })
      );
    }

    // Frames 16..35: sustained contact with LEFT partial opening and RIGHT squeezing
    for (let i = 16; i <= 35; i++) {
      const dur = (i - 16) * 33;
      filter.update(
        makeSyntheticFeatures(i * 33, {
          inContact: true,
          contactDurationMs: dur,
          left: {
            partialOpening: i === 22 || i === 28,
            partialOpeningStrength: 0.75,
            fistCompactness: 0.64,
            totalCurl: 0.62,
            compactnessDelta: -0.05,
          },
          right: {
            fistCompactness: 0.79,
            totalCurl: 0.8,
            compactnessDelta: 0.06,
          },
        })
      );
    }

    // Frames 36..55: separated after transfer, RIGHT holds tighter fist
    for (let i = 36; i <= 55; i++) {
      filter.update(
        makeSyntheticFeatures(i * 33, {
          inContact: false,
          left: { fistCompactness: 0.66, totalCurl: 0.66 },
          right: { fistCompactness: 0.78, totalCurl: 0.79 },
        })
      );
    }

    const snapshot = filter.getSnapshot();
    expect(snapshot.mostLikely).toBe('RIGHT');
    expect(snapshot.pRight).toBeGreaterThan(0.75);
    expect(snapshot.hasHadContact).toBe(true);
    expect(snapshot.timeline.some((e) => e.type === 'CONTACT_START')).toBe(true);
    expect(snapshot.timeline.some((e) => e.type === 'PARTIAL_OPENING')).toBe(true);
  });

  it('3. fake contact without transfer: brief touch without partial opening keeps belief on LEFT', () => {
    const filter = new BeliefFilter();
    filter.init('LEFT', 0);

    for (let i = 1; i <= 15; i++) {
      filter.update(
        makeSyntheticFeatures(i * 33, {
          inContact: false,
          left: { fistCompactness: 0.76, totalCurl: 0.76 },
          right: { fistCompactness: 0.72, totalCurl: 0.72 },
        })
      );
    }

    // Short 260ms touch with NO partial opening and LEFT remaining tighter
    for (let i = 16; i <= 23; i++) {
      filter.update(
        makeSyntheticFeatures(i * 33, {
          inContact: true,
          contactDurationMs: (i - 16) * 33,
          left: { fistCompactness: 0.76, totalCurl: 0.76 },
          right: { fistCompactness: 0.72, totalCurl: 0.72 },
        })
      );
    }

    for (let i = 24; i <= 45; i++) {
      filter.update(
        makeSyntheticFeatures(i * 33, {
          inContact: false,
          left: { fistCompactness: 0.76, totalCurl: 0.76 },
          right: { fistCompactness: 0.72, totalCurl: 0.72 },
        })
      );
    }

    const belief = filter.getBelief();
    expect(belief.mostLikely).toBe('LEFT');
    expect(belief.pLeft).toBeGreaterThan(0.8);
  });

  it('4. occlusion & gol sighting: flattens weak likelihoods on occluded frames and snaps on sighting', () => {
    const filter = new BeliefFilter();
    filter.init('LEFT', 0);

    // Occluded while apart -> belief stays nearly unchanged
    for (let i = 1; i <= 10; i++) {
      filter.update(
        makeSyntheticFeatures(i * 33, {
          inContact: false,
          anyOccluded: true,
          // Even with bogus right compactness, occluded frame flattens likelihood
          right: { fistCompactness: 0.99, totalCurl: 0.99 },
          left: { fistCompactness: 0.1, totalCurl: 0.1 },
        })
      );
    }
    expect(filter.getBelief().pLeft).toBeGreaterThan(0.95);

    // Occluded during contact -> uncertainty increases (confidence drops)
    const preContactConf = filter.getBelief().confidence;
    for (let i = 11; i <= 35; i++) {
      filter.update(
        makeSyntheticFeatures(i * 33, {
          inContact: true,
          contactDurationMs: (i - 11) * 33,
          anyOccluded: true,
        })
      );
    }
    expect(filter.getBelief().confidence).toBeLessThan(preContactConf);
    expect(filter.getObservationQuality()).toBeLessThan(0.2);

    // Brief gol sighting in RIGHT strongly updates belief to RIGHT
    filter.update(
      makeSyntheticFeatures(36 * 33, {
        inContact: false,
        right: { golSightingScore: 0.9 },
      })
    );
    expect(filter.getBelief().mostLikely).toBe('RIGHT');
    expect(filter.getBelief().pRight).toBeGreaterThan(0.7);
  });
});
