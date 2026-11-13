import fc from 'fast-check';
import { describe, expect, it } from 'vitest';
import {
  chooseComputerHidingHand,
  generateHidingChoreography,
  type HumanGuessRecord,
} from '../../src/ai/hider';
import type { LevelId } from '../../src/game/levels';

describe('Computer Hider Policy & Choreography (src/ai/hider.ts)', () => {
  it('uses strictly uniform distributions for Levels 1-4', () => {
    const biasedHistory: HumanGuessRecord[] = Array.from({ length: 20 }, (_, i) => ({
      roundIndex: i + 1,
      handsCount: 2,
      guessedHandIndex: 0,
      humanFoundGol: false,
    }));

    for (const levelId of [1, 2, 3] as const) {
      const dec = chooseComputerHidingHand(levelId, 2, biasedHistory);
      expect(dec.probabilities).toEqual([0.5, 0.5]);
      expect(dec.maxDeviationFromUniform).toBe(0);
      expect(dec.detectedPattern).toBe('uniform');
    }

    const dec4 = chooseComputerHidingHand(4, 6, []);
    expect(dec4.probabilities.length).toBe(6);
    for (const p of dec4.probabilities) {
      expect(p).toBeCloseTo(1 / 6, 8);
    }
  });

  it('exploits detected human bias at Level 5 while strictly enforcing bounded exploitation', () => {
    // Strongly biased human who always guesses hand 0
    const freqBias: HumanGuessRecord[] = Array.from({ length: 25 }, (_, i) => ({
      roundIndex: i + 1,
      handsCount: 2,
      guessedHandIndex: 0,
      humanFoundGol: i % 3 === 0,
    }));

    const dec = chooseComputerHidingHand(5, 2, freqBias);
    expect(dec.probabilities[1]!).toBeGreaterThan(dec.probabilities[0]!);
    expect(dec.maxDeviationFromUniform).toBeGreaterThan(0.05);
    expect(dec.maxDeviationFromUniform).toBeLessThanOrEqual(0.18 + 1e-9);
    expect(dec.probabilities[0]!).toBeGreaterThanOrEqual(0.32 - 1e-9);
    expect(dec.probabilities[1]!).toBeLessThanOrEqual(0.68 + 1e-9);
  });

  it('fast-check invariant: Level 5 hiding probabilities always sum to 1 and remain bounded', () => {
    const historyArb = fc.array(
      fc.record({
        roundIndex: fc.integer({ min: 1, max: 200 }),
        handsCount: fc.constantFrom(2, 4, 6),
        guessedHandIndex: fc.integer({ min: 0, max: 5 }),
        humanFoundGol: fc.boolean(),
      }),
      { minLength: 0, maxLength: 60 }
    );

    fc.assert(
      fc.property(
        fc.constantFrom<LevelId>(1, 2, 3, 4, 5),
        fc.constantFrom(2, 4, 6),
        historyArb,
        fc.double({ min: 0, max: 0.999999, noNaN: true }),
        (levelId, handsCount, rawHistory, rngVal) => {
          const history = rawHistory.map((h) => ({
            ...h,
            handsCount,
            guessedHandIndex: h.guessedHandIndex % handsCount,
          }));
          const dec = chooseComputerHidingHand(levelId, handsCount, history, () => rngVal);

          expect(dec.probabilities.length).toBe(handsCount);
          const sum = dec.probabilities.reduce((a, b) => a + b, 0);
          expect(sum).toBeCloseTo(1, 7);
          expect(dec.chosenHandIndex).toBeGreaterThanOrEqual(0);
          expect(dec.chosenHandIndex).toBeLessThan(handsCount);

          const uniform = 1 / handsCount;
          const allowedMaxDev = levelId === 5 ? Math.min(0.18, 0.45 / handsCount) : 0;
          for (const p of dec.probabilities) {
            expect(p).toBeGreaterThan(0);
            expect(Math.abs(p - uniform)).toBeLessThanOrEqual(allowedMaxDev + 1e-7);
          }
        }
      ),
      { numRuns: 250 }
    );
  });

  it('generates level-appropriate shuffle/feint/misdirection choreography and honors reduced motion', () => {
    const c1 = generateHidingChoreography(1, 2, false);
    expect(c1.steps.some((s) => s.type === 'FEINT_SWAP')).toBe(false);

    const c2 = generateHidingChoreography(2, 2, false);
    expect(c2.steps.filter((s) => s.type === 'FEINT_SWAP').length).toBe(1);

    const c3 = generateHidingChoreography(3, 2, false);
    expect(c3.steps.filter((s) => s.type === 'FEINT_SWAP').length).toBeGreaterThanOrEqual(2);

    const c4 = generateHidingChoreography(4, 6, false);
    expect(
      c4.steps.filter((s) => s.type === 'MISDIRECTION_CROSS').length
    ).toBeGreaterThanOrEqual(2);

    const reduced = generateHidingChoreography(5, 4, true);
    expect(reduced.totalDurationMs).toBe(600);
    expect(reduced.steps.length).toBe(1);
    expect(reduced.steps[0]?.type).toBe('PAUSE');
  });
});
