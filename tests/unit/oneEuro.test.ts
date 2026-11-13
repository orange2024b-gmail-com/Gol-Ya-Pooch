import { describe, expect, it } from 'vitest';
import { LandmarkOneEuroFilter, OneEuroFilter1D } from '../../src/vision/oneEuro';

describe('OneEuroFilter1D & LandmarkOneEuroFilter', () => {
  it('smooths high-frequency jitter while tracking a step signal', () => {
    const filter = new OneEuroFilter1D({ minCutoff: 1.2, beta: 0.05, dCutoff: 1.0 });
    const noisyStatic: number[] = [];
    const filteredStatic: number[] = [];

    for (let i = 0; i < 30; i++) {
      const raw = 0.5 + (i % 2 === 0 ? 0.02 : -0.02);
      noisyStatic.push(raw);
      filteredStatic.push(filter.filter(raw, i * 33));
    }

    const rawRange = Math.max(...noisyStatic) - Math.min(...noisyStatic);
    const filteredTail = filteredStatic.slice(10);
    const filteredRange = Math.max(...filteredTail) - Math.min(...filteredTail);
    expect(filteredRange).toBeLessThan(rawRange * 0.65);

    // Step movement to 0.85 should converge quickly
    let out = 0.5;
    for (let i = 30; i < 45; i++) {
      out = filter.filter(0.85, i * 33);
    }
    expect(out).toBeGreaterThan(0.8);
  });

  it('resets cleanly and preserves landmark visibility', () => {
    const lmFilter = new LandmarkOneEuroFilter();
    const res1 = lmFilter.filter([{ x: 0.2, y: 0.3, z: -0.05, visibility: 0.95 }], 0);
    expect(res1[0]).toEqual({ x: 0.2, y: 0.3, z: -0.05, visibility: 0.95 });

    lmFilter.reset();
    const res2 = lmFilter.filter([{ x: 0.7, y: 0.8, z: 0.01 }], 100);
    expect(res2[0]).toEqual({ x: 0.7, y: 0.8, z: 0.01 });
  });
});
