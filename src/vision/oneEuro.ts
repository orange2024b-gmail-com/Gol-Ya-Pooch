import type { Landmark } from './types';

export interface OneEuroOptions {
  /** Minimum cutoff frequency in Hz (lower = smoother at rest). */
  minCutoff: number;
  /** Speed coefficient (higher = less lag during fast movement). */
  beta: number;
  /** Cutoff frequency for derivative in Hz. */
  dCutoff: number;
}

const DEFAULT_OPTIONS: OneEuroOptions = {
  minCutoff: 1.5,
  beta: 2.5,
  dCutoff: 1.0,
};

function alphaFromCutoff(cutoffHz: number, dtSec: number): number {
  const tau = 1 / (2 * Math.PI * cutoffHz);
  return 1 / (1 + tau / dtSec);
}

/**
 * 1D One-Euro filter (Casiez et al., CHI 2012) for low-latency landmark smoothing.
 */
export class OneEuroFilter1D {
  private readonly options: OneEuroOptions;
  private lastTimestampMs: number | null = null;
  private xPrev: number | null = null;
  private dxPrev = 0;

  constructor(options: Partial<OneEuroOptions> = {}) {
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  reset(): void {
    this.lastTimestampMs = null;
    this.xPrev = null;
    this.dxPrev = 0;
  }

  filter(value: number, timestampMs: number): number {
    if (this.lastTimestampMs === null || this.xPrev === null) {
      this.lastTimestampMs = timestampMs;
      this.xPrev = value;
      this.dxPrev = 0;
      return value;
    }

    const rawDtSec = (timestampMs - this.lastTimestampMs) / 1000;
    const dtSec = Math.min(Math.max(rawDtSec, 1 / 240), 0.25);

    const dAlpha = alphaFromCutoff(this.options.dCutoff, dtSec);
    const dx = (value - this.xPrev) / dtSec;
    const dxHat = dAlpha * dx + (1 - dAlpha) * this.dxPrev;

    const cutoff = this.options.minCutoff + this.options.beta * Math.abs(dxHat);
    const alpha = alphaFromCutoff(cutoff, dtSec);
    const xHat = alpha * value + (1 - alpha) * this.xPrev;

    this.lastTimestampMs = timestampMs;
    this.xPrev = xHat;
    this.dxPrev = dxHat;
    return xHat;
  }
}

export class LandmarkOneEuroFilter {
  private filters: Array<[OneEuroFilter1D, OneEuroFilter1D, OneEuroFilter1D]> = [];
  private readonly options: Partial<OneEuroOptions>;

  constructor(options: Partial<OneEuroOptions> = {}) {
    this.options = options;
  }

  reset(): void {
    for (const triplet of this.filters) {
      triplet[0].reset();
      triplet[1].reset();
      triplet[2].reset();
    }
  }

  filter(landmarks: readonly Landmark[], timestampMs: number): Landmark[] {
    while (this.filters.length < landmarks.length) {
      this.filters.push([
        new OneEuroFilter1D(this.options),
        new OneEuroFilter1D(this.options),
        new OneEuroFilter1D(this.options),
      ]);
    }

    return landmarks.map((lm, idx) => {
      const triplet = this.filters[idx]!;
      const out: Landmark = {
        x: triplet[0].filter(lm.x, timestampMs),
        y: triplet[1].filter(lm.y, timestampMs),
        z: triplet[2].filter(lm.z, timestampMs),
      };
      if (lm.visibility !== undefined) {
        out.visibility = lm.visibility;
      }
      return out;
    });
  }
}
