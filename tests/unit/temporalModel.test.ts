import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { describe, expect, it } from 'vitest';
import type { EnsemblePrediction } from '../../src/ai/ensemble';
import { fuseHmmAndTemporal, makeFinalGuess } from '../../src/ai/guesser';
import type { BeliefFilterSnapshot } from '../../src/tracking/beliefFilter';
import {
  TEMPORAL_FEATURE_DIM,
  TEMPORAL_SEQ_LEN,
  resampleHidingFeatures,
} from '../../src/tracking/temporalFeatures';
import { TemporalModelRunner } from '../../src/tracking/temporalModel';
import type { FrameFeatures } from '../../src/vision/features';

function makeDummyFrameFeatures(
  timestampMs: number,
  overrides: Partial<{
    inContact: boolean;
    leftPartial: boolean;
    rightPartial: boolean;
    leftCurl: number;
    rightCurl: number;
  }> = {}
): FrameFeatures {
  const mkHand = (side: 'LEFT' | 'RIGHT', curl: number, partial: boolean) => ({
    side,
    detected: true,
    occluded: false,
    confidence: 0.94,
    palmSize: 0.14,
    fingerAngles: {
      thumb: { mcp: 0.8, pip: 0.8, dip: 0.4, curl },
      index: { mcp: 0.9, pip: 0.9, dip: 0.5, curl },
      middle: { mcp: 0.9, pip: 0.9, dip: 0.5, curl },
      ring: { mcp: 0.9, pip: 0.9, dip: 0.5, curl },
      pinky: { mcp: 0.9, pip: 0.9, dip: 0.5, curl },
    },
    totalCurl: curl,
    convexHullAreaNorm: 0.85,
    meanFingertipDistNorm: 0.48,
    fistCompactness: 0.78,
    compactnessDelta: 0.01,
    convexHullAreaDelta: 0,
    meanFingertipDistDelta: 0,
    wristVelocityNorm: { vx: 0.1, vy: 0, speed: 0.1 },
    wristAccelerationNorm: { ax: 0, ay: 0, magnitude: 0 },
    jitter: 0.08,
    palmNormal: { nx: 0, ny: 0, nz: 1 },
    palmFacingScore: 0.9,
    depthProxy: { wristZ: 0, meanZ: 0, bboxSize: 0.2 },
    partialOpening: partial,
    partialOpeningStrength: partial ? 0.72 : 0,
    golSightingScore: 0,
  });

  return {
    timestampMs,
    left: mkHand('LEFT', overrides.leftCurl ?? 0.84, Boolean(overrides.leftPartial)),
    right: mkHand('RIGHT', overrides.rightCurl ?? 0.82, Boolean(overrides.rightPartial)),
    interHand: {
      bothDetected: true,
      minLandmarkDistance: overrides.inContact ? 0.04 : 0.3,
      minLandmarkDistanceNorm: overrides.inContact ? 0.28 : 2.1,
      inContact: Boolean(overrides.inContact),
      contactDurationMs: overrides.inContact ? 350 : 0,
      relativeVelocityNorm: { vx: 0, vy: 0, speed: 0 },
    },
    quality: {
      blurScore: 110,
      blurOk: true,
      meanLuminance: 132,
      lightingOk: true,
      minConfidence: 0.92,
      leftConfidence: 0.94,
      rightConfidence: 0.94,
      leftOccluded: false,
      rightOccluded: false,
      anyOccluded: false,
      lowQuality: false,
    },
  };
}

describe('Temporal feature resampling, ORT-Web ONNX inference, parity, and fallback', () => {
  it('resamples variable-rate frames to 10 Hz, zero-pads up to 64, and preserves discrete spikes', () => {
    const frames: FrameFeatures[] = [];
    for (let i = 0; i < 30; i++) {
      frames.push(
        makeDummyFrameFeatures(i * 33.33, {
          inContact: i >= 10 && i <= 16,
          leftPartial: i === 13,
        })
      );
    }

    const res = resampleHidingFeatures(frames, 'LEFT');
    expect(res.seqLen).toBe(TEMPORAL_SEQ_LEN);
    expect(res.featureDim).toBe(TEMPORAL_FEATURE_DIM);
    expect(res.data.length).toBe(TEMPORAL_SEQ_LEN * TEMPORAL_FEATURE_DIM);
    expect(res.resampledSteps).toBe(10); // ~967ms at 10Hz => 10 bins

    // First row has initialSideSign = +1 (LEFT)
    expect(res.matrix[0]![0]).toBe(1);
    // Discrete partialOpening spike in bin ~4 must be preserved via max-pooling
    const maxPartial = Math.max(...res.matrix.slice(0, res.resampledSteps).map((r) => r[15]!));
    expect(maxPartial).toBeCloseTo(0.72, 2);

    // Padded rows after resampledSteps must be all zeros
    const paddedRow = res.matrix[res.resampledSteps]!;
    expect(paddedRow.every((v) => v === 0)).toBe(true);

    // Long > 6.4s sequence truncates cleanly to most recent 64 bins
    const longFrames: FrameFeatures[] = [];
    for (let i = 0; i < 100; i++) {
      longFrames.push(makeDummyFrameFeatures(i * 100));
    }
    const longRes = resampleHidingFeatures(longFrames, 'RIGHT');
    expect(longRes.resampledSteps).toBe(TEMPORAL_SEQ_LEN);
    expect(longRes.matrix[0]![0]).toBe(-1);
  });

  it('runs real ONNX Runtime inference on shipped model matching Python ORT parity within 1e-4 and <10ms latency', async () => {
    const modelBytes = new Uint8Array(
      readFileSync(join(process.cwd(), 'public', 'models', 'temporal_tracker.onnx'))
    );
    const configJson = JSON.parse(
      readFileSync(join(process.cwd(), 'public', 'models', 'fusion_config.json'), 'utf8')
    ) as {
      modelVersion: string;
      shipGatePassed: boolean;
      seqLen: number;
      featureDim: number;
      fusionWeights: { hmmWeight: number; temporalWeight: number; bias: number };
    };
    const parityFixture = JSON.parse(
      readFileSync(
        join(process.cwd(), 'tests', 'unit', 'fixtures', 'temporal_parity.json'),
        'utf8'
      )
    ) as {
      seqLen: number;
      featureDim: number;
      cases: Array<{
        roundId: string;
        expectedLogitLeft: number;
        expectedProbLeft: number;
        features: number[][];
      }>;
    };

    expect(configJson.shipGatePassed).toBe(true);

    const runner = new TemporalModelRunner();
    const status = await runner.initFromBytes(modelBytes, configJson);
    expect(status.ready).toBe(true);
    expect(status.status).toBe('READY');

    const warmLatencies: number[] = [];
    for (const c of parityFixture.cases) {
      const flat = new Float32Array(c.features.flat());
      const pred = await runner.predictFromTensorData(
        flat,
        parityFixture.seqLen,
        parityFixture.featureDim
      );
      expect(pred.available).toBe(true);
      expect(pred.logitLeft).not.toBeNull();
      expect(pred.pLeft).not.toBeNull();
      expect(Math.abs(pred.logitLeft! - c.expectedLogitLeft)).toBeLessThanOrEqual(1e-4);
      expect(Math.abs(pred.pLeft! - c.expectedProbLeft)).toBeLessThanOrEqual(1e-4);
      warmLatencies.push(pred.inferenceMs);
    }

    const avgMs = warmLatencies.reduce((a, b) => a + b, 0) / warmLatencies.length;
    expect(avgMs).toBeLessThan(10);
  });

  it('falls back gracefully when ONNX model is missing, returns HTML, is corrupt, or fails ship gate', async () => {
    // 1. HTTP 404
    const runner404 = new TemporalModelRunner();
    const fetch404 = (() =>
      Promise.resolve(new Response('Not found', { status: 404 }))) as unknown as typeof fetch;
    const status404 = await runner404.init('/', { fetchImpl: fetch404 });
    expect(status404.ready).toBe(false);
    expect(status404.status).toBe('FALLBACK');
    const pred404 = await runner404.predict([makeDummyFrameFeatures(0)], 'LEFT');
    expect(pred404.available).toBe(false);
    expect(pred404.pLeft).toBeNull();

    // 2. HTML 200 fallback page
    const runnerHtml = new TemporalModelRunner();
    const fetchHtml = ((url: string) => {
      if (url.endsWith('.json')) {
        return Promise.resolve(
          new Response(JSON.stringify({ shipGatePassed: true }), { status: 200 })
        );
      }
      return Promise.resolve(
        new Response('<!doctype html><html><body>Not Found</body></html>', { status: 200 })
      );
    }) as unknown as typeof fetch;
    const statusHtml = await runnerHtml.init('/', { fetchImpl: fetchHtml });
    expect(statusHtml.ready).toBe(false);
    expect(statusHtml.fallbackReason).toContain('HTML');

    // 3. Corrupt binary model
    const runnerCorrupt = new TemporalModelRunner();
    const corruptStatus = await runnerCorrupt.initFromBytes(new Uint8Array(128).fill(0xff));
    expect(corruptStatus.ready).toBe(false);
    expect(corruptStatus.status).toBe('FALLBACK');

    // 4. Ship gate failed
    const runnerGate = new TemporalModelRunner();
    const fetchGate = (() =>
      Promise.resolve(
        new Response(JSON.stringify({ shipGatePassed: false }), { status: 200 })
      )) as unknown as typeof fetch;
    const gateStatus = await runnerGate.init('/', { fetchImpl: fetchGate });
    expect(gateStatus.ready).toBe(false);
    expect(gateStatus.fallbackReason).toContain('ship gate');
  });

  it('fuses HMM belief and temporal ONNX prediction via calibrated logistic combiner and falls back cleanly', () => {
    const fused = fuseHmmAndTemporal(
      0.55,
      {
        available: true,
        pLeft: 0.08,
        logitLeft: -2.44,
        inferenceMs: 0.5,
        fallbackReason: null,
        modelVersion: 'phase4-cnn-v1',
      },
      { hmmWeight: 0.3, temporalWeight: 0.7, bias: 0 }
    );
    expect(fused.temporalUsed).toBe(true);
    expect(fused.trackingPLeft).toBeLessThan(0.25);

    const fallback = fuseHmmAndTemporal(0.77, null);
    expect(fallback.temporalUsed).toBe(false);
    expect(fallback.trackingPLeft).toBeCloseTo(0.77, 5);

    const beliefSnap: BeliefFilterSnapshot = {
      timestampMs: 3000,
      pLeft: 0.52,
      pRight: 0.48,
      confidence: 0.04,
      mostLikely: 'LEFT',
      inContact: false,
      lowQuality: false,
      initialSide: 'LEFT',
      hasHadContact: true,
      timeline: [],
      summary: {
        totalFrames: 90,
        contactCount: 1,
        totalContactMs: 400,
        partialOpeningCount: 0,
        sightingCount: 0,
        occludedFrames: 0,
        lowQualityFrames: 0,
        events: [],
      },
      trace: [],
    };
    const behavior: EnsemblePrediction = {
      pLeft: 0.5,
      pRight: 0.5,
      confidence: 0,
      experts: [],
    };

    const guessWithOnnx = makeFinalGuess({
      belief: beliefSnap,
      observationQuality: 0.95,
      behavior,
      temporal: {
        available: true,
        pLeft: 0.05,
        logitLeft: -2.94,
        inferenceMs: 0.45,
        fallbackReason: null,
        modelVersion: 'phase4-cnn-v1',
      },
    });
    expect(guessWithOnnx.temporalUsed).toBe(true);
    expect(guessWithOnnx.guess).toBe('RIGHT');

    const guessWithoutOnnx = makeFinalGuess({
      belief: beliefSnap,
      observationQuality: 0.95,
      behavior,
      temporal: {
        available: false,
        pLeft: null,
        logitLeft: null,
        inferenceMs: 0,
        fallbackReason: 'missing',
        modelVersion: null,
      },
    });
    expect(guessWithoutOnnx.temporalUsed).toBe(false);
    expect(guessWithoutOnnx.guess).toBe('LEFT');
  });
});
