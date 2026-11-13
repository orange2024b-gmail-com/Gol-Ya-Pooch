import { BeliefFilter } from '../tracking/beliefFilter';
import { TemporalModelRunner } from '../tracking/temporalModel';
import { FeatureExtractor, type FrameFeatures } from '../vision/features';
import { replayRecordedSession } from '../vision/recorder';
import { createPhase2TransferSampleSession } from '../vision/sampleSession';
import type { QualityReport } from '../vision/types';

export interface TimingStats {
  runs: number;
  minMs: number;
  meanMs: number;
  p95Ms: number;
  maxMs: number;
}

export interface DeviceBenchmarkReport {
  timestampIso: string;
  userAgent: string;
  hardwareConcurrency: number;
  rafFrameLoop: {
    framesMeasured: number;
    meanFrameMs: number;
    p95FrameMs: number;
    estimatedFps: number;
  };
  featureExtraction: TimingStats;
  hmmFilterStep: TimingStats;
  onnxInference: TimingStats & {
    ready: boolean;
    backend: string;
    modelVersion: string | null;
    fallbackReason: string | null;
    within10msBudget: boolean;
  };
  estimatedPipelineFrameMs: number;
  meets24FpsBudget: boolean;
  meets30FpsBudget: boolean;
}

function summarizeTimings(samples: readonly number[]): TimingStats {
  if (samples.length === 0) {
    return { runs: 0, minMs: 0, meanMs: 0, p95Ms: 0, maxMs: 0 };
  }
  const sorted = [...samples].sort((a, b) => a - b);
  const sum = sorted.reduce((acc, v) => acc + v, 0);
  const p95Idx = Math.min(sorted.length - 1, Math.floor(sorted.length * 0.95));
  return {
    runs: sorted.length,
    minMs: sorted[0]!,
    meanMs: sum / sorted.length,
    p95Ms: sorted[p95Idx]!,
    maxMs: sorted[sorted.length - 1]!,
  };
}

async function measureRafLoop(framesCount: number): Promise<DeviceBenchmarkReport['rafFrameLoop']> {
  if (typeof globalThis.requestAnimationFrame !== 'function') {
    return {
      framesMeasured: 0,
      meanFrameMs: 0,
      p95FrameMs: 0,
      estimatedFps: 0,
    };
  }

  const deltas: number[] = [];
  await new Promise<void>((resolve) => {
    let lastTs = 0;
    let remaining = framesCount + 1;
    const step = (ts: number) => {
      if (lastTs > 0) {
        deltas.push(Math.max(1, ts - lastTs));
      }
      lastTs = ts;
      remaining -= 1;
      if (remaining <= 0) {
        resolve();
        return;
      }
      globalThis.requestAnimationFrame(step);
    };
    globalThis.requestAnimationFrame(step);
  });

  const stats = summarizeTimings(deltas);
  const estimatedFps = stats.meanMs > 0 ? 1000 / stats.meanMs : 0;
  return {
    framesMeasured: deltas.length,
    meanFrameMs: stats.meanMs,
    p95FrameMs: stats.p95Ms,
    estimatedFps,
  };
}

export async function runDeviceBenchmark(options: {
  temporalModel?: TemporalModelRunner;
  rafFrames?: number;
  onnxRuns?: number;
} = {}): Promise<DeviceBenchmarkReport> {
  const rafFrames = options.rafFrames ?? 24;
  const onnxRuns = options.onnxRuns ?? 20;

  const session = createPhase2TransferSampleSession();
  const replayed = replayRecordedSession(session, { requiredPinchFrames: 15 });

  // 1. Feature Extraction benchmark
  const extractor = new FeatureExtractor();
  const featTimes: number[] = [];
  const extractedFrames: FrameFeatures[] = [];
  for (let pass = 0; pass < 2; pass++) {
    extractor.reset();
    for (let idx = 0; idx < replayed.frames.length; idx++) {
      const f = replayed.frames[idx]!;
      const rq = session.frames[idx]?.quality;
      const q: QualityReport = {
        meanLuminance: rq?.meanLuminance ?? f.features.quality.meanLuminance,
        contrastStd: rq?.contrastStd ?? 34,
        blurScore: rq?.blurScore ?? f.features.quality.blurScore,
        lightingOk: rq?.lightingOk ?? f.features.quality.lightingOk,
        lightingHintFa: null,
        blurOk: f.features.quality.blurOk,
        bothHandsDetected: f.hands.bothDetected,
        minHandConfidence: f.features.quality.minConfidence,
        confidenceOk: true,
        distanceOk: rq?.distanceOk ?? true,
        distanceHintFa: null,
      };
      const t0 = performance.now();
      const out = extractor.update({
        hands: f.hands,
        quality: q,
        golSightings: {
          left: f.features.left.golSightingScore,
          right: f.features.right.golSightingScore,
        },
      });
      featTimes.push(performance.now() - t0);
      if (pass === 0) extractedFrames.push(out);
    }
  }
  const featureStats = summarizeTimings(featTimes);

  // 2. HMM Belief Filter step benchmark
  const hidingFeatures = extractedFrames.slice(18, -1);
  const initialSide = replayed.confirmedSide ?? 'LEFT';
  const hmmTimes: number[] = [];
  const filter = new BeliefFilter();
  for (let pass = 0; pass < 3; pass++) {
    filter.init(initialSide, hidingFeatures[0]?.timestampMs ?? 0);
    for (const f of hidingFeatures) {
      const t0 = performance.now();
      filter.update(f);
      hmmTimes.push(performance.now() - t0);
    }
  }
  const hmmStats = summarizeTimings(hmmTimes);

  // 3. ONNX Runtime Web WASM inference benchmark
  const runner = options.temporalModel ?? new TemporalModelRunner();
  await runner.init();
  const ortStatus = runner.getStatus();
  const onnxTimes: number[] = [];
  // Warmup inference
  await runner.predict(hidingFeatures, initialSide);
  for (let i = 0; i < onnxRuns; i++) {
    const t0 = performance.now();
    await runner.predict(hidingFeatures, initialSide);
    onnxTimes.push(performance.now() - t0);
  }
  const onnxStats = summarizeTimings(onnxTimes);

  // 4. Browser RAF compositor loop measurement
  const rafFrameLoop = await measureRafLoop(rafFrames);

  const perFrameCoreMs = featureStats.meanMs + hmmStats.meanMs;
  // ONNX runs once at end of hiding phase, not every camera frame; report both per-frame & total
  const estimatedPipelineFrameMs = perFrameCoreMs + onnxStats.meanMs / 30;

  return {
    timestampIso: new Date().toISOString(),
    userAgent: typeof navigator !== 'undefined' ? navigator.userAgent : 'node',
    hardwareConcurrency:
      typeof navigator !== 'undefined' && typeof navigator.hardwareConcurrency === 'number'
        ? navigator.hardwareConcurrency
        : 1,
    rafFrameLoop,
    featureExtraction: featureStats,
    hmmFilterStep: hmmStats,
    onnxInference: {
      ...onnxStats,
      ready: ortStatus.ready,
      backend: ortStatus.backend,
      modelVersion: ortStatus.modelVersion,
      fallbackReason: ortStatus.fallbackReason,
      within10msBudget: onnxStats.meanMs < 10,
    },
    estimatedPipelineFrameMs,
    meets24FpsBudget: estimatedPipelineFrameMs < 41.67,
    meets30FpsBudget: estimatedPipelineFrameMs < 33.33,
  };
}
