import * as ort from 'onnxruntime-web/wasm';
import type { FrameFeatures } from '../vision/features';
import type { HandSide } from '../vision/types';
import {
  TEMPORAL_FEATURE_DIM,
  TEMPORAL_SEQ_LEN,
  resampleHidingFeatures,
} from './temporalFeatures';

export interface TemporalFusionWeights {
  hmmWeight: number;
  temporalWeight: number;
  bias: number;
}

export interface TemporalFusionConfig {
  modelVersion: string;
  shipGatePassed: boolean;
  seqLen: number;
  featureDim: number;
  fusionWeights: TemporalFusionWeights;
}

export const DEFAULT_TEMPORAL_FUSION_CONFIG: TemporalFusionConfig = {
  modelVersion: 'hmm-fallback',
  shipGatePassed: false,
  seqLen: TEMPORAL_SEQ_LEN,
  featureDim: TEMPORAL_FEATURE_DIM,
  fusionWeights: {
    hmmWeight: 1.0,
    temporalWeight: 0.0,
    bias: 0.0,
  },
};

export interface TemporalPredictionResult {
  available: boolean;
  pLeft: number | null;
  logitLeft: number | null;
  inferenceMs: number;
  fallbackReason: string | null;
  modelVersion: string | null;
}

export interface TemporalModelStatus {
  ready: boolean;
  status: 'UNINITIALIZED' | 'READY' | 'FALLBACK';
  backend: 'wasm';
  modelVersion: string | null;
  lastInferenceMs: number | null;
  warmupMs: number | null;
  inferenceCount: number;
  fallbackReason: string | null;
  fusionConfig: TemporalFusionConfig;
}

function sigmoid(x: number): number {
  if (x >= 20) return 0.999999;
  if (x <= -20) return 0.000001;
  return 1 / (1 + Math.exp(-x));
}

function clampProb(p: number): number {
  if (!Number.isFinite(p)) return 0.5;
  return Math.max(1e-6, Math.min(1 - 1e-6, p));
}

function isBrowserEnvironment(): boolean {
  return typeof window !== 'undefined' && typeof document !== 'undefined';
}

function parseFusionConfig(raw: unknown): TemporalFusionConfig {
  if (typeof raw !== 'object' || raw === null) {
    throw new Error('Invalid fusion_config.json payload');
  }
  const obj = raw as Record<string, unknown>;
  const fw = (obj.fusionWeights ?? {}) as Record<string, unknown>;
  const hmmWeight = Number(fw.hmmWeight);
  const temporalWeight = Number(fw.temporalWeight);
  const bias = Number(fw.bias);

  return {
    modelVersion:
      typeof obj.modelVersion === 'string' && obj.modelVersion.length > 0
        ? obj.modelVersion
        : 'phase4-onnx',
    shipGatePassed: Boolean(obj.shipGatePassed ?? true),
    seqLen: Math.max(1, Math.round(Number(obj.seqLen) || TEMPORAL_SEQ_LEN)),
    featureDim: Math.max(1, Math.round(Number(obj.featureDim) || TEMPORAL_FEATURE_DIM)),
    fusionWeights: {
      hmmWeight: Number.isFinite(hmmWeight) ? hmmWeight : 0.5,
      temporalWeight: Number.isFinite(temporalWeight) ? temporalWeight : 0.5,
      bias: Number.isFinite(bias) ? bias : 0,
    },
  };
}

/**
 * Self-hosted ONNX Runtime Web (WASM backend) wrapper for the Phase 4 temporal tracker.
 * Automatically falls back to the HMM belief filter if assets are missing, corrupt,
 * fail the ship gate, or raise an inference error.
 */
export class TemporalModelRunner {
  private session: ort.InferenceSession | null = null;
  private status: 'UNINITIALIZED' | 'READY' | 'FALLBACK' = 'UNINITIALIZED';
  private fusionConfig: TemporalFusionConfig = { ...DEFAULT_TEMPORAL_FUSION_CONFIG };
  private fallbackReason: string | null = null;
  private lastInferenceMs: number | null = null;
  private warmupMs: number | null = null;
  private inferenceCount = 0;
  private initPromise: Promise<TemporalModelStatus> | null = null;

  get isReady(): boolean {
    return this.status === 'READY' && this.session !== null;
  }

  getStatus(): TemporalModelStatus {
    return {
      ready: this.isReady,
      status: this.status,
      backend: 'wasm',
      modelVersion: this.isReady ? this.fusionConfig.modelVersion : null,
      lastInferenceMs: this.lastInferenceMs,
      warmupMs: this.warmupMs,
      inferenceCount: this.inferenceCount,
      fallbackReason: this.fallbackReason,
      fusionConfig: {
        ...this.fusionConfig,
        fusionWeights: { ...this.fusionConfig.fusionWeights },
      },
    };
  }

  async init(
    baseUrl = import.meta.env?.BASE_URL ?? '/',
    options: {
      modelUrl?: string;
      configUrl?: string;
      fetchImpl?: typeof fetch;
    } = {}
  ): Promise<TemporalModelStatus> {
    if (this.status === 'READY') return this.getStatus();
    if (this.initPromise) return this.initPromise;

    this.initPromise = this.performInit(baseUrl, options);
    try {
      return await this.initPromise;
    } finally {
      this.initPromise = null;
    }
  }

  private async performInit(
    baseUrl: string,
    options: {
      modelUrl?: string;
      configUrl?: string;
      fetchImpl?: typeof fetch;
    }
  ): Promise<TemporalModelStatus> {
    const normalizedBase = baseUrl.endsWith('/') ? baseUrl : `${baseUrl}/`;
    const fetchFn = options.fetchImpl ?? globalThis.fetch;

    if (typeof fetchFn !== 'function') {
      this.markFallback('fetch API unavailable');
      return this.getStatus();
    }

    if (isBrowserEnvironment()) {
      const resolvedOrtBase = new URL(`${normalizedBase}ort/`, document.baseURI).href;
      ort.env.wasm.wasmPaths = resolvedOrtBase;
      ort.env.wasm.numThreads = 1;
      ort.env.wasm.simd = true;
    }

    const configUrl = options.configUrl ?? `${normalizedBase}models/fusion_config.json`;
    const modelUrl = options.modelUrl ?? `${normalizedBase}models/temporal_tracker.onnx`;

    try {
      const cfgResp = await fetchFn(configUrl);
      if (cfgResp.ok) {
        const cfgJson: unknown = await cfgResp.json();
        const parsedCfg = parseFusionConfig(cfgJson);
        this.fusionConfig = parsedCfg;
        if (!parsedCfg.shipGatePassed) {
          this.markFallback('Model did not pass offline ship gate');
          return this.getStatus();
        }
      }
    } catch {
      // Optional config fetch failure uses conservative default weights if model loads
    }

    try {
      const modelResp = await fetchFn(modelUrl);
      if (!modelResp.ok) {
        this.markFallback(`ONNX model HTTP ${modelResp.status}`);
        return this.getStatus();
      }
      const buffer = await modelResp.arrayBuffer();
      const bytes = new Uint8Array(buffer);
      // Guard against SPA/dev server returning an HTML 404 page with status 200
      if (bytes.byteLength > 0 && bytes[0] === 0x3c /* '<' */) {
        this.markFallback('ONNX model URL returned HTML instead of binary');
        return this.getStatus();
      }
      if (bytes.byteLength < 64) {
        this.markFallback('ONNX model file empty or truncated');
        return this.getStatus();
      }

      return await this.initFromBytes(bytes, this.fusionConfig);
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.markFallback(`ONNX load failed: ${msg}`);
      return this.getStatus();
    }
  }

  async initFromBytes(
    modelBytes: Uint8Array,
    fusionConfig?: Partial<TemporalFusionConfig>
  ): Promise<TemporalModelStatus> {
    try {
      if (fusionConfig) {
        this.fusionConfig = {
          ...this.fusionConfig,
          ...fusionConfig,
          fusionWeights: {
            ...this.fusionConfig.fusionWeights,
            ...(fusionConfig.fusionWeights ?? {}),
          },
        };
      }
      if (this.fusionConfig.modelVersion === 'hmm-fallback') {
        this.fusionConfig.modelVersion = 'phase4-cnn-v1';
        this.fusionConfig.shipGatePassed = true;
      }

      if (!isBrowserEnvironment() && !ort.env.wasm.wasmPaths) {
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.wasmPaths = import.meta.url.replace(
          /\/src\/tracking\/[^/]+$/,
          '/public/ort/'
        );
      }

      const sessionOptions: ort.InferenceSession.SessionOptions = {
        executionProviders: ['wasm'],
      };

      this.session = await ort.InferenceSession.create(modelBytes, sessionOptions);

      // Single warmup pass so live gameplay predictions stay under <10ms
      const t0 = performance.now();
      const warmupTensor = new ort.Tensor(
        'float32',
        new Float32Array(this.fusionConfig.seqLen * this.fusionConfig.featureDim),
        [1, this.fusionConfig.seqLen, this.fusionConfig.featureDim]
      );
      await this.session.run({ features: warmupTensor });
      this.warmupMs = performance.now() - t0;

      this.status = 'READY';
      this.fallbackReason = null;
      return this.getStatus();
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.markFallback(`ORT session init failed: ${msg}`);
      return this.getStatus();
    }
  }

  private markFallback(reason: string): void {
    this.session = null;
    this.status = 'FALLBACK';
    this.fallbackReason = reason;
  }

  async predictFromTensorData(
    flatFeatures: Float32Array,
    seqLen = this.fusionConfig.seqLen,
    featureDim = this.fusionConfig.featureDim
  ): Promise<TemporalPredictionResult> {
    if (!this.isReady || !this.session) {
      return {
        available: false,
        pLeft: null,
        logitLeft: null,
        inferenceMs: 0,
        fallbackReason: this.fallbackReason ?? 'Temporal model not initialized',
        modelVersion: null,
      };
    }

    if (flatFeatures.length !== seqLen * featureDim) {
      return {
        available: false,
        pLeft: null,
        logitLeft: null,
        inferenceMs: 0,
        fallbackReason: `Invalid tensor length ${flatFeatures.length}, expected ${seqLen * featureDim}`,
        modelVersion: this.fusionConfig.modelVersion,
      };
    }

    try {
      const t0 = performance.now();
      const inputTensor = new ort.Tensor('float32', flatFeatures, [1, seqLen, featureDim]);
      const outputs = await this.session.run({ features: inputTensor });
      const inferenceMs = performance.now() - t0;

      const outputName = this.session.outputNames[0] ?? 'logit_left';
      const outTensor = outputs[outputName] ?? Object.values(outputs)[0];
      const rawVal = outTensor ? Number((outTensor.data as Float32Array)[0]) : NaN;
      if (!Number.isFinite(rawVal)) {
        return {
          available: false,
          pLeft: null,
          logitLeft: null,
          inferenceMs,
          fallbackReason: 'Non-finite ONNX model output',
          modelVersion: this.fusionConfig.modelVersion,
        };
      }

      this.lastInferenceMs = inferenceMs;
      this.inferenceCount += 1;

      return {
        available: true,
        pLeft: clampProb(sigmoid(rawVal)),
        logitLeft: rawVal,
        inferenceMs,
        fallbackReason: null,
        modelVersion: this.fusionConfig.modelVersion,
      };
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      this.fallbackReason = `ONNX inference error: ${msg}`;
      return {
        available: false,
        pLeft: null,
        logitLeft: null,
        inferenceMs: 0,
        fallbackReason: this.fallbackReason,
        modelVersion: this.fusionConfig.modelVersion,
      };
    }
  }

  async predict(
    frames: readonly FrameFeatures[],
    initialSide: HandSide
  ): Promise<TemporalPredictionResult> {
    if (!this.isReady || !this.session) {
      return {
        available: false,
        pLeft: null,
        logitLeft: null,
        inferenceMs: 0,
        fallbackReason: this.fallbackReason ?? 'Temporal model not ready',
        modelVersion: null,
      };
    }

    const resampled = resampleHidingFeatures(frames, initialSide, {
      seqLen: this.fusionConfig.seqLen,
    });
    return this.predictFromTensorData(
      resampled.data,
      resampled.seqLen,
      resampled.featureDim
    );
  }
}
