import type { FrameFeatures } from './features';
import { FeatureExtractor } from './features';
import { PinchConfirmationTracker } from './golDetector';
import { HandIdentityTracker } from './handTracker';
import type {
  GolColorModel,
  HandSide,
  PinchConfirmationState,
  PinchObservation,
  QualityReport,
  RawHandDetection,
  RecordedFrame,
  RecordedPinchSummary,
  RecordedRevealSummary,
  RecordedSession,
  RevealOutcome,
  VisionFrameHands,
} from './types';

export interface RecorderFrameInput {
  timestampMs: number;
  rawDetections: RawHandDetection[];
  hands: VisionFrameHands;
  quality?: QualityReport;
  features?: FrameFeatures;
  pinch?: PinchObservation[];
  reveal?: RecordedRevealSummary;
}

export class SessionRecorder {
  private recording = false;
  private frames: RecordedFrame[] = [];
  private golColorModel: GolColorModel | undefined;
  private notes: string | undefined;
  private source: 'camera' | 'synthetic' = 'camera';

  get isRecording(): boolean {
    return this.recording;
  }

  get frameCount(): number {
    return this.frames.length;
  }

  start(options: { source?: 'camera' | 'synthetic'; golColorModel?: GolColorModel; notes?: string } = {}): void {
    this.recording = true;
    this.frames = [];
    this.source = options.source ?? 'camera';
    this.golColorModel = options.golColorModel;
    this.notes = options.notes;
  }

  setGolColorModel(model: GolColorModel): void {
    this.golColorModel = model;
  }

  recordFrame(input: RecorderFrameInput): void {
    if (!this.recording) return;

    const frame: RecordedFrame = {
      timestampMs: input.timestampMs,
      rawDetections: input.rawDetections.map((d) => ({
        handedness: d.handedness,
        handednessScore: d.handednessScore,
        landmarks: d.landmarks.map((lm) => ({
          x: lm.x,
          y: lm.y,
          z: lm.z,
          ...(lm.visibility !== undefined ? { visibility: lm.visibility } : {}),
        })),
      })),
      trackedLeft: {
        detected: input.hands.left.detected,
        occluded: input.hands.left.occluded,
        lostFrames: input.hands.left.lostFrames,
        confidence: input.hands.left.confidence,
        wrist: input.hands.left.wrist ? { ...input.hands.left.wrist } : null,
      },
      trackedRight: {
        detected: input.hands.right.detected,
        occluded: input.hands.right.occluded,
        lostFrames: input.hands.right.lostFrames,
        confidence: input.hands.right.confidence,
        wrist: input.hands.right.wrist ? { ...input.hands.right.wrist } : null,
      },
      ...(input.quality
        ? {
            quality: {
              meanLuminance: input.quality.meanLuminance,
              contrastStd: input.quality.contrastStd,
              blurScore: input.quality.blurScore,
              lightingOk: input.quality.lightingOk,
              distanceOk: input.quality.distanceOk,
            },
          }
        : {}),
      ...(input.features ? { features: input.features } : {}),
      ...(input.pinch && input.pinch.length > 0
        ? {
            pinch: input.pinch.map(
              (p): RecordedPinchSummary => ({
                side: p.side,
                pinchPostureOk: p.pinchPostureOk,
                pinchDistanceNorm: p.pinchDistanceNorm,
                blobMatchRatio: p.blob.matchRatio,
                blobCompactness: p.blob.compactness,
                golPresentInFrame: p.golPresentInFrame,
              })
            ),
          }
        : {}),
      ...(input.reveal ? { reveal: { ...input.reveal } } : {}),
    };

    this.frames.push(frame);
  }

  stop(): RecordedSession {
    this.recording = false;
    return this.buildSession();
  }

  buildSession(): RecordedSession {
    return {
      schemaVersion: 1,
      createdAtIso: new Date().toISOString(),
      source: this.source,
      ...(this.notes ? { notes: this.notes } : {}),
      ...(this.golColorModel ? { golColorModel: this.golColorModel } : {}),
      frames: this.frames,
    };
  }

  exportJson(pretty = true): string {
    return JSON.stringify(this.buildSession(), null, pretty ? 2 : undefined);
  }
}

export function parseRecordedSession(rawJson: string): RecordedSession {
  const parsed = JSON.parse(rawJson) as Partial<RecordedSession>;
  if (parsed.schemaVersion !== 1 || !Array.isArray(parsed.frames)) {
    throw new Error('Invalid recording schema: expected schemaVersion 1 and frames array');
  }
  return parsed as RecordedSession;
}

export interface ReplayedFrameResult {
  frameIndex: number;
  timestampMs: number;
  hands: VisionFrameHands;
  features: FrameFeatures;
  pinchState: PinchConfirmationState;
  quality?: RecordedFrame['quality'];
  reveal?: RecordedRevealSummary;
}

export interface ReplaySummary {
  totalFrames: number;
  pinchConfirmed: boolean;
  confirmedSide: HandSide | null;
  confirmationFrameIndex: number | null;
  maxLostDurationMs: number;
  finalRevealOutcome: RevealOutcome | null;
  frames: ReplayedFrameResult[];
}

export function replayRecordedSession(
  session: RecordedSession,
  options: { requiredPinchFrames?: number } = {}
): ReplaySummary {
  const identityTracker = new HandIdentityTracker();
  const featureExtractor = new FeatureExtractor();
  const pinchTracker = new PinchConfirmationTracker(options.requiredPinchFrames);
  const frames: ReplayedFrameResult[] = [];

  let confirmationFrameIndex: number | null = null;
  let maxLostDurationMs = 0;
  let finalRevealOutcome: RevealOutcome | null = null;

  for (let i = 0; i < session.frames.length; i++) {
    const recFrame = session.frames[i]!;
    const hands = identityTracker.update(recFrame.rawDetections, recFrame.timestampMs);
    maxLostDurationMs = Math.max(maxLostDurationMs, hands.maxLostDurationMs);

    const features =
      recFrame.features ??
      featureExtractor.update({
        hands,
        ...(recFrame.quality
          ? {
              quality: {
                ...recFrame.quality,
                lightingHintFa: null,
                blurOk: recFrame.quality.blurScore >= 18,
                bothHandsDetected: hands.bothDetected,
                minHandConfidence: Math.min(hands.left.confidence, hands.right.confidence),
                confidenceOk: hands.bothDetected,
                distanceHintFa: null,
              },
            }
          : {}),
      });

    const pinchObs: PinchObservation[] = (recFrame.pinch ?? []).map((p) => ({
      side: p.side,
      pinchPostureOk: p.pinchPostureOk,
      pinchDistanceNorm: p.pinchDistanceNorm,
      roiCenter: { x: 0.5, y: 0.5 },
      roiRadiusNorm: 0.05,
      blob: {
        matchRatio: p.blobMatchRatio,
        compactness: p.blobCompactness,
        matchedPixels: 0,
        totalPixels: 0,
        detected: p.golPresentInFrame,
      },
      golPresentInFrame: p.golPresentInFrame,
    }));

    const pinchState = pinchTracker.update(pinchObs);
    if (pinchState.confirmed && confirmationFrameIndex === null) {
      confirmationFrameIndex = i;
    }

    if (recFrame.reveal) {
      finalRevealOutcome = recFrame.reveal.outcome;
    }

    frames.push({
      frameIndex: i,
      timestampMs: recFrame.timestampMs,
      hands,
      features,
      pinchState,
      ...(recFrame.quality ? { quality: recFrame.quality } : {}),
      ...(recFrame.reveal ? { reveal: recFrame.reveal } : {}),
    });
  }

  const lastPinch = frames[frames.length - 1]?.pinchState;
  return {
    totalFrames: frames.length,
    pinchConfirmed: Boolean(lastPinch?.confirmed),
    confirmedSide: lastPinch?.confirmedSide ?? null,
    confirmationFrameIndex,
    maxLostDurationMs,
    finalRevealOutcome,
    frames,
  };
}
