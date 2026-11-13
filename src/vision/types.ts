export type HandSide = 'LEFT' | 'RIGHT';

export interface Landmark {
  x: number;
  y: number;
  z: number;
  visibility?: number;
}

export interface NormalizedBBox {
  minX: number;
  minY: number;
  maxX: number;
  maxY: number;
  width: number;
  height: number;
  area: number;
}

export interface RawHandDetection {
  landmarks: Landmark[];
  handedness: HandSide;
  handednessScore: number;
}

export interface TrackedHand {
  side: HandSide;
  detected: boolean;
  occluded: boolean;
  lostFrames: number;
  lostDurationMs: number;
  confidence: number;
  rawLandmarks: Landmark[] | null;
  smoothedLandmarks: Landmark[] | null;
  wrist: { x: number; y: number } | null;
  velocity: { vx: number; vy: number };
  bbox: NormalizedBBox | null;
  palmSize: number;
  mediaPipeHandedness: HandSide | null;
}

export interface VisionFrameHands {
  timestampMs: number;
  left: TrackedHand;
  right: TrackedHand;
  maxLostDurationMs: number;
  bothDetected: boolean;
}

export interface QualityReport {
  meanLuminance: number;
  contrastStd: number;
  blurScore: number;
  lightingOk: boolean;
  lightingHintFa: string | null;
  blurOk: boolean;
  bothHandsDetected: boolean;
  minHandConfidence: number;
  confidenceOk: boolean;
  distanceOk: boolean;
  distanceHintFa: string | null;
}

export interface GolColorModel {
  calibratedAtMs: number;
  referenceLuminance: number;
  hsvMean: [number, number, number];
  hsvStd: [number, number, number];
  labMean: [number, number, number];
  /** Row-major 3x3 inverse covariance matrix for CIE Lab Mahalanobis distance. */
  labCovarianceInv: [
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
    number,
  ];
  sampleCount: number;
}

export interface BlobDetectionResult {
  matchRatio: number;
  compactness: number;
  matchedPixels: number;
  totalPixels: number;
  detected: boolean;
}

export interface PinchObservation {
  side: HandSide;
  pinchPostureOk: boolean;
  pinchDistanceNorm: number;
  roiCenter: { x: number; y: number };
  roiRadiusNorm: number;
  blob: BlobDetectionResult;
  golPresentInFrame: boolean;
}

export interface PinchConfirmationState {
  confirmed: boolean;
  confirmedSide: HandSide | null;
  activeSide: HandSide | null;
  consecutiveFrames: number;
  requiredFrames: number;
  progress: number;
}

export interface OpenPalmObservation {
  side: HandSide;
  fingersExtended: boolean;
  palmFacingCamera: boolean;
  isOpenPalm: boolean;
  palmCenter: { x: number; y: number };
  palmRadiusNorm: number;
  blob: BlobDetectionResult;
  golDetected: boolean;
}

export type RevealOutcome =
  | 'LEFT'
  | 'RIGHT'
  | 'NEITHER'
  | 'BOTH'
  | 'HANDS_NOT_OPEN';

export interface RevealVerificationResult {
  bothOpen: boolean;
  left: OpenPalmObservation | null;
  right: OpenPalmObservation | null;
  outcome: RevealOutcome;
}

export interface RecordedPinchSummary {
  side: HandSide;
  pinchPostureOk: boolean;
  pinchDistanceNorm: number;
  blobMatchRatio: number;
  blobCompactness: number;
  golPresentInFrame: boolean;
}

export interface RecordedRevealSummary {
  bothOpen: boolean;
  outcome: RevealOutcome;
  leftGolDetected: boolean;
  rightGolDetected: boolean;
}

export interface RecordedFrame {
  timestampMs: number;
  rawDetections: RawHandDetection[];
  trackedLeft: {
    detected: boolean;
    occluded: boolean;
    lostFrames: number;
    confidence: number;
    wrist: { x: number; y: number } | null;
  };
  trackedRight: {
    detected: boolean;
    occluded: boolean;
    lostFrames: number;
    confidence: number;
    wrist: { x: number; y: number } | null;
  };
  quality?: Pick<
    QualityReport,
    'meanLuminance' | 'contrastStd' | 'blurScore' | 'lightingOk' | 'distanceOk'
  >;
  features?: import('./features').FrameFeatures;
  pinch?: RecordedPinchSummary[];
  reveal?: RecordedRevealSummary;
}

export interface RecordedSession {
  schemaVersion: 1;
  createdAtIso: string;
  source: 'camera' | 'synthetic';
  notes?: string;
  golColorModel?: GolColorModel;
  frames: RecordedFrame[];
}
