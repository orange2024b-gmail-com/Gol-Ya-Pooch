import type { BeliefState } from '../tracking/beliefFilter';
import type {
  Landmark,
  PinchConfirmationState,
  PinchObservation,
  QualityReport,
  RevealVerificationResult,
  VisionFrameHands,
} from '../vision/types';

const HAND_CONNECTIONS: ReadonlyArray<[number, number]> = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];

export interface OverlayRenderState {
  mirrored: boolean;
  showLandmarks: boolean;
  showDebug: boolean;
  showCalibrationCircle: boolean;
  calibrationCircle: { x: number; y: number; radiusNorm: number };
  hands: VisionFrameHands | null;
  quality: QualityReport | null;
  pinchObs: PinchObservation[];
  pinchState: PinchConfirmationState | null;
  revealResult: RevealVerificationResult | null;
  /** Strictly rendered ONLY when showDebug is true. */
  debugBelief?: BeliefState | null;
  fps: number;
  inferenceMs: number;
}

function mapX(xNorm: number, width: number, mirrored: boolean): number {
  return (mirrored ? 1 - xNorm : xNorm) * width;
}

function drawLandmarksForHand(
  ctx: CanvasRenderingContext2D,
  landmarks: readonly Landmark[],
  width: number,
  height: number,
  mirrored: boolean,
  color: string,
  labelFa: string
): void {
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = 2.5;
  ctx.lineCap = 'round';

  for (const [a, b] of HAND_CONNECTIONS) {
    const la = landmarks[a];
    const lb = landmarks[b];
    if (!la || !lb) continue;
    ctx.beginPath();
    ctx.moveTo(mapX(la.x, width, mirrored), la.y * height);
    ctx.lineTo(mapX(lb.x, width, mirrored), lb.y * height);
    ctx.stroke();
  }

  ctx.fillStyle = '#ffffff';
  for (let i = 0; i < landmarks.length; i++) {
    const lm = landmarks[i]!;
    const px = mapX(lm.x, width, mirrored);
    const py = lm.y * height;
    ctx.beginPath();
    ctx.arc(px, py, i === 4 || i === 8 ? 5 : 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  const wrist = landmarks[0];
  if (wrist) {
    const wx = mapX(wrist.x, width, mirrored);
    const wy = wrist.y * height + 22;
    ctx.font = '700 13px Vazirmatn, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillStyle = 'rgba(10, 18, 22, 0.78)';
    ctx.fillRect(wx - 34, wy - 14, 68, 20);
    ctx.fillStyle = color;
    ctx.fillText(labelFa, wx, wy);
  }
  ctx.restore();
}

export function renderVisionOverlay(
  ctx: CanvasRenderingContext2D,
  width: number,
  height: number,
  state: OverlayRenderState
): void {
  ctx.clearRect(0, 0, width, height);

  if (state.showCalibrationCircle) {
    const cx = mapX(state.calibrationCircle.x, width, state.mirrored);
    const cy = state.calibrationCircle.y * height;
    const r = state.calibrationCircle.radiusNorm * Math.min(width, height);

    ctx.save();
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 3;
    ctx.setLineDash([8, 6]);
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    ctx.setLineDash([]);
    ctx.fillStyle = 'rgba(245, 158, 11, 0.14)';
    ctx.fill();
    ctx.restore();
  }

  if (state.showLandmarks && state.hands) {
    if (state.hands.left.detected && state.hands.left.smoothedLandmarks) {
      drawLandmarksForHand(
        ctx,
        state.hands.left.smoothedLandmarks,
        width,
        height,
        state.mirrored,
        '#34d399',
        'چپ'
      );
    }
    if (state.hands.right.detected && state.hands.right.smoothedLandmarks) {
      drawLandmarksForHand(
        ctx,
        state.hands.right.smoothedLandmarks,
        width,
        height,
        state.mirrored,
        '#38bdf8',
        'راست'
      );
    }
  }

  // Live pinch confirmation ring during SHOW_GOL
  for (const obs of state.pinchObs) {
    if (!obs.pinchPostureOk && !state.showDebug) continue;
    const cx = mapX(obs.roiCenter.x, width, state.mirrored);
    const cy = obs.roiCenter.y * height;
    const r = Math.max(22, obs.roiRadiusNorm * Math.min(width, height));

    ctx.save();
    ctx.strokeStyle = obs.golPresentInFrame
      ? 'rgba(16, 185, 129, 0.45)'
      : 'rgba(255, 255, 255, 0.35)';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI * 2);
    ctx.stroke();

    if (
      state.pinchState &&
      state.pinchState.progress > 0 &&
      (state.pinchState.activeSide === obs.side || state.pinchState.confirmedSide === obs.side)
    ) {
      ctx.strokeStyle = state.pinchState.confirmed ? '#10b981' : '#f59e0b';
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.arc(
        cx,
        cy,
        r + 5,
        -Math.PI / 2,
        -Math.PI / 2 + Math.PI * 2 * state.pinchState.progress
      );
      ctx.stroke();
    }
    ctx.restore();
  }

  // Reveal verification palm ROIs
  if (state.revealResult) {
    for (const palm of [state.revealResult.left, state.revealResult.right]) {
      if (!palm) continue;
      const cx = mapX(palm.palmCenter.x, width, state.mirrored);
      const cy = palm.palmCenter.y * height;
      const r = Math.max(24, palm.palmRadiusNorm * Math.min(width, height));

      ctx.save();
      ctx.strokeStyle = palm.golDetected
        ? '#10b981'
        : palm.isOpenPalm
          ? '#38bdf8'
          : '#f97316';
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.stroke();
      ctx.restore();
    }
  }

  // On-canvas debug telemetry strip (belief is ONLY visible when showDebug === true)
  if (state.showDebug) {
    ctx.save();
    const hasBelief = Boolean(state.debugBelief);
    const boxH = hasBelief ? 124 : 98;
    ctx.fillStyle = 'rgba(8, 15, 20, 0.82)';
    ctx.fillRect(10, 10, Math.min(width - 20, 320), boxH);
    ctx.fillStyle = '#e2e8f0';
    ctx.font = '600 12px monospace';
    ctx.textAlign = 'left';
    const q = state.quality;
    const lLost = state.hands?.left.lostFrames ?? 0;
    const rLost = state.hands?.right.lostFrames ?? 0;
    ctx.fillText(`FPS: ${state.fps.toFixed(1)} | Infer: ${state.inferenceMs.toFixed(1)} ms`, 20, 30);
    ctx.fillText(
      `Luma: ${q ? q.meanLuminance.toFixed(0) : '-'} | Std: ${q ? q.contrastStd.toFixed(0) : '-'} | Blur: ${q ? q.blurScore.toFixed(0) : '-'}`,
      20,
      50
    );
    ctx.fillText(
      `L lost:${lLost} (${state.hands?.left.occluded ? 'OCC' : 'OK'}) | R lost:${rLost} (${state.hands?.right.occluded ? 'OCC' : 'OK'})`,
      20,
      70
    );
    const pRatio = state.pinchObs[0]?.blob.matchRatio ?? 0;
    const pComp = state.pinchObs[0]?.blob.compactness ?? 0;
    ctx.fillText(
      `Pinch blob ratio: ${(pRatio * 100).toFixed(0)}% | compact: ${pComp.toFixed(2)}`,
      20,
      90
    );
    if (state.debugBelief) {
      const b = state.debugBelief;
      ctx.fillStyle = '#fbbf24';
      ctx.fillText(
        `HMM P(L)=${(b.pLeft * 100).toFixed(0)}% P(R)=${(b.pRight * 100).toFixed(0)}% conf=${b.confidence.toFixed(2)}`,
        20,
        112
      );
    }
    ctx.restore();
  }
}

/**
 * Renders the live Bayesian belief trace graph inside the Debug Overlay.
 * Plots P(LEFT) (emerald) and P(RIGHT) (sky) over frames, shading contact periods.
 */
export function renderBeliefGraphCanvas(
  canvas: HTMLCanvasElement,
  trace: readonly BeliefState[]
): void {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;

  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // Background & horizontal grid lines (0%, 50%, 100%)
  ctx.fillStyle = '#071013';
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = 'rgba(255, 255, 255, 0.12)';
  ctx.lineWidth = 1;
  for (const frac of [0.25, 0.5, 0.75]) {
    const y = h * frac;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  ctx.fillStyle = 'rgba(226, 232, 240, 0.65)';
  ctx.font = '11px monospace';
  ctx.textAlign = 'left';
  ctx.fillText('100% L', 6, 13);
  ctx.fillText('50%', 6, h * 0.5 - 3);
  ctx.fillText('0% L', 6, h - 5);

  if (trace.length < 2) {
    ctx.fillStyle = 'rgba(182, 200, 201, 0.6)';
    ctx.fillText('Waiting for hiding-phase frames...', 60, h * 0.5 + 4);
    return;
  }

  const padL = 48;
  const padR = 10;
  const padT = 10;
  const padB = 10;
  const plotW = Math.max(10, w - padL - padR);
  const plotH = Math.max(10, h - padT - padB);
  const stepX = plotW / Math.max(1, trace.length - 1);

  // Shade contact episodes
  ctx.fillStyle = 'rgba(245, 185, 66, 0.16)';
  for (let i = 0; i < trace.length; i++) {
    if (trace[i]!.inContact) {
      const x = padL + i * stepX;
      ctx.fillRect(x, padT, Math.max(2, stepX + 0.5), plotH);
    }
  }

  const drawSeries = (getter: (b: BeliefState) => number, color: string) => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    for (let i = 0; i < trace.length; i++) {
      const x = padL + i * stepX;
      const y = padT + (1 - getter(trace[i]!)) * plotH;
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };

  drawSeries((b) => b.pRight, '#38bdf8');
  drawSeries((b) => b.pLeft, '#34d399');
}
