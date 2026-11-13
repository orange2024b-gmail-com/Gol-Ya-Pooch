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

  // On-canvas debug telemetry strip
  if (state.showDebug) {
    ctx.save();
    ctx.fillStyle = 'rgba(8, 15, 20, 0.78)';
    ctx.fillRect(10, 10, Math.min(width - 20, 310), 98);
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
    ctx.restore();
  }
}
