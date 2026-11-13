import type { HandSide, Landmark, RawHandDetection } from '../../src/vision/types';

/**
 * Generates a realistic 21-landmark hand pose anchored at wrist (wx, wy).
 * Supports open palm or closed fist / pinch posture.
 */
export function makeSyntheticHandLandmarks(options: {
  wrist: { x: number; y: number };
  scale?: number;
  posture?: 'open' | 'fist' | 'pinch';
  mirrorX?: boolean;
}): Landmark[] {
  const { wrist, scale = 0.16, posture = 'open', mirrorX = false } = options;
  const dir = mirrorX ? -1 : 1;

  const pt = (dx: number, dy: number, z = 0): Landmark => ({
    x: wrist.x + dir * dx * scale,
    y: wrist.y + dy * scale,
    z,
    visibility: 0.99,
  });

  if (posture === 'open') {
    return [
      pt(0, 0), // 0 wrist
      pt(-0.22, -0.18), // 1 thumb cmc
      pt(-0.38, -0.38), // 2 thumb mcp
      pt(-0.52, -0.56), // 3 thumb ip
      pt(-0.64, -0.74), // 4 thumb tip
      pt(-0.22, -0.82), // 5 index mcp
      pt(-0.26, -1.16), // 6 index pip
      pt(-0.28, -1.42), // 7 index dip
      pt(-0.30, -1.66), // 8 index tip
      pt(0.0, -0.88), // 9 middle mcp
      pt(0.0, -1.26), // 10 middle pip
      pt(0.0, -1.54), // 11 middle dip
      pt(0.0, -1.78), // 12 middle tip
      pt(0.22, -0.82), // 13 ring mcp
      pt(0.25, -1.18), // 14 ring pip
      pt(0.27, -1.44), // 15 ring dip
      pt(0.29, -1.66), // 16 ring tip
      pt(0.42, -0.70), // 17 pinky mcp
      pt(0.47, -0.98), // 18 pinky pip
      pt(0.51, -1.18), // 19 pinky dip
      pt(0.55, -1.36), // 20 pinky tip
    ];
  }

  if (posture === 'pinch') {
    return [
      pt(0, 0),
      pt(-0.20, -0.18),
      pt(-0.32, -0.42),
      pt(-0.34, -0.68),
      pt(-0.24, -0.92), // 4 thumb tip near index tip
      pt(-0.18, -0.82),
      pt(-0.20, -1.08),
      pt(-0.20, -1.12),
      pt(-0.18, -1.04), // 8 index tip (~0.13*scale apart from thumb tip)
      pt(0.0, -0.88),
      pt(0.02, -0.68),
      pt(0.02, -0.52),
      pt(0.02, -0.42),
      pt(0.20, -0.82),
      pt(0.20, -0.62),
      pt(0.18, -0.48),
      pt(0.16, -0.38),
      pt(0.38, -0.70),
      pt(0.36, -0.52),
      pt(0.34, -0.40),
      pt(0.32, -0.32),
    ];
  }

  // fist
  return [
    pt(0, 0),
    pt(-0.20, -0.18),
    pt(-0.30, -0.38),
    pt(-0.22, -0.54),
    pt(-0.12, -0.58),
    pt(-0.18, -0.80),
    pt(-0.16, -0.58),
    pt(-0.14, -0.42),
    pt(-0.12, -0.32),
    pt(0.0, -0.84),
    pt(0.0, -0.56),
    pt(0.0, -0.40),
    pt(0.0, -0.30),
    pt(0.18, -0.80),
    pt(0.16, -0.56),
    pt(0.14, -0.40),
    pt(0.12, -0.30),
    pt(0.35, -0.68),
    pt(0.32, -0.50),
    pt(0.28, -0.38),
    pt(0.24, -0.28),
  ];
}

export function makeSyntheticRgbaFrame(
  width: number,
  height: number,
  bgRgb: [number, number, number] = [135, 115, 100]
): Uint8ClampedArray {
  const buf = new Uint8ClampedArray(width * height * 4);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      // Subtle checkerboard texture so contrast & blur metrics are realistic
      const texture = ((x >> 2) + (y >> 2)) % 2 === 0 ? 22 : -22;
      const idx = (y * width + x) * 4;
      buf[idx] = Math.max(0, Math.min(255, bgRgb[0] + texture));
      buf[idx + 1] = Math.max(0, Math.min(255, bgRgb[1] + texture));
      buf[idx + 2] = Math.max(0, Math.min(255, bgRgb[2] + texture));
      buf[idx + 3] = 255;
    }
  }
  return buf;
}

export function drawSyntheticCircle(
  rgba: Uint8ClampedArray,
  width: number,
  height: number,
  centerNorm: { x: number; y: number },
  radiusNorm: number,
  rgb: [number, number, number],
  noiseAmplitude = 4
): void {
  const cx = centerNorm.x * width;
  const cy = centerNorm.y * height;
  const radiusPx = radiusNorm * Math.min(width, height);
  const rSq = radiusPx * radiusPx;

  const x0 = Math.max(0, Math.floor(cx - radiusPx));
  const x1 = Math.min(width - 1, Math.ceil(cx + radiusPx));
  const y0 = Math.max(0, Math.floor(cy - radiusPx));
  const y1 = Math.min(height - 1, Math.ceil(cy + radiusPx));

  for (let py = y0; py <= y1; py++) {
    for (let px = x0; px <= x1; px++) {
      const dx = px - cx;
      const dy = py - cy;
      if (dx * dx + dy * dy <= rSq) {
        const idx = (py * width + px) * 4;
        const n = ((px * 31 + py * 17) % (noiseAmplitude * 2 + 1)) - noiseAmplitude;
        rgba[idx] = Math.max(0, Math.min(255, rgb[0] + n));
        rgba[idx + 1] = Math.max(0, Math.min(255, rgb[1] + n));
        rgba[idx + 2] = Math.max(0, Math.min(255, rgb[2] + n));
        rgba[idx + 3] = 255;
      }
    }
  }
}

export function makeDetection(
  wrist: { x: number; y: number },
  reportedHandedness: HandSide,
  score = 0.92
): RawHandDetection {
  return {
    landmarks: makeSyntheticHandLandmarks({ wrist, posture: 'fist' }),
    handedness: reportedHandedness,
    handednessScore: score,
  };
}
