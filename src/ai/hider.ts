import { getLevelDefinition, type LevelDefinition, type LevelId } from '../game/levels';

export interface HumanGuessRecord {
  roundIndex: number;
  handsCount: number;
  /** Primary hand index guessed by the human (first GOL call, or first call in round). */
  guessedHandIndex: number;
  /** True if human found the gol in that computer-hides round. */
  humanFoundGol: boolean;
}

export interface HidingPolicyDecision {
  chosenHandIndex: number;
  probabilities: number[];
  predictedHumanGuessProbs: number[];
  exploitAlpha: number;
  maxDeviationFromUniform: number;
  detectedPattern: 'uniform' | 'frequency' | 'sticky' | 'alternating' | 'wsls';
}

export type ChoreographyActionType =
  | 'SWAP'
  | 'FEINT_SWAP'
  | 'MISDIRECTION_CROSS'
  | 'PAUSE';

export interface ChoreographyStep {
  type: ChoreographyActionType;
  slotA: number;
  slotB: number;
  durationMs: number;
  easing: 'smooth' | 'snappy';
}

export interface HidingChoreography {
  handsCount: number;
  totalDurationMs: number;
  steps: ChoreographyStep[];
}

/**
 * Unbiased cryptographic uniform random float in [0, 1).
 */
export function cryptoRandomFloat(): number {
  if (
    typeof globalThis.crypto !== 'undefined' &&
    typeof globalThis.crypto.getRandomValues === 'function'
  ) {
    const arr = new Uint32Array(1);
    globalThis.crypto.getRandomValues(arr);
    return arr[0]! / 4_294_967_296;
  }
  return Math.random();
}

/**
 * Unbiased cryptographic uniform integer in [0, maxExclusive).
 * Uses rejection sampling on 32-bit unsigned integers to avoid modulo bias.
 */
export function cryptoRandomInt(maxExclusive: number, rngFloat = cryptoRandomFloat): number {
  if (!Number.isInteger(maxExclusive) || maxExclusive <= 0) {
    throw new RangeError(`maxExclusive must be a positive integer, got ${maxExclusive}`);
  }
  if (maxExclusive === 1) return 0;

  if (
    rngFloat === cryptoRandomFloat &&
    typeof globalThis.crypto !== 'undefined' &&
    typeof globalThis.crypto.getRandomValues === 'function'
  ) {
    const range = 4_294_967_296;
    const limit = range - (range % maxExclusive);
    const buf = new Uint32Array(1);
    while (true) {
      globalThis.crypto.getRandomValues(buf);
      const val = buf[0]!;
      if (val < limit) {
        return val % maxExclusive;
      }
    }
  }

  const u = Math.max(0, Math.min(0.999999999999, rngFloat()));
  return Math.floor(u * maxExclusive);
}

export function sampleFromDistribution(
  probs: readonly number[],
  rngFloat: () => number = cryptoRandomFloat
): number {
  const r = Math.max(0, Math.min(0.999999999999, rngFloat()));
  let cdf = 0;
  for (let i = 0; i < probs.length; i++) {
    cdf += probs[i]!;
    if (r < cdf) return i;
  }
  return probs.length - 1;
}

function normalizeProbs(weights: readonly number[]): number[] {
  const sum = weights.reduce((acc, w) => acc + Math.max(0, w), 0);
  if (sum <= 0) {
    return weights.map(() => 1 / weights.length);
  }
  return weights.map((w) => Math.max(0, w) / sum);
}

interface GuesserPredictorOutput {
  name: 'frequency' | 'sticky' | 'alternating' | 'wsls';
  probs: number[];
}

function evaluateGuesserPredictors(
  historySlice: readonly HumanGuessRecord[],
  handsCount: number
): GuesserPredictorOutput[] {
  const uniform = Array.from({ length: handsCount }, () => 1 / handsCount);
  if (historySlice.length === 0) {
    return [
      { name: 'frequency', probs: [...uniform] },
      { name: 'sticky', probs: [...uniform] },
      { name: 'alternating', probs: [...uniform] },
      { name: 'wsls', probs: [...uniform] },
    ];
  }

  // 1. Frequency predictor with recency weighting & Laplace smoothing
  const counts = new Array<number>(handsCount).fill(0.5);
  for (let i = 0; i < historySlice.length; i++) {
    const rec = historySlice[i]!;
    const idx = Math.max(0, Math.min(handsCount - 1, rec.guessedHandIndex));
    const recencyWeight = 0.6 + (0.8 * (i + 1)) / historySlice.length;
    counts[idx] = (counts[idx] ?? 0) + recencyWeight;
  }
  const freqProbs = normalizeProbs(counts);

  const last = historySlice[historySlice.length - 1]!;
  const lastIdx = Math.max(0, Math.min(handsCount - 1, last.guessedHandIndex));

  // 2. Sticky predictor (empirical repeat rate)
  let repeatCount = 0;
  let pairTotal = 0;
  for (let i = 1; i < historySlice.length; i++) {
    pairTotal += 1;
    if (historySlice[i]!.guessedHandIndex === historySlice[i - 1]!.guessedHandIndex) {
      repeatCount += 1;
    }
  }
  const repeatProb = pairTotal > 0 ? (repeatCount + 0.5) / (pairTotal + 1) : 1 / handsCount;
  const stickyProbs = Array.from({ length: handsCount }, (_, i) =>
    i === lastIdx ? repeatProb : (1 - repeatProb) / Math.max(1, handsCount - 1)
  );

  // 3. Alternating predictor (prefers switching away from lastIdx; for K>2 also tracks +1 step)
  let switchCount = 0;
  let stepPlusOneCount = 0;
  for (let i = 1; i < historySlice.length; i++) {
    const prev = historySlice[i - 1]!.guessedHandIndex;
    const cur = historySlice[i]!.guessedHandIndex;
    if (cur !== prev) switchCount += 1;
    if (cur === (prev + 1) % handsCount) stepPlusOneCount += 1;
  }
  const switchProb = pairTotal > 0 ? (switchCount + 0.5) / (pairTotal + 1) : 1 - 1 / handsCount;
  const altProbs = new Array<number>(handsCount).fill(0);
  if (handsCount === 2) {
    altProbs[lastIdx] = 1 - switchProb;
    altProbs[1 - lastIdx] = switchProb;
  } else {
    const nextStepIdx = (lastIdx + 1) % handsCount;
    const stepPlusOneRate = pairTotal > 0 ? stepPlusOneCount / pairTotal : 0;
    const stayProb = 1 - switchProb;
    const stepProb = Math.min(switchProb, Math.max(switchProb / (handsCount - 1), stepPlusOneRate));
    const restProb = Math.max(0, switchProb - stepProb) / Math.max(1, handsCount - 2);
    for (let i = 0; i < handsCount; i++) {
      if (i === lastIdx) altProbs[i] = stayProb;
      else if (i === nextStepIdx) altProbs[i] = stepProb;
      else altProbs[i] = restProb;
    }
  }

  // 4. Win-Stay / Lose-Shift predictor
  let winTotal = 0;
  let winStay = 0;
  let loseTotal = 0;
  let loseShift = 0;
  for (let i = 1; i < historySlice.length; i++) {
    const prev = historySlice[i - 1]!;
    const cur = historySlice[i]!;
    if (prev.humanFoundGol) {
      winTotal += 1;
      if (cur.guessedHandIndex === prev.guessedHandIndex) winStay += 1;
    } else {
      loseTotal += 1;
      if (cur.guessedHandIndex !== prev.guessedHandIndex) loseShift += 1;
    }
  }
  const winStayRate = winTotal > 0 ? (winStay + 0.5) / (winTotal + 1) : 1 / handsCount;
  const loseShiftRate = loseTotal > 0 ? (loseShift + 0.5) / (loseTotal + 1) : 1 - 1 / handsCount;
  const stayProbWsls = last.humanFoundGol ? winStayRate : 1 - loseShiftRate;
  const wslsProbs = Array.from({ length: handsCount }, (_, i) =>
    i === lastIdx ? stayProbWsls : (1 - stayProbWsls) / Math.max(1, handsCount - 1)
  );

  return [
    { name: 'frequency', probs: normalizeProbs(freqProbs) },
    { name: 'sticky', probs: normalizeProbs(stickyProbs) },
    { name: 'alternating', probs: normalizeProbs(altProbs) },
    { name: 'wsls', probs: normalizeProbs(wslsProbs) },
  ];
}

function predictHumanGuessDistribution(
  history: readonly HumanGuessRecord[],
  handsCount: number
): {
  predictedProbs: number[];
  dominantPattern: 'uniform' | 'frequency' | 'sticky' | 'alternating' | 'wsls';
} {
  const relevant = history.filter((r) => r.handsCount === handsCount).slice(-40);
  const uniform = Array.from({ length: handsCount }, () => 1 / handsCount);
  if (relevant.length < 4) {
    return { predictedProbs: uniform, dominantPattern: 'uniform' };
  }

  // Online Hedge scoring over recent history to weight the 4 predictors
  const expertNames: Array<'frequency' | 'sticky' | 'alternating' | 'wsls'> = [
    'frequency',
    'sticky',
    'alternating',
    'wsls',
  ];
  const expertWeights = [1, 1, 1, 1];
  const eta = 1.6;

  for (let t = 2; t < relevant.length; t++) {
    const prefix = relevant.slice(0, t);
    const preds = evaluateGuesserPredictors(prefix, handsCount);
    const actualIdx = Math.max(0, Math.min(handsCount - 1, relevant[t]!.guessedHandIndex));
    for (let e = 0; e < preds.length; e++) {
      const pActual = Math.max(0.02, preds[e]!.probs[actualIdx] ?? 1 / handsCount);
      const loss = -Math.log(pActual);
      expertWeights[e] = expertWeights[e]! * Math.exp(-eta * Math.min(2.5, loss));
    }
    const wSum = expertWeights.reduce((a, b) => a + b, 0);
    if (wSum > 0) {
      for (let e = 0; e < expertWeights.length; e++) {
        expertWeights[e] = expertWeights[e]! / wSum;
      }
    }
  }

  const finalPreds = evaluateGuesserPredictors(relevant, handsCount);
  const mixed = new Array<number>(handsCount).fill(0);
  let bestExpertIdx = 0;
  for (let e = 0; e < finalPreds.length; e++) {
    if (expertWeights[e]! > expertWeights[bestExpertIdx]!) {
      bestExpertIdx = e;
    }
    for (let i = 0; i < handsCount; i++) {
      mixed[i] = (mixed[i] ?? 0) + expertWeights[e]! * finalPreds[e]!.probs[i]!;
    }
  }

  const normalized = normalizeProbs(mixed);
  const maxP = Math.max(...normalized);
  const minP = Math.min(...normalized);
  const dominantPattern = maxP - minP < 0.08 ? 'uniform' : expertNames[bestExpertIdx]!;

  return { predictedProbs: normalized, dominantPattern };
}

/**
 * Computes the computer's hiding probability distribution and chooses a hand.
 * - Levels 1-4: strictly uniform 1/K via unbiased crypto RNG.
 * - Level 5: Nash uniform baseline + bounded exploitation of detected human guess bias.
 */
export function chooseComputerHidingHand(
  levelOrId: LevelId | LevelDefinition,
  handsCount: number,
  humanGuessHistory: readonly HumanGuessRecord[] = [],
  rngFloat: () => number = cryptoRandomFloat
): HidingPolicyDecision {
  const level = typeof levelOrId === 'number' ? getLevelDefinition(levelOrId) : levelOrId;
  const uniformVal = 1 / handsCount;
  const uniformProbs = Array.from({ length: handsCount }, () => uniformVal);

  if (!level.allowBiasExploitation || level.maxExploitDeviation <= 0) {
    const chosen = cryptoRandomInt(handsCount, rngFloat);
    return {
      chosenHandIndex: chosen,
      probabilities: uniformProbs,
      predictedHumanGuessProbs: uniformProbs,
      exploitAlpha: 0,
      maxDeviationFromUniform: 0,
      detectedPattern: 'uniform',
    };
  }

  const relevant = humanGuessHistory.filter((r) => r.handsCount === handsCount).slice(-40);
  const { predictedProbs, dominantPattern } = predictHumanGuessDistribution(relevant, handsCount);

  if (relevant.length < 4 || dominantPattern === 'uniform') {
    const chosen = cryptoRandomInt(handsCount, rngFloat);
    return {
      chosenHandIndex: chosen,
      probabilities: uniformProbs,
      predictedHumanGuessProbs: predictedProbs,
      exploitAlpha: 0,
      maxDeviationFromUniform: 0,
      detectedPattern: 'uniform',
    };
  }

  const maxQ = Math.max(...predictedProbs);
  const minQ = Math.min(...predictedProbs);
  const biasGap = maxQ - minQ;
  const sampleRamp = Math.min(1, (relevant.length - 3) / 8);
  const exploitStrength = Math.max(0, Math.min(1, sampleRamp * (biasGap / 0.28)));

  // Softmax best-response: concentrate hiding weight on the hands least likely to be guessed
  const tau = 7.0;
  const rawAnti = predictedProbs.map((q) => Math.exp(-tau * q));
  const antiDist = normalizeProbs(rawAnti);

  // Strictly bound deviation |p_i - 1/K| <= effectiveMaxDev < 1/K so computer stays unpredictable
  const effectiveMaxDev = Math.min(level.maxExploitDeviation, 0.45 / handsCount);
  let maxAntiDev = 0;
  for (const a of antiDist) {
    maxAntiDev = Math.max(maxAntiDev, Math.abs(a - uniformVal));
  }
  const maxMixWeight = maxAntiDev > 1e-6 ? Math.min(1, effectiveMaxDev / maxAntiDev) : 0;
  const mixWeight = exploitStrength * maxMixWeight;

  const probabilities = normalizeProbs(
    antiDist.map((a, i) => (1 - mixWeight) * uniformProbs[i]! + mixWeight * a)
  );

  let maxDeviationFromUniform = 0;
  for (const p of probabilities) {
    maxDeviationFromUniform = Math.max(maxDeviationFromUniform, Math.abs(p - uniformVal));
  }

  const chosenHandIndex = sampleFromDistribution(probabilities, rngFloat);

  return {
    chosenHandIndex,
    probabilities,
    predictedHumanGuessProbs: predictedProbs,
    exploitAlpha: mixWeight,
    maxDeviationFromUniform,
    detectedPattern: dominantPattern,
  };
}

/**
 * Generates level-driven shuffle, feint, uneven-tempo, and misdirection choreography
 * for the PixiJS v8 computer-hiding scene.
 */
export function generateHidingChoreography(
  levelOrId: LevelId | LevelDefinition,
  handsCount: number,
  reducedMotion = false,
  rngFloat: () => number = cryptoRandomFloat
): HidingChoreography {
  const level = typeof levelOrId === 'number' ? getLevelDefinition(levelOrId) : levelOrId;

  if (reducedMotion) {
    return {
      handsCount,
      totalDurationMs: 600,
      steps: [{ type: 'PAUSE', slotA: 0, slotB: 0, durationMs: 600, easing: 'smooth' }],
    };
  }

  const anim = level.animation;
  const steps: ChoreographyStep[] = [];
  const totalOps = anim.shuffleSteps + anim.feintCount + anim.misdirectionSwaps;
  const baseStepMs = Math.max(
    220,
    Math.round(anim.shuffleDurationMs / Math.max(1, totalOps))
  );

  const pickTwoDistinct = (): [number, number] => {
    const a = cryptoRandomInt(handsCount, rngFloat);
    let b = cryptoRandomInt(Math.max(1, handsCount - 1), rngFloat);
    if (b >= a) b += 1;
    return [a, Math.min(handsCount - 1, b)];
  };

  let remainingSwaps = anim.shuffleSteps;
  let remainingFeints = anim.feintCount;
  let remainingMisdirection = anim.misdirectionSwaps;

  while (remainingSwaps + remainingFeints + remainingMisdirection > 0) {
    const [slotA, slotB] = pickTwoDistinct();
    const tempoFactor = anim.unevenTempo
      ? rngFloat() < 0.35
        ? 0.62
        : rngFloat() < 0.7
          ? 1.0
          : 1.38
      : 1.0;

    if (remainingFeints > 0 && (remainingSwaps <= remainingFeints || rngFloat() < 0.38)) {
      steps.push({
        type: 'FEINT_SWAP',
        slotA,
        slotB,
        durationMs: Math.round(baseStepMs * tempoFactor * 1.05),
        easing: 'snappy',
      });
      remainingFeints -= 1;
    } else if (remainingMisdirection > 0 && handsCount >= 4 && rngFloat() < 0.45) {
      steps.push({
        type: 'MISDIRECTION_CROSS',
        slotA,
        slotB,
        durationMs: Math.round(baseStepMs * tempoFactor),
        easing: 'snappy',
      });
      remainingMisdirection -= 1;
    } else if (remainingSwaps > 0) {
      steps.push({
        type: 'SWAP',
        slotA,
        slotB,
        durationMs: Math.round(baseStepMs * tempoFactor),
        easing: anim.unevenTempo ? 'snappy' : 'smooth',
      });
      remainingSwaps -= 1;
    } else if (remainingMisdirection > 0) {
      steps.push({
        type: 'MISDIRECTION_CROSS',
        slotA,
        slotB,
        durationMs: Math.round(baseStepMs * tempoFactor),
        easing: 'snappy',
      });
      remainingMisdirection -= 1;
    }

    if (anim.unevenTempo && rngFloat() < 0.25 && steps.length < totalOps + 2) {
      steps.push({
        type: 'PAUSE',
        slotA: 0,
        slotB: 0,
        durationMs: 140,
        easing: 'smooth',
      });
    }
  }

  const totalDurationMs = steps.reduce((acc, s) => acc + s.durationMs, 0);
  return {
    handsCount,
    totalDurationMs,
    steps,
  };
}
