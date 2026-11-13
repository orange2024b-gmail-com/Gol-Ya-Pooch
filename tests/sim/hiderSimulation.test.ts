import { describe, expect, it } from 'vitest';
import {
  chooseComputerHidingHand,
  cryptoRandomFloat,
  type HumanGuessRecord,
} from '../../src/ai/hider';
import type { LevelId } from '../../src/game/levels';

function computeChiSquare(counts: readonly number[], total: number): number {
  const expected = total / counts.length;
  let chi2 = 0;
  for (const c of counts) {
    const diff = c - expected;
    chi2 += (diff * diff) / expected;
  }
  return chi2;
}

function makeSeededRng(seedInitial: number): () => number {
  let state = seedInitial >>> 0;
  return () => {
    // Mulberry32 deterministic PRNG for reproducible simulation
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
  };
}

type BotType = 'sticky' | 'alternating' | 'frequency' | 'wsls' | 'uniform';

function simulateLevel5AgainstBot(
  bot: BotType,
  handsCount: number,
  rounds: number,
  seed: number
): {
  bot: BotType;
  handsCount: number;
  rounds: number;
  computerPreventedHitRate: number;
  humanHitRate: number;
  maxDeviationObserved: number;
  meanDeviationObserved: number;
} {
  const botRng = makeSeededRng(seed);
  const history: HumanGuessRecord[] = [];
  let computerWins = 0;
  let maxDev = 0;
  let sumDev = 0;
  let prevGuess = 0;
  let prevHumanWon = false;

  for (let r = 0; r < rounds; r++) {
    const decision = chooseComputerHidingHand(5, handsCount, history, cryptoRandomFloat);
    maxDev = Math.max(maxDev, decision.maxDeviationFromUniform);
    sumDev += decision.maxDeviationFromUniform;

    let guess: number;
    if (bot === 'uniform') {
      guess = Math.floor(botRng() * handsCount);
    } else if (bot === 'frequency') {
      guess = botRng() < 0.85 ? 0 : Math.floor(botRng() * (handsCount - 1)) + 1;
    } else if (bot === 'sticky') {
      if (r === 0) {
        guess = 0;
      } else if (botRng() < 0.9) {
        guess = prevGuess;
      } else {
        guess = (prevGuess + 1) % handsCount;
      }
    } else if (bot === 'alternating') {
      if (r === 0) {
        guess = 0;
      } else if (botRng() < 0.9) {
        guess = (prevGuess + 1) % handsCount;
      } else {
        guess = prevGuess;
      }
    } else {
      // win-stay-lose-shift
      if (r === 0) {
        guess = 0;
      } else if (prevHumanWon) {
        guess = botRng() < 0.9 ? prevGuess : (prevGuess + 1) % handsCount;
      } else {
        guess = botRng() < 0.9 ? (prevGuess + 1) % handsCount : prevGuess;
      }
    }

    const humanWon = guess === decision.chosenHandIndex;
    if (!humanWon) computerWins += 1;

    history.push({
      roundIndex: r + 1,
      handsCount,
      guessedHandIndex: guess,
      humanFoundGol: humanWon,
    });
    prevGuess = guess;
    prevHumanWon = humanWon;
  }

  return {
    bot,
    handsCount,
    rounds,
    computerPreventedHitRate: computerWins / rounds,
    humanHitRate: 1 - computerWins / rounds,
    maxDeviationObserved: maxDev,
    meanDeviationObserved: sumDev / rounds,
  };
}

describe('Phase 3 Hider Simulation Suite (tests/sim/)', () => {
  it('passes chi-square uniformity test over 10,000 simulated rounds for Levels 1-4', () => {
    const ROUNDS = 10_000;
    // Strict alpha = 0.05 thresholds for deterministic check & alpha = 0.0001 thresholds for live crypto check
    const configs: Array<{
      levelId: LevelId;
      handsCount: number;
      critAlpha01: number;
      critCrypto: number;
    }> = [
      { levelId: 1, handsCount: 2, critAlpha01: 6.635, critCrypto: 15.137 },
      { levelId: 2, handsCount: 2, critAlpha01: 6.635, critCrypto: 15.137 },
      { levelId: 3, handsCount: 2, critAlpha01: 6.635, critCrypto: 15.137 },
      { levelId: 4, handsCount: 4, critAlpha01: 11.345, critCrypto: 21.107 },
      { levelId: 4, handsCount: 6, critAlpha01: 15.086, critCrypto: 25.689 },
    ];

    const reportRows: Array<Record<string, unknown>> = [];

    for (let idx = 0; idx < configs.length; idx++) {
      const cfg = configs[idx]!;
      const cryptoCounts = new Array<number>(cfg.handsCount).fill(0);
      for (let i = 0; i < ROUNDS; i++) {
        const dec = chooseComputerHidingHand(cfg.levelId, cfg.handsCount, []);
        cryptoCounts[dec.chosenHandIndex] = (cryptoCounts[dec.chosenHandIndex] ?? 0) + 1;
      }
      const cryptoChi2 = computeChiSquare(cryptoCounts, ROUNDS);
      expect(cryptoChi2).toBeLessThan(cfg.critCrypto);

      const detRng = makeSeededRng(100_003 + idx * 997);
      const detCounts = new Array<number>(cfg.handsCount).fill(0);
      for (let i = 0; i < ROUNDS; i++) {
        const dec = chooseComputerHidingHand(cfg.levelId, cfg.handsCount, [], detRng);
        detCounts[dec.chosenHandIndex] = (detCounts[dec.chosenHandIndex] ?? 0) + 1;
      }
      const detChi2 = computeChiSquare(detCounts, ROUNDS);
      expect(detChi2).toBeLessThan(cfg.critAlpha01);

      reportRows.push({
        levelId: cfg.levelId,
        handsCount: cfg.handsCount,
        rounds: ROUNDS,
        cryptoCounts,
        cryptoChi2: Number(cryptoChi2.toFixed(3)),
        detChi2: Number(detChi2.toFixed(3)),
      });
    }

    console.info('[SIM_UNIFORMITY]', JSON.stringify(reportRows));
  });

  it('achieves measurably higher hit-prevention against biased bots at Level 5 within bounded exploitation limit', () => {
    const ROUNDS = 2_000;
    const biasedBots: BotType[] = ['sticky', 'alternating', 'frequency', 'wsls'];
    const results = [...biasedBots, 'uniform' as BotType].map((bot, idx) =>
      simulateLevel5AgainstBot(bot, 2, ROUNDS, 1001 + idx * 77)
    );

    const results4 = biasedBots.map((bot, idx) =>
      simulateLevel5AgainstBot(bot, 4, ROUNDS, 2001 + idx * 77)
    );

    for (const res of results) {
      expect(res.maxDeviationObserved).toBeLessThanOrEqual(0.18 + 1e-9);
      if (res.bot !== 'uniform') {
        expect(res.computerPreventedHitRate).toBeGreaterThan(0.58);
      } else {
        expect(res.computerPreventedHitRate).toBeGreaterThan(0.45);
        expect(res.computerPreventedHitRate).toBeLessThan(0.55);
      }
    }

    for (const res of results4) {
      expect(res.maxDeviationObserved).toBeLessThanOrEqual(0.1125 + 1e-9);
      expect(res.computerPreventedHitRate).toBeGreaterThan(0.8);
    }

    console.info('[SIM_LEVEL5_K2]', JSON.stringify(results));
    console.info('[SIM_LEVEL5_K4]', JSON.stringify(results4));
  });
});
