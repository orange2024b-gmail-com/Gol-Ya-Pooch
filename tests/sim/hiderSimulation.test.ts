import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { describe, expect, it } from 'vitest';
import {
  chooseComputerHidingHand,
  cryptoRandomFloat,
  type HumanGuessRecord,
} from '../../src/ai/hider';
import { getLevelDefinition, type LevelId } from '../../src/game/levels';

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
    state = (state + 0x6d2b79f5) >>> 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4_294_967_296;
  };
}

export type SimBotType = 'random' | 'alternating' | 'sticky' | 'wsls' | 'adversarial';

export interface LevelBotSimResult {
  levelId: LevelId;
  levelNameEn: string;
  handsCount: number;
  bot: SimBotType;
  rounds: number;
  computerWinRate: number;
  botWinRate: number;
  maxDeviationObserved: number;
  meanDeviationObserved: number;
}

export function simulateLevelAgainstBot(
  levelId: LevelId,
  bot: SimBotType,
  handsCount: number,
  rounds: number,
  seed: number
): LevelBotSimResult {
  const botRng = makeSeededRng(seed);
  const level = getLevelDefinition(levelId);
  const history: HumanGuessRecord[] = [];
  const computerCountsRecent: number[] = new Array<number>(handsCount).fill(0);
  let computerWins = 0;
  let maxDev = 0;
  let sumDev = 0;
  let prevGuess = 0;
  let prevHumanWon = false;

  for (let r = 0; r < rounds; r++) {
    const decision = chooseComputerHidingHand(levelId, handsCount, history, cryptoRandomFloat);
    maxDev = Math.max(maxDev, decision.maxDeviationFromUniform);
    sumDev += decision.maxDeviationFromUniform;

    let guess: number;
    if (bot === 'random') {
      guess = Math.floor(botRng() * handsCount);
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
    } else if (bot === 'wsls') {
      if (r === 0) {
        guess = 0;
      } else if (prevHumanWon) {
        guess = botRng() < 0.9 ? prevGuess : (prevGuess + 1) % handsCount;
      } else {
        guess = botRng() < 0.9 ? (prevGuess + 1) % handsCount : prevGuess;
      }
    } else {
      // adversarial anti-pattern bot:
      // Tracks the computer's hiding history with exponential decay and guesses the
      // computer's highest-frequency slot while injecting 15% anti-exploitation noise.
      if (r < 4 || botRng() < 0.15) {
        guess = Math.floor(botRng() * handsCount);
      } else {
        let bestIdx = 0;
        let bestScore = -1;
        for (let i = 0; i < handsCount; i++) {
          const score = computerCountsRecent[i] ?? 0;
          if (score > bestScore) {
            bestScore = score;
            bestIdx = i;
          }
        }
        guess = bestIdx;
      }
    }

    const humanWon = guess === decision.chosenHandIndex;
    if (!humanWon) computerWins += 1;

    for (let i = 0; i < handsCount; i++) {
      computerCountsRecent[i] =
        (computerCountsRecent[i] ?? 0) * 0.92 + (i === decision.chosenHandIndex ? 1 : 0);
    }

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
    levelId,
    levelNameEn: level.nameEn,
    handsCount,
    bot,
    rounds,
    computerWinRate: computerWins / rounds,
    botWinRate: 1 - computerWins / rounds,
    maxDeviationObserved: maxDev,
    meanDeviationObserved: sumDev / rounds,
  };
}

describe('Phase 5 Hider Simulation Suite (tests/sim/)', () => {
  it('passes chi-square uniformity test over 10,000 simulated rounds for Levels 1-4', () => {
    const ROUNDS = 10_000;
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

  it(
    'evaluates random, alternating, sticky, win-stay-lose-shift, and adversarial bots across Levels 1-5',
    () => {
    const ROUNDS = 2_000;
    const bots: SimBotType[] = ['random', 'alternating', 'sticky', 'wsls', 'adversarial'];
    const levelConfigs: Array<{ levelId: LevelId; handsCount: number }> = [
      { levelId: 1, handsCount: 2 },
      { levelId: 2, handsCount: 2 },
      { levelId: 3, handsCount: 2 },
      { levelId: 4, handsCount: 4 },
      { levelId: 5, handsCount: 2 },
    ];

    const allResults: LevelBotSimResult[] = [];
    for (const cfg of levelConfigs) {
      for (let bIdx = 0; bIdx < bots.length; bIdx++) {
        const bot = bots[bIdx]!;
        const res = simulateLevelAgainstBot(
          cfg.levelId,
          bot,
          cfg.handsCount,
          ROUNDS,
          3001 + cfg.levelId * 131 + bIdx * 47
        );
        allResults.push(res);

        if (cfg.levelId <= 4) {
          expect(res.maxDeviationObserved).toBe(0);
          const expectedComputerWin = 1 - 1 / cfg.handsCount;
          expect(res.computerWinRate).toBeGreaterThan(expectedComputerWin - 0.06);
          expect(res.computerWinRate).toBeLessThan(expectedComputerWin + 0.06);
        } else {
          // Level 5 bounded exploitation check across all 5 bots
          expect(res.maxDeviationObserved).toBeLessThanOrEqual(0.18 + 1e-9);
          if (bot === 'alternating' || bot === 'sticky' || bot === 'wsls') {
            expect(res.computerWinRate).toBeGreaterThan(0.58);
          } else {
            // Even against random and adversarial anti-pattern bots, computer stays near/above equilibrium
            expect(res.computerWinRate).toBeGreaterThanOrEqual(0.45);
          }
        }
      }
    }

    const level5K4 = bots.map((bot, idx) =>
      simulateLevelAgainstBot(5, bot, 4, ROUNDS, 9001 + idx * 53)
    );
    for (const res of level5K4) {
      expect(res.maxDeviationObserved).toBeLessThanOrEqual(0.1125 + 1e-9);
      expect(res.computerWinRate).toBeGreaterThanOrEqual(0.71);
    }

    if (process.env.WRITE_SIM_REPORT === '1') {
      const lines = [
        '# Bot Simulation Report (Levels 1–5)',
        '',
        `Generated from \`tests/sim/hiderSimulation.test.ts\` (${ROUNDS.toLocaleString('en-US')} rounds per bot-level pair).`,
        '',
        '| Level | Hands (K) | Bot Strategy | Rounds | Computer Win Rate | Bot Win Rate | Max Deviation from 1/K |',
        '|---|---:|---|---:|---:|---:|---:|',
        ...allResults.map(
          (r) =>
            `| Level ${r.levelId} (${r.levelNameEn}) | ${r.handsCount} | ${r.bot} | ${r.rounds} | ${(r.computerWinRate * 100).toFixed(2)}% | ${(r.botWinRate * 100).toFixed(2)}% | ${r.maxDeviationObserved.toFixed(4)} |`
        ),
        '',
        '## Level 5 Team Variant (K = 4 Hands)',
        '',
        '| Level | Hands (K) | Bot Strategy | Rounds | Computer Win Rate | Bot Win Rate | Max Deviation from 1/K |',
        '|---|---:|---|---:|---:|---:|---:|',
        ...level5K4.map(
          (r) =>
            `| Level ${r.levelId} (${r.levelNameEn}) | ${r.handsCount} | ${r.bot} | ${r.rounds} | ${(r.computerWinRate * 100).toFixed(2)}% | ${(r.botWinRate * 100).toFixed(2)}% | ${r.maxDeviationObserved.toFixed(4)} |`
        ),
        '',
      ];
      writeFileSync(resolve(process.cwd(), 'docs/sim-report.md'), lines.join('\n'), 'utf8');
    }

    console.info('[SIM_ALL_LEVELS]', JSON.stringify(allResults));
    console.info('[SIM_LEVEL5_K4]', JSON.stringify(level5K4));
  },
  20_000
  );
});
