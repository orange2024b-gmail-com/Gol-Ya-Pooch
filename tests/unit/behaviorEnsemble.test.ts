import { describe, expect, it } from 'vitest';
import {
  MarkovWinStayLoseShiftExpert,
  NGramExpert,
  OverallFrequencyExpert,
  RecentFrequencyExpert,
  RepeatAfterSuccessExpert,
  SwitchAfterCaughtExpert,
  type BehaviorExpert,
  type RoundBehaviorRecord,
} from '../../src/ai/behavior';
import { HedgeEnsemble } from '../../src/ai/ensemble';
import {
  DEFAULT_PROFILE_ID,
  InMemoryProfileStore,
  IndexedDbProfileStore,
} from '../../src/storage/profileStore';
import type { HandSide } from '../../src/vision/types';

function makeRecord(
  idx: number,
  chosenHand: HandSide,
  computerGuess: HandSide
): RoundBehaviorRecord {
  return {
    roundIndex: idx,
    timestampMs: idx * 1000,
    chosenHand,
    computerGuess,
    humanWon: chosenHand !== computerGuess,
  };
}

describe('Behavioral experts & Hedge ensemble (src/ai/behavior/, src/ai/ensemble.ts)', () => {
  it('overall and recent frequency experts track global and sliding-window biases', () => {
    const overall = new OverallFrequencyExpert();
    const recent = new RecentFrequencyExpert(4);

    // First 8 rounds LEFT, last 4 rounds RIGHT
    const history: RoundBehaviorRecord[] = [
      ...Array.from({ length: 8 }, (_, i) => makeRecord(i + 1, 'LEFT', 'RIGHT')),
      ...Array.from({ length: 4 }, (_, i) => makeRecord(i + 9, 'RIGHT', 'LEFT')),
    ];

    expect(overall.predictLeft(history)).toBeGreaterThan(0.55);
    expect(recent.predictLeft(history)).toBeLessThan(0.35);
  });

  it('n-gram experts (n = 1..3) learn deterministic cyclic sequences', () => {
    const ngram1 = new NGramExpert(1);
    const ngram2 = new NGramExpert(2);
    const ngram3 = new NGramExpert(3);

    // Alternating LEFT -> RIGHT -> LEFT -> RIGHT ...
    const altHistory: RoundBehaviorRecord[] = Array.from({ length: 10 }, (_, i) =>
      makeRecord(i + 1, i % 2 === 0 ? 'LEFT' : 'RIGHT', 'LEFT')
    );
    // Last was RIGHT, so next should be LEFT
    expect(ngram1.predictLeft(altHistory)).toBeGreaterThan(0.75);

    // Period-3 pattern: L, L, R, L, L, R, L, L, R, L, L -> next should be R
    const seq3: HandSide[] = ['LEFT', 'LEFT', 'RIGHT', 'LEFT', 'LEFT', 'RIGHT', 'LEFT', 'LEFT', 'RIGHT', 'LEFT', 'LEFT'];
    const history3 = seq3.map((h, i) => makeRecord(i + 1, h, 'RIGHT'));
    expect(ngram2.predictLeft(history3)).toBeLessThan(0.3);
    expect(ngram3.predictLeft(history3)).toBeLessThan(0.35);
  });

  it('Markov WSLS, repeat-after-success, and switch-after-caught experts capture outcome-conditioned habits', () => {
    const markov = new MarkovWinStayLoseShiftExpert();
    const repeatWin = new RepeatAfterSuccessExpert();
    const switchLoss = new SwitchAfterCaughtExpert();

    // Player always repeats after winning and switches after being caught
    const history: RoundBehaviorRecord[] = [
      makeRecord(1, 'LEFT', 'RIGHT'), // human won -> repeats LEFT
      makeRecord(2, 'LEFT', 'RIGHT'), // human won -> repeats LEFT
      makeRecord(3, 'LEFT', 'LEFT'), // caught! -> switches to RIGHT
      makeRecord(4, 'RIGHT', 'LEFT'), // human won -> repeats RIGHT
      makeRecord(5, 'RIGHT', 'RIGHT'), // caught! -> switches to LEFT
      makeRecord(6, 'LEFT', 'RIGHT'), // human won -> should repeat LEFT
    ];

    expect(markov.predictLeft(history)).toBeGreaterThan(0.65);
    expect(repeatWin.predictLeft(history)).toBeGreaterThan(0.65);
    expect(switchLoss.predictLeft(history)).toBe(0.5); // last round was a win

    const afterCaught: RoundBehaviorRecord[] = [
      ...history,
      makeRecord(7, 'LEFT', 'LEFT'), // caught on LEFT -> should switch to RIGHT (low P(LEFT))
    ];
    expect(markov.predictLeft(afterCaught)).toBeLessThan(0.35);
    expect(switchLoss.predictLeft(afterCaught)).toBeLessThan(0.3);
  });

  it('HedgeEnsemble shifts multiplicative weights toward accurate experts and persists via profile store', async () => {
    const oracleLeft: BehaviorExpert = {
      id: 'oracle-left',
      description: 'Always predicts LEFT high',
      predictLeft: () => 0.95,
    };
    const badRight: BehaviorExpert = {
      id: 'bad-right',
      description: 'Always predicts RIGHT high',
      predictLeft: () => 0.05,
    };

    const ensemble = new HedgeEnsemble([oracleLeft, badRight], { learningRate: 0.5 });
    const history: RoundBehaviorRecord[] = [];
    for (let i = 0; i < 10; i++) {
      ensemble.update(history, 'LEFT');
      history.push(makeRecord(i + 1, 'LEFT', 'RIGHT'));
    }

    const weights = ensemble.getWeights();
    expect(weights['oracle-left']!).toBeGreaterThan(0.9);
    expect(weights['bad-right']!).toBeGreaterThanOrEqual(0.01);
    expect(weights['oracle-left']! + weights['bad-right']!).toBeCloseTo(1, 6);

    const pred = ensemble.predict(history);
    expect(pred.pLeft).toBeGreaterThan(0.85);

    // Serialize -> ProfileStore -> Restore roundtrip
    const memStore = new InMemoryProfileStore();
    await memStore.saveProfile({
      profileId: DEFAULT_PROFILE_ID,
      updatedAtMs: Date.now(),
      roundsPlayed: 10,
      ensemble: ensemble.serialize(),
      history,
    });

    const loaded = await memStore.loadProfile(DEFAULT_PROFILE_ID);
    expect(loaded).not.toBeNull();
    expect(loaded!.roundsPlayed).toBe(10);

    const restored = new HedgeEnsemble([oracleLeft, badRight]);
    restored.restore(loaded!.ensemble);
    expect(restored.getWeights()['oracle-left']).toBeCloseTo(weights['oracle-left']!, 6);

    // IndexedDbProfileStore falls back cleanly in Node (where indexedDB is undefined)
    const idbStore = new IndexedDbProfileStore();
    await idbStore.saveProfile(loaded!);
    const idbLoaded = await idbStore.loadProfile(DEFAULT_PROFILE_ID);
    expect(idbLoaded?.history).toHaveLength(10);
    await idbStore.clearProfile(DEFAULT_PROFILE_ID);
    expect(await idbStore.loadProfile(DEFAULT_PROFILE_ID)).toBeNull();
  });
});
