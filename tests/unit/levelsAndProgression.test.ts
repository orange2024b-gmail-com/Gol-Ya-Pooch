import { describe, expect, it } from 'vitest';
import {
  LEVELS,
  buildRuleSetForLevel,
  computeNextUnlockedLevel,
  getLevelDefinition,
  isLevelUnlocked,
} from '../../src/game/levels';
import {
  InMemoryProgressionStore,
  applyMatchCompletionToProgression,
  applyRoundCompletionToProgression,
  createDefaultProgressionState,
} from '../../src/storage/progressionStore';

describe('Levels 1-5 & Progression Store (src/game/levels.ts, src/storage/progressionStore.ts)', () => {
  it('defines all 5 Section 9 levels with exact rules, timeouts, and constraints', () => {
    expect(LEVELS.length).toBe(5);
    expect(getLevelDefinition(1).nameFa).toBe('تازه‌کار');
    expect(getLevelDefinition(1).humanHideDurationMs).toBe(15_000);
    expect(getLevelDefinition(2).nameFa).toBe('آماتور');
    expect(getLevelDefinition(2).humanHideDurationMs).toBe(10_000);
    expect(getLevelDefinition(3).nameFa).toBe('حرفه‌ای');
    expect(getLevelDefinition(3).humanHideDurationMs).toBe(7_000);
    expect(getLevelDefinition(3).minHandContactsRequired).toBe(1);
    expect(getLevelDefinition(4).nameFa).toBe('تیمی');
    expect(getLevelDefinition(4).ruleSetIds).toContain('traditional-team');
    expect(getLevelDefinition(5).nameFa).toBe('استاد گل‌گیر');
    expect(getLevelDefinition(5).bestOfMatches).toBe(3);
    expect(getLevelDefinition(5).humanHideDurationMs).toBe(5_000);

    const l4SixHands = buildRuleSetForLevel(4, { teamHandsCount: 6 });
    expect(l4SixHands.id).toBe('traditional-team');
    expect(l4SixHands.handsCount).toBe(6);

    const l5Match0 = buildRuleSetForLevel(5, { seriesMatchIndex: 0 });
    const l5Match1 = buildRuleSetForLevel(5, { seriesMatchIndex: 1, teamHandsCount: 4 });
    expect(l5Match0.id).toBe('classic-1v1');
    expect(l5Match1.id).toBe('traditional-team');
  });

  it('unlocks subsequent levels upon winning and persists stats in progression store', async () => {
    expect(isLevelUnlocked(1, 1)).toBe(true);
    expect(isLevelUnlocked(2, 1)).toBe(false);
    expect(computeNextUnlockedLevel(1, 1)).toBe(2);
    expect(computeNextUnlockedLevel(3, 2)).toBe(3);
    expect(computeNextUnlockedLevel(5, 5)).toBe(5);

    const store = new InMemoryProgressionStore();
    let prog = await store.load();
    expect(prog.unlockedMaxLevel).toBe(1);

    prog = applyRoundCompletionToProgression(prog, {
      winner: 'HUMAN',
      wasComputerHiding: true,
      reason: 'GOL_DIRECT',
      guessRecord: {
        roundIndex: 1,
        handsCount: 4,
        guessedHandIndex: 2,
        humanFoundGol: true,
      },
    });
    expect(prog.humanDirectGolCount).toBe(1);
    expect(prog.humanCurrentWinStreak).toBe(1);

    prog = applyMatchCompletionToProgression(prog, {
      levelId: 1,
      humanWonLevel: true,
      scoreDiff: 4,
    });
    expect(prog.unlockedMaxLevel).toBe(2);
    expect(prog.completedLevels).toContain(1);
    await store.save(prog);

    const reloaded = await store.load();
    expect(reloaded.unlockedMaxLevel).toBe(2);
    expect(reloaded.perLevelStats[1]?.won).toBe(1);
    expect(reloaded.humanGuessHistory.length).toBe(1);

    const reset = await store.reset();
    expect(reset).toEqual({
      ...createDefaultProgressionState(),
      updatedAtMs: reset.updatedAtMs,
    });
  });
});
