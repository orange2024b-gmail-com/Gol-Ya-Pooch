import { openDB, type DBSchema, type IDBPDatabase } from 'idb';
import type { HumanGuessRecord } from '../ai/hider';
import { computeNextUnlockedLevel, type LevelId } from '../game/levels';
import { DEFAULT_PROFILE_ID } from './profileStore';

const PROGRESSION_DB_NAME = 'gol-ya-pooch-progression-db';
const PROGRESSION_DB_VERSION = 1;
const MAX_GUESS_HISTORY = 120;

export interface PerLevelStat {
  played: number;
  won: number;
  bestScoreDiff: number;
}

export interface PlayerProgressionState {
  profileId: string;
  updatedAtMs: number;
  unlockedMaxLevel: LevelId;
  completedLevels: LevelId[];
  selectedLevelId: LevelId;
  teamHandsCount: 4 | 6;
  totalMatchesPlayed: number;
  totalMatchesWon: number;
  totalRoundsPlayed: number;
  humanRoundsWon: number;
  computerRoundsWon: number;
  computerHidesRoundsPlayed: number;
  humanDirectGolCount: number;
  humanCorrectAfterPoochCount: number;
  humanCurrentWinStreak: number;
  humanBestWinStreak: number;
  perLevelStats: Record<number, PerLevelStat>;
  humanGuessHistory: HumanGuessRecord[];
}

interface ProgressionDbSchema extends DBSchema {
  progression: {
    key: string;
    value: PlayerProgressionState;
  };
}

function clampLevelId(val: unknown): LevelId {
  const n = Number(val);
  if (n === 1 || n === 2 || n === 3 || n === 4 || n === 5) return n;
  return 1;
}

export function createDefaultProgressionState(
  profileId = DEFAULT_PROFILE_ID
): PlayerProgressionState {
  return {
    profileId,
    updatedAtMs: Date.now(),
    unlockedMaxLevel: 1,
    completedLevels: [],
    selectedLevelId: 1,
    teamHandsCount: 4,
    totalMatchesPlayed: 0,
    totalMatchesWon: 0,
    totalRoundsPlayed: 0,
    humanRoundsWon: 0,
    computerRoundsWon: 0,
    computerHidesRoundsPlayed: 0,
    humanDirectGolCount: 0,
    humanCorrectAfterPoochCount: 0,
    humanCurrentWinStreak: 0,
    humanBestWinStreak: 0,
    perLevelStats: {
      1: { played: 0, won: 0, bestScoreDiff: 0 },
      2: { played: 0, won: 0, bestScoreDiff: 0 },
      3: { played: 0, won: 0, bestScoreDiff: 0 },
      4: { played: 0, won: 0, bestScoreDiff: 0 },
      5: { played: 0, won: 0, bestScoreDiff: 0 },
    },
    humanGuessHistory: [],
  };
}

function sanitizeProgression(raw: PlayerProgressionState): PlayerProgressionState {
  const def = createDefaultProgressionState(raw.profileId || DEFAULT_PROFILE_ID);
  const completedSet = new Set<LevelId>();
  if (Array.isArray(raw.completedLevels)) {
    for (const id of raw.completedLevels) {
      completedSet.add(clampLevelId(id));
    }
  }

  const perLevelStats: Record<number, PerLevelStat> = { ...def.perLevelStats };
  for (const id of [1, 2, 3, 4, 5] as const) {
    const entry = raw.perLevelStats?.[id];
    if (entry) {
      perLevelStats[id] = {
        played: Math.max(0, Math.round(Number(entry.played) || 0)),
        won: Math.max(0, Math.round(Number(entry.won) || 0)),
        bestScoreDiff: Math.round(Number(entry.bestScoreDiff) || 0),
      };
    }
  }

  const humanGuessHistory: HumanGuessRecord[] = Array.isArray(raw.humanGuessHistory)
    ? raw.humanGuessHistory.slice(-MAX_GUESS_HISTORY).map((r) => ({
        roundIndex: Math.max(0, Math.round(Number(r.roundIndex) || 0)),
        handsCount: Math.max(2, Math.min(8, Math.round(Number(r.handsCount) || 2))),
        guessedHandIndex: Math.max(0, Math.min(7, Math.round(Number(r.guessedHandIndex) || 0))),
        humanFoundGol: Boolean(r.humanFoundGol),
      }))
    : [];

  return {
    profileId: String(raw.profileId || DEFAULT_PROFILE_ID),
    updatedAtMs: Number(raw.updatedAtMs) || Date.now(),
    unlockedMaxLevel: clampLevelId(raw.unlockedMaxLevel),
    completedLevels: [...completedSet].sort((a, b) => a - b),
    selectedLevelId: clampLevelId(raw.selectedLevelId),
    teamHandsCount: raw.teamHandsCount === 6 ? 6 : 4,
    totalMatchesPlayed: Math.max(0, Math.round(Number(raw.totalMatchesPlayed) || 0)),
    totalMatchesWon: Math.max(0, Math.round(Number(raw.totalMatchesWon) || 0)),
    totalRoundsPlayed: Math.max(0, Math.round(Number(raw.totalRoundsPlayed) || 0)),
    humanRoundsWon: Math.max(0, Math.round(Number(raw.humanRoundsWon) || 0)),
    computerRoundsWon: Math.max(0, Math.round(Number(raw.computerRoundsWon) || 0)),
    computerHidesRoundsPlayed: Math.max(
      0,
      Math.round(Number(raw.computerHidesRoundsPlayed) || 0)
    ),
    humanDirectGolCount: Math.max(0, Math.round(Number(raw.humanDirectGolCount) || 0)),
    humanCorrectAfterPoochCount: Math.max(
      0,
      Math.round(Number(raw.humanCorrectAfterPoochCount) || 0)
    ),
    humanCurrentWinStreak: Math.max(0, Math.round(Number(raw.humanCurrentWinStreak) || 0)),
    humanBestWinStreak: Math.max(0, Math.round(Number(raw.humanBestWinStreak) || 0)),
    perLevelStats,
    humanGuessHistory,
  };
}

export function applyRoundCompletionToProgression(
  prev: PlayerProgressionState,
  info: {
    winner: 'HUMAN' | 'COMPUTER';
    wasComputerHiding: boolean;
    reason?: string | null;
    guessRecord?: HumanGuessRecord;
  }
): PlayerProgressionState {
  const humanWon = info.winner === 'HUMAN';
  const nextStreak = humanWon ? prev.humanCurrentWinStreak + 1 : 0;
  const bestStreak = Math.max(prev.humanBestWinStreak, nextStreak);
  const nextHistory = info.guessRecord
    ? [...prev.humanGuessHistory, info.guessRecord].slice(-MAX_GUESS_HISTORY)
    : prev.humanGuessHistory;

  return sanitizeProgression({
    ...prev,
    updatedAtMs: Date.now(),
    totalRoundsPlayed: prev.totalRoundsPlayed + 1,
    humanRoundsWon: prev.humanRoundsWon + (humanWon ? 1 : 0),
    computerRoundsWon: prev.computerRoundsWon + (humanWon ? 0 : 1),
    computerHidesRoundsPlayed:
      prev.computerHidesRoundsPlayed + (info.wasComputerHiding ? 1 : 0),
    humanDirectGolCount:
      prev.humanDirectGolCount +
      (info.wasComputerHiding && info.reason === 'GOL_DIRECT' ? 1 : 0),
    humanCorrectAfterPoochCount:
      prev.humanCorrectAfterPoochCount +
      (info.wasComputerHiding && info.reason === 'GOL_AFTER_POOCH' ? 1 : 0),
    humanCurrentWinStreak: nextStreak,
    humanBestWinStreak: bestStreak,
    humanGuessHistory: nextHistory,
  });
}

export function applyMatchCompletionToProgression(
  prev: PlayerProgressionState,
  info: {
    levelId: LevelId;
    humanWonLevel: boolean;
    scoreDiff: number;
  }
): PlayerProgressionState {
  const curLevelStat = prev.perLevelStats[info.levelId] ?? {
    played: 0,
    won: 0,
    bestScoreDiff: 0,
  };
  const nextLevelStat: PerLevelStat = {
    played: curLevelStat.played + 1,
    won: curLevelStat.won + (info.humanWonLevel ? 1 : 0),
    bestScoreDiff:
      curLevelStat.played === 0
        ? info.scoreDiff
        : Math.max(curLevelStat.bestScoreDiff, info.scoreDiff),
  };

  const nextUnlocked = info.humanWonLevel
    ? computeNextUnlockedLevel(prev.unlockedMaxLevel, info.levelId)
    : prev.unlockedMaxLevel;

  const completed =
    info.humanWonLevel && !prev.completedLevels.includes(info.levelId)
      ? [...prev.completedLevels, info.levelId]
      : prev.completedLevels;

  return sanitizeProgression({
    ...prev,
    updatedAtMs: Date.now(),
    unlockedMaxLevel: nextUnlocked,
    completedLevels: completed,
    totalMatchesPlayed: prev.totalMatchesPlayed + 1,
    totalMatchesWon: prev.totalMatchesWon + (info.humanWonLevel ? 1 : 0),
    perLevelStats: {
      ...prev.perLevelStats,
      [info.levelId]: nextLevelStat,
    },
  });
}

export interface ProgressionStorageBackend {
  load(profileId?: string): Promise<PlayerProgressionState>;
  save(state: PlayerProgressionState): Promise<void>;
  reset(profileId?: string): Promise<PlayerProgressionState>;
}

export class InMemoryProgressionStore implements ProgressionStorageBackend {
  private readonly store = new Map<string, PlayerProgressionState>();

  load(profileId = DEFAULT_PROFILE_ID): Promise<PlayerProgressionState> {
    const existing = this.store.get(profileId);
    return Promise.resolve(
      existing ? sanitizeProgression(existing) : createDefaultProgressionState(profileId)
    );
  }

  save(state: PlayerProgressionState): Promise<void> {
    const clean = sanitizeProgression(state);
    this.store.set(clean.profileId, clean);
    return Promise.resolve();
  }

  reset(profileId = DEFAULT_PROFILE_ID): Promise<PlayerProgressionState> {
    const fresh = createDefaultProgressionState(profileId);
    this.store.set(profileId, fresh);
    return Promise.resolve(fresh);
  }
}

export class IndexedDbProgressionStore implements ProgressionStorageBackend {
  private dbPromise: Promise<IDBPDatabase<ProgressionDbSchema>> | null = null;
  private readonly fallback = new InMemoryProgressionStore();

  private getDb(): Promise<IDBPDatabase<ProgressionDbSchema>> | null {
    if (typeof indexedDB === 'undefined') {
      return null;
    }
    if (!this.dbPromise) {
      this.dbPromise = openDB<ProgressionDbSchema>(
        PROGRESSION_DB_NAME,
        PROGRESSION_DB_VERSION,
        {
          upgrade(db) {
            if (!db.objectStoreNames.contains('progression')) {
              db.createObjectStore('progression', { keyPath: 'profileId' });
            }
          },
        }
      );
    }
    return this.dbPromise;
  }

  async load(profileId = DEFAULT_PROFILE_ID): Promise<PlayerProgressionState> {
    const dbP = this.getDb();
    if (!dbP) return this.fallback.load(profileId);
    try {
      const db = await dbP;
      const row = await db.get('progression', profileId);
      return row ? sanitizeProgression(row) : createDefaultProgressionState(profileId);
    } catch {
      return this.fallback.load(profileId);
    }
  }

  async save(state: PlayerProgressionState): Promise<void> {
    const clean = sanitizeProgression(state);
    const dbP = this.getDb();
    if (!dbP) {
      await this.fallback.save(clean);
      return;
    }
    try {
      const db = await dbP;
      await db.put('progression', clean);
    } catch {
      await this.fallback.save(clean);
    }
  }

  async reset(profileId = DEFAULT_PROFILE_ID): Promise<PlayerProgressionState> {
    const fresh = createDefaultProgressionState(profileId);
    const dbP = this.getDb();
    if (!dbP) {
      return this.fallback.reset(profileId);
    }
    try {
      const db = await dbP;
      await db.put('progression', fresh);
      return fresh;
    } catch {
      return this.fallback.reset(profileId);
    }
  }
}
