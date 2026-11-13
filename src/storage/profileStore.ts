import { openDB, type DBSchema, type IDBPDatabase } from 'idb';
import type { RoundBehaviorRecord } from '../ai/behavior/types';
import type { SerializedEnsembleState } from '../ai/ensemble';

export const DEFAULT_PROFILE_ID = 'default-player';
const DB_NAME = 'gol-ya-pooch-db';
const DB_VERSION = 1;
const MAX_HISTORY_RECORDS = 500;

export interface PlayerBehaviorProfile {
  profileId: string;
  updatedAtMs: number;
  roundsPlayed: number;
  ensemble: SerializedEnsembleState;
  history: RoundBehaviorRecord[];
}

interface GolYaPoochDbSchema extends DBSchema {
  profiles: {
    key: string;
    value: PlayerBehaviorProfile;
  };
}

function sanitizeNumericProfile(profile: PlayerBehaviorProfile): PlayerBehaviorProfile {
  // Strictly strip/reconstruct only numeric & enum fields so no media/image data can ever be persisted.
  const weights: Record<string, number> = {};
  for (const [k, v] of Object.entries(profile.ensemble.weights)) {
    if (typeof v === 'number' && Number.isFinite(v)) weights[k] = v;
  }
  const cumulativeLosses: Record<string, number> = {};
  for (const [k, v] of Object.entries(profile.ensemble.cumulativeLosses)) {
    if (typeof v === 'number' && Number.isFinite(v)) cumulativeLosses[k] = v;
  }

  const history: RoundBehaviorRecord[] = profile.history
    .slice(-MAX_HISTORY_RECORDS)
    .map((r) => ({
      roundIndex: Number(r.roundIndex) || 0,
      timestampMs: Number(r.timestampMs) || 0,
      chosenHand: r.chosenHand === 'RIGHT' ? 'RIGHT' : 'LEFT',
      computerGuess: r.computerGuess === 'RIGHT' ? 'RIGHT' : 'LEFT',
      humanWon: Boolean(r.humanWon),
    }));

  return {
    profileId: String(profile.profileId),
    updatedAtMs: Number(profile.updatedAtMs) || Date.now(),
    roundsPlayed: Number(profile.roundsPlayed) || history.length,
    ensemble: {
      weights,
      cumulativeLosses,
      roundsUpdated: Number(profile.ensemble.roundsUpdated) || 0,
    },
    history,
  };
}

export interface ProfileStorageBackend {
  loadProfile(profileId?: string): Promise<PlayerBehaviorProfile | null>;
  saveProfile(profile: PlayerBehaviorProfile): Promise<void>;
  clearProfile(profileId?: string): Promise<void>;
}

/**
 * In-memory fallback used in headless Node tests or restricted browser environments
 * where IndexedDB is unavailable.
 */
export class InMemoryProfileStore implements ProfileStorageBackend {
  private readonly store = new Map<string, PlayerBehaviorProfile>();

  loadProfile(profileId = DEFAULT_PROFILE_ID): Promise<PlayerBehaviorProfile | null> {
    const found = this.store.get(profileId);
    return Promise.resolve(found ? sanitizeNumericProfile(found) : null);
  }

  saveProfile(profile: PlayerBehaviorProfile): Promise<void> {
    const clean = sanitizeNumericProfile(profile);
    this.store.set(clean.profileId, clean);
    return Promise.resolve();
  }

  clearProfile(profileId = DEFAULT_PROFILE_ID): Promise<void> {
    this.store.delete(profileId);
    return Promise.resolve();
  }
}

/**
 * IndexedDB-backed numeric-only player profile store via `idb`.
 * Never stores images or video frames.
 */
export class IndexedDbProfileStore implements ProfileStorageBackend {
  private dbPromise: Promise<IDBPDatabase<GolYaPoochDbSchema>> | null = null;
  private readonly fallback = new InMemoryProfileStore();

  private getDb(): Promise<IDBPDatabase<GolYaPoochDbSchema>> | null {
    if (typeof indexedDB === 'undefined') {
      return null;
    }
    if (!this.dbPromise) {
      this.dbPromise = openDB<GolYaPoochDbSchema>(DB_NAME, DB_VERSION, {
        upgrade(db) {
          if (!db.objectStoreNames.contains('profiles')) {
            db.createObjectStore('profiles', { keyPath: 'profileId' });
          }
        },
      });
    }
    return this.dbPromise;
  }

  async loadProfile(profileId = DEFAULT_PROFILE_ID): Promise<PlayerBehaviorProfile | null> {
    const dbP = this.getDb();
    if (!dbP) return this.fallback.loadProfile(profileId);
    try {
      const db = await dbP;
      const row = await db.get('profiles', profileId);
      return row ? sanitizeNumericProfile(row) : null;
    } catch {
      return this.fallback.loadProfile(profileId);
    }
  }

  async saveProfile(profile: PlayerBehaviorProfile): Promise<void> {
    const clean = sanitizeNumericProfile(profile);
    const dbP = this.getDb();
    if (!dbP) {
      await this.fallback.saveProfile(clean);
      return;
    }
    try {
      const db = await dbP;
      await db.put('profiles', clean);
    } catch {
      await this.fallback.saveProfile(clean);
    }
  }

  async clearProfile(profileId = DEFAULT_PROFILE_ID): Promise<void> {
    const dbP = this.getDb();
    if (!dbP) {
      await this.fallback.clearProfile(profileId);
      return;
    }
    try {
      const db = await dbP;
      await db.delete('profiles', profileId);
    } catch {
      await this.fallback.clearProfile(profileId);
    }
  }
}
