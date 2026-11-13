import { openDB, type DBSchema, type IDBPDatabase } from 'idb';
import type { LevelId } from '../game/levels';
import type { BeliefState } from '../tracking/beliefFilter';
import type { TrackingEvent } from '../tracking/eventDetector';
import {
  TEMPORAL_FEATURE_DIM,
  TEMPORAL_FEATURE_NAMES,
  TEMPORAL_SAMPLE_RATE_HZ,
  TEMPORAL_SEQ_LEN,
} from '../tracking/temporalFeatures';
import type { HandSide } from '../vision/types';
import { DEFAULT_PROFILE_ID } from './profileStore';

const ROUND_LOG_DB_NAME = 'gol-ya-pooch-round-logs-db';
const ROUND_LOG_DB_VERSION = 1;
const MAX_STORED_ROUNDS = 2000;

export class PrivacySchemaError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'PrivacySchemaError';
  }
}

const FORBIDDEN_MEDIA_KEY_PATTERN =
  /(image|imagedata|frameimage|video|videoblob|canvas|pixel|pixels|rgba|rgb|bgr|png|jpeg|jpg|webp|webm|mp4|blob|dataurl|base64|screenshot|mediastream|framebytes)/i;

const FORBIDDEN_STRING_PREFIX_PATTERN = /^(data:image\/|data:video\/|blob:)/i;

export interface ConsentSettings {
  profileId: string;
  consentEnabled: boolean;
  updatedAtMs: number;
}

export interface RawNumericFrameEntry {
  tMs: number;
  v: number[];
}

export interface RoundLogFinalGuess {
  guess: HandSide;
  probability: number;
  pLeft: number;
  hmmPLeft: number;
  temporalPLeft: number | null;
  behaviorPLeft: number;
  dominantSource: 'TRACKING' | 'BEHAVIOR' | 'COMBINED';
}

export interface RoundLogTimings {
  hidingDurationMs: number;
  frameCount: number;
  sampleRateHz: number;
  temporalInferenceMs: number | null;
  totalProcessingMs: number;
}

export interface RoundLogRecord {
  schemaVersion: 1;
  id: string;
  profileId: string;
  recordedAtMs: number;
  levelId: LevelId;
  ruleSetId: string;
  initialShownHand: HandSide;
  resampledSequence: number[][];
  rawFrameFeatures: RawNumericFrameEntry[];
  eventTimeline: TrackingEvent[];
  beliefTrace: BeliefState[];
  finalGuess: RoundLogFinalGuess;
  verifiedGroundTruth: HandSide;
  timings: RoundLogTimings;
}

export interface RoundLogExportPayload {
  schemaVersion: 1;
  exportedAtIso: string;
  profileId: string;
  consentEnabled: true;
  privacyNotice: 'numeric-features-and-events-only-no-images-no-video';
  featureNames: readonly string[];
  sampleRateHz: number;
  seqLen: number;
  rounds: RoundLogRecord[];
}

interface RoundLogDbSchema extends DBSchema {
  settings: {
    key: string;
    value: ConsentSettings;
  };
  roundLogs: {
    key: string;
    value: RoundLogRecord;
    indexes: { 'by-profile': string };
  };
}

export function createDefaultConsentSettings(
  profileId = DEFAULT_PROFILE_ID
): ConsentSettings {
  return {
    profileId,
    consentEnabled: false,
    updatedAtMs: 0,
  };
}

function isDisallowedMediaObject(val: unknown): boolean {
  if (typeof val !== 'object' || val === null) return false;
  if (ArrayBuffer.isView(val)) return true;
  if (val instanceof ArrayBuffer) return true;
  if (typeof Blob !== 'undefined' && val instanceof Blob) return true;
  if (typeof ImageData !== 'undefined' && val instanceof ImageData) return true;
  if (typeof HTMLCanvasElement !== 'undefined' && val instanceof HTMLCanvasElement) return true;
  if (typeof HTMLVideoElement !== 'undefined' && val instanceof HTMLVideoElement) return true;
  if (typeof HTMLImageElement !== 'undefined' && val instanceof HTMLImageElement) return true;
  if (typeof MediaStream !== 'undefined' && val instanceof MediaStream) return true;
  return false;
}

/**
 * Recursively asserts that a payload contains strictly numeric/enum/safe metadata fields
 * and contains zero image, video, canvas, blob, or base64 media payloads.
 */
export function assertNumericOnlyPayload(value: unknown, path = 'root'): void {
  if (value === null || value === undefined) return;

  if (typeof value === 'string') {
    if (FORBIDDEN_STRING_PREFIX_PATTERN.test(value.trim())) {
      throw new PrivacySchemaError(
        `Rejected forbidden media URI/blob string at ${path}`
      );
    }
    if (value.length > 4096) {
      throw new PrivacySchemaError(
        `Rejected oversized string (${value.length} chars) at ${path}; only numeric/short metadata allowed`
      );
    }
    return;
  }

  if (typeof value === 'number') {
    if (!Number.isFinite(value)) {
      throw new PrivacySchemaError(`Non-finite numeric value at ${path}`);
    }
    return;
  }

  if (typeof value === 'boolean') return;

  if (isDisallowedMediaObject(value)) {
    throw new PrivacySchemaError(`Rejected binary/media object at ${path}`);
  }

  if (Array.isArray(value)) {
    for (let i = 0; i < value.length; i++) {
      assertNumericOnlyPayload(value[i], `${path}[${i}]`);
    }
    return;
  }

  if (typeof value === 'object') {
    for (const [key, child] of Object.entries(value as Record<string, unknown>)) {
      if (FORBIDDEN_MEDIA_KEY_PATTERN.test(key)) {
        throw new PrivacySchemaError(
          `Rejected forbidden media/image field "${key}" at ${path}.${key}`
        );
      }
      assertNumericOnlyPayload(child, `${path}.${key}`);
    }
  }
}

function clampLevelId(val: unknown): LevelId {
  const n = Number(val);
  if (n === 1 || n === 2 || n === 3 || n === 4 || n === 5) return n;
  return 1;
}

function sanitizeHandSide(val: unknown): HandSide {
  return val === 'RIGHT' ? 'RIGHT' : 'LEFT';
}

function sanitizeNumberMatrix(
  matrix: unknown,
  expectedRows: number,
  expectedCols: number
): number[][] {
  if (!Array.isArray(matrix)) {
    throw new PrivacySchemaError('Expected numeric resampledSequence matrix');
  }
  const rows: number[][] = [];
  for (let r = 0; r < expectedRows; r++) {
    const srcRow: unknown = matrix[r];
    if (srcRow === undefined) {
      rows.push(new Array<number>(expectedCols).fill(0));
      continue;
    }
    if (!Array.isArray(srcRow) || srcRow.length !== expectedCols) {
      throw new PrivacySchemaError(
        `Invalid resampledSequence row ${r}: expected length ${expectedCols}`
      );
    }
    rows.push(
      srcRow.map((v) => {
        const n = Number(v);
        if (!Number.isFinite(n)) {
          throw new PrivacySchemaError(`Non-finite number in resampledSequence row ${r}`);
        }
        return n;
      })
    );
  }
  return rows;
}

function sanitizeRawFrames(rawFrames: unknown): RawNumericFrameEntry[] {
  if (!Array.isArray(rawFrames)) return [];
  return rawFrames.slice(0, 1200).map((entry, idx) => {
    if (typeof entry !== 'object' || entry === null) {
      throw new PrivacySchemaError(`Invalid rawFrameFeatures entry at index ${idx}`);
    }
    const obj = entry as { tMs?: unknown; v?: unknown };
    const tMs = Number(obj.tMs);
    if (!Number.isFinite(tMs)) {
      throw new PrivacySchemaError(`Invalid tMs in rawFrameFeatures[${idx}]`);
    }
    if (!Array.isArray(obj.v) || obj.v.length !== TEMPORAL_FEATURE_DIM) {
      throw new PrivacySchemaError(
        `Invalid feature vector length in rawFrameFeatures[${idx}]`
      );
    }
    const v = obj.v.map((x) => {
      const n = Number(x);
      if (!Number.isFinite(n)) {
        throw new PrivacySchemaError(`Non-finite feature value in rawFrameFeatures[${idx}]`);
      }
      return n;
    });
    return { tMs, v };
  });
}

function sanitizeTimeline(events: unknown): TrackingEvent[] {
  if (!Array.isArray(events)) return [];
  const clean: TrackingEvent[] = [];
  for (const ev of events) {
    if (typeof ev !== 'object' || ev === null) continue;
    const obj = ev as Record<string, unknown>;
    const ts = Number(obj.timestampMs) || 0;
    const rel = Number(obj.relativeSec) || 0;
    switch (obj.type) {
      case 'CONTACT_START':
        clean.push({
          type: 'CONTACT_START',
          timestampMs: ts,
          relativeSec: rel,
          minDistanceNorm: Number(obj.minDistanceNorm) || 0,
        });
        break;
      case 'CONTACT_END':
        clean.push({
          type: 'CONTACT_END',
          timestampMs: ts,
          relativeSec: rel,
          durationMs: Number(obj.durationMs) || 0,
          partialOpeningsDuringContact: Array.isArray(obj.partialOpeningsDuringContact)
            ? obj.partialOpeningsDuringContact.map((s) => sanitizeHandSide(s))
            : [],
        });
        break;
      case 'PARTIAL_OPENING':
        clean.push({
          type: 'PARTIAL_OPENING',
          timestampMs: ts,
          relativeSec: rel,
          side: sanitizeHandSide(obj.side),
          strength: Number(obj.strength) || 0,
          duringContact: Boolean(obj.duringContact),
        });
        break;
      case 'GOL_SIGHTING':
        clean.push({
          type: 'GOL_SIGHTING',
          timestampMs: ts,
          relativeSec: rel,
          side: sanitizeHandSide(obj.side),
          score: Number(obj.score) || 0,
        });
        break;
      case 'OCCLUSION_START':
        clean.push({
          type: 'OCCLUSION_START',
          timestampMs: ts,
          relativeSec: rel,
          side: obj.side === 'BOTH' ? 'BOTH' : sanitizeHandSide(obj.side),
        });
        break;
      case 'OCCLUSION_END':
        clean.push({
          type: 'OCCLUSION_END',
          timestampMs: ts,
          relativeSec: rel,
          durationMs: Number(obj.durationMs) || 0,
        });
        break;
    }
  }
  return clean;
}

function sanitizeBeliefTrace(trace: unknown): BeliefState[] {
  if (!Array.isArray(trace)) return [];
  return trace.slice(0, 1200).map((b) => {
    const obj = (b ?? {}) as Record<string, unknown>;
    const pLeft = Math.max(0, Math.min(1, Number(obj.pLeft) || 0.5));
    const pRight = Math.max(0, Math.min(1, Number(obj.pRight) || 1 - pLeft));
    return {
      timestampMs: Number(obj.timestampMs) || 0,
      pLeft,
      pRight,
      confidence: Math.max(0, Math.min(1, Number(obj.confidence) || Math.abs(pLeft - pRight))),
      mostLikely: pLeft >= pRight ? 'LEFT' : 'RIGHT',
      inContact: Boolean(obj.inContact),
      lowQuality: Boolean(obj.lowQuality),
    };
  });
}

export function sanitizeRoundLogRecord(raw: unknown): RoundLogRecord {
  assertNumericOnlyPayload(raw, 'roundLog');
  if (typeof raw !== 'object' || raw === null) {
    throw new PrivacySchemaError('Round log record must be a non-null object');
  }
  const r = raw as Record<string, unknown>;
  if (r.schemaVersion !== 1) {
    throw new PrivacySchemaError(`Unsupported round log schemaVersion: ${String(r.schemaVersion)}`);
  }

  const fg = (r.finalGuess ?? {}) as Record<string, unknown>;
  const tm = (r.timings ?? {}) as Record<string, unknown>;

  const dominantSource =
    fg.dominantSource === 'TRACKING' || fg.dominantSource === 'BEHAVIOR'
      ? fg.dominantSource
      : 'COMBINED';

  return {
    schemaVersion: 1,
    id: typeof r.id === 'string' && r.id.length > 0 ? r.id : `rl-${Date.now()}`,
    profileId:
      typeof r.profileId === 'string' && r.profileId.length > 0
        ? r.profileId
        : DEFAULT_PROFILE_ID,
    recordedAtMs: Number(r.recordedAtMs) || Date.now(),
    levelId: clampLevelId(r.levelId),
    ruleSetId:
      typeof r.ruleSetId === 'string' && r.ruleSetId.length > 0
        ? r.ruleSetId
        : 'classic-1v1',
    initialShownHand: sanitizeHandSide(r.initialShownHand),
    resampledSequence: sanitizeNumberMatrix(
      r.resampledSequence,
      TEMPORAL_SEQ_LEN,
      TEMPORAL_FEATURE_DIM
    ),
    rawFrameFeatures: sanitizeRawFrames(r.rawFrameFeatures),
    eventTimeline: sanitizeTimeline(r.eventTimeline),
    beliefTrace: sanitizeBeliefTrace(r.beliefTrace),
    finalGuess: {
      guess: sanitizeHandSide(fg.guess),
      probability: Math.max(0, Math.min(1, Number(fg.probability) || 0.5)),
      pLeft: Math.max(0, Math.min(1, Number(fg.pLeft) || 0.5)),
      hmmPLeft: Math.max(0, Math.min(1, Number(fg.hmmPLeft) || 0.5)),
      temporalPLeft:
        fg.temporalPLeft === null || fg.temporalPLeft === undefined
          ? null
          : Math.max(0, Math.min(1, Number(fg.temporalPLeft))),
      behaviorPLeft: Math.max(0, Math.min(1, Number(fg.behaviorPLeft) || 0.5)),
      dominantSource,
    },
    verifiedGroundTruth: sanitizeHandSide(r.verifiedGroundTruth),
    timings: {
      hidingDurationMs: Math.max(0, Number(tm.hidingDurationMs) || 0),
      frameCount: Math.max(0, Math.round(Number(tm.frameCount) || 0)),
      sampleRateHz: Math.max(1, Number(tm.sampleRateHz) || TEMPORAL_SAMPLE_RATE_HZ),
      temporalInferenceMs:
        tm.temporalInferenceMs === null || tm.temporalInferenceMs === undefined
          ? null
          : Math.max(0, Number(tm.temporalInferenceMs)),
      totalProcessingMs: Math.max(0, Number(tm.totalProcessingMs) || 0),
    },
  };
}

export interface RoundLogStorageBackend {
  getConsentSettings(profileId?: string): Promise<ConsentSettings>;
  setConsentEnabled(enabled: boolean, profileId?: string): Promise<ConsentSettings>;
  appendRoundLog(record: RoundLogRecord): Promise<boolean>;
  listRoundLogs(profileId?: string): Promise<RoundLogRecord[]>;
  getRoundLogCount(profileId?: string): Promise<number>;
  exportLogsJson(profileId?: string, pretty?: boolean): Promise<string>;
  importLogsJson(
    rawJson: string,
    options?: { profileIdOverride?: string }
  ): Promise<{ importedCount: number; skippedDuplicates: number }>;
  clearLogs(profileId?: string): Promise<void>;
}

function buildExportPayload(
  profileId: string,
  rounds: RoundLogRecord[]
): RoundLogExportPayload {
  const cleanRounds = rounds.map((r) => sanitizeRoundLogRecord(r));
  const payload: RoundLogExportPayload = {
    schemaVersion: 1,
    exportedAtIso: new Date().toISOString(),
    profileId,
    consentEnabled: true,
    privacyNotice: 'numeric-features-and-events-only-no-images-no-video',
    featureNames: [...TEMPORAL_FEATURE_NAMES],
    sampleRateHz: TEMPORAL_SAMPLE_RATE_HZ,
    seqLen: TEMPORAL_SEQ_LEN,
    rounds: cleanRounds,
  };
  assertNumericOnlyPayload(payload, 'exportPayload');
  return payload;
}

function parseImportPayload(
  rawJson: string,
  profileIdOverride?: string
): RoundLogRecord[] {
  const parsed = JSON.parse(rawJson) as unknown;
  assertNumericOnlyPayload(parsed, 'importJson');
  if (typeof parsed !== 'object' || parsed === null) {
    throw new PrivacySchemaError('Import JSON must be an object');
  }
  const obj = parsed as { schemaVersion?: unknown; rounds?: unknown };
  if (obj.schemaVersion !== 1 || !Array.isArray(obj.rounds)) {
    throw new PrivacySchemaError(
      'Invalid log export schema: expected schemaVersion 1 and rounds array'
    );
  }
  return obj.rounds.map((raw) => {
    const clean = sanitizeRoundLogRecord(raw);
    if (profileIdOverride) {
      return { ...clean, profileId: profileIdOverride };
    }
    return clean;
  });
}

/**
 * In-memory consent-gated round log store (used in headless Node tests or when IndexedDB is unavailable).
 */
export class InMemoryRoundLogStore implements RoundLogStorageBackend {
  private readonly settings = new Map<string, ConsentSettings>();
  private readonly logs = new Map<string, RoundLogRecord>();

  getConsentSettings(profileId = DEFAULT_PROFILE_ID): Promise<ConsentSettings> {
    const existing = this.settings.get(profileId);
    return Promise.resolve(
      existing ? { ...existing } : createDefaultConsentSettings(profileId)
    );
  }

  setConsentEnabled(
    enabled: boolean,
    profileId = DEFAULT_PROFILE_ID
  ): Promise<ConsentSettings> {
    const next: ConsentSettings = {
      profileId,
      consentEnabled: Boolean(enabled),
      updatedAtMs: Date.now(),
    };
    this.settings.set(profileId, next);
    return Promise.resolve({ ...next });
  }

  async appendRoundLog(record: RoundLogRecord): Promise<boolean> {
    const clean = sanitizeRoundLogRecord(record);
    const consent = await this.getConsentSettings(clean.profileId);
    if (!consent.consentEnabled) {
      return false;
    }
    if (this.logs.size >= MAX_STORED_ROUNDS) {
      const oldestKey = this.logs.keys().next().value;
      if (oldestKey) this.logs.delete(oldestKey);
    }
    this.logs.set(clean.id, clean);
    return true;
  }

  listRoundLogs(profileId?: string): Promise<RoundLogRecord[]> {
    const items = [...this.logs.values()].filter((r) =>
      profileId ? r.profileId === profileId : true
    );
    items.sort((a, b) => a.recordedAtMs - b.recordedAtMs);
    return Promise.resolve(items.map((r) => sanitizeRoundLogRecord(r)));
  }

  async getRoundLogCount(profileId?: string): Promise<number> {
    const items = await this.listRoundLogs(profileId);
    return items.length;
  }

  async exportLogsJson(profileId = DEFAULT_PROFILE_ID, pretty = true): Promise<string> {
    const rounds = await this.listRoundLogs(profileId);
    const payload = buildExportPayload(profileId, rounds);
    return JSON.stringify(payload, null, pretty ? 2 : undefined);
  }

  async importLogsJson(
    rawJson: string,
    options: { profileIdOverride?: string } = {}
  ): Promise<{ importedCount: number; skippedDuplicates: number }> {
    const records = parseImportPayload(rawJson, options.profileIdOverride);
    let importedCount = 0;
    let skippedDuplicates = 0;
    for (const rec of records) {
      if (this.logs.has(rec.id)) {
        skippedDuplicates += 1;
        continue;
      }
      this.logs.set(rec.id, rec);
      importedCount += 1;
    }
    return Promise.resolve({ importedCount, skippedDuplicates });
  }

  clearLogs(profileId?: string): Promise<void> {
    if (!profileId) {
      this.logs.clear();
      return Promise.resolve();
    }
    for (const [id, rec] of this.logs.entries()) {
      if (rec.profileId === profileId) {
        this.logs.delete(id);
      }
    }
    return Promise.resolve();
  }
}

/**
 * IndexedDB-backed consent-gated numeric-only round log store via `idb`.
 * Default consent is OFF (`consentEnabled: false`). Never stores images or video.
 */
export class IndexedDbRoundLogStore implements RoundLogStorageBackend {
  private dbPromise: Promise<IDBPDatabase<RoundLogDbSchema>> | null = null;
  private readonly fallback = new InMemoryRoundLogStore();

  private getDb(): Promise<IDBPDatabase<RoundLogDbSchema>> | null {
    if (typeof indexedDB === 'undefined') {
      return null;
    }
    if (!this.dbPromise) {
      this.dbPromise = openDB<RoundLogDbSchema>(
        ROUND_LOG_DB_NAME,
        ROUND_LOG_DB_VERSION,
        {
          upgrade(db) {
            if (!db.objectStoreNames.contains('settings')) {
              db.createObjectStore('settings', { keyPath: 'profileId' });
            }
            if (!db.objectStoreNames.contains('roundLogs')) {
              const store = db.createObjectStore('roundLogs', { keyPath: 'id' });
              store.createIndex('by-profile', 'profileId');
            }
          },
        }
      );
    }
    return this.dbPromise;
  }

  async getConsentSettings(profileId = DEFAULT_PROFILE_ID): Promise<ConsentSettings> {
    const dbP = this.getDb();
    if (!dbP) return this.fallback.getConsentSettings(profileId);
    try {
      const db = await dbP;
      const row = await db.get('settings', profileId);
      if (!row) return createDefaultConsentSettings(profileId);
      return {
        profileId: String(row.profileId || profileId),
        consentEnabled: Boolean(row.consentEnabled),
        updatedAtMs: Number(row.updatedAtMs) || 0,
      };
    } catch {
      return this.fallback.getConsentSettings(profileId);
    }
  }

  async setConsentEnabled(
    enabled: boolean,
    profileId = DEFAULT_PROFILE_ID
  ): Promise<ConsentSettings> {
    const next: ConsentSettings = {
      profileId,
      consentEnabled: Boolean(enabled),
      updatedAtMs: Date.now(),
    };
    const dbP = this.getDb();
    if (!dbP) {
      return this.fallback.setConsentEnabled(enabled, profileId);
    }
    try {
      const db = await dbP;
      await db.put('settings', next);
      return next;
    } catch {
      return this.fallback.setConsentEnabled(enabled, profileId);
    }
  }

  async appendRoundLog(record: RoundLogRecord): Promise<boolean> {
    const clean = sanitizeRoundLogRecord(record);
    const consent = await this.getConsentSettings(clean.profileId);
    if (!consent.consentEnabled) {
      return false;
    }
    const dbP = this.getDb();
    if (!dbP) {
      // Ensure fallback also knows consent is enabled
      await this.fallback.setConsentEnabled(true, clean.profileId);
      return this.fallback.appendRoundLog(clean);
    }
    try {
      const db = await dbP;
      await db.put('roundLogs', clean);
      return true;
    } catch {
      await this.fallback.setConsentEnabled(true, clean.profileId);
      return this.fallback.appendRoundLog(clean);
    }
  }

  async listRoundLogs(profileId?: string): Promise<RoundLogRecord[]> {
    const dbP = this.getDb();
    if (!dbP) return this.fallback.listRoundLogs(profileId);
    try {
      const db = await dbP;
      const rows = profileId
        ? await db.getAllFromIndex('roundLogs', 'by-profile', profileId)
        : await db.getAll('roundLogs');
      rows.sort((a, b) => a.recordedAtMs - b.recordedAtMs);
      return rows.map((r) => sanitizeRoundLogRecord(r));
    } catch {
      return this.fallback.listRoundLogs(profileId);
    }
  }

  async getRoundLogCount(profileId?: string): Promise<number> {
    const items = await this.listRoundLogs(profileId);
    return items.length;
  }

  async exportLogsJson(profileId = DEFAULT_PROFILE_ID, pretty = true): Promise<string> {
    const rounds = await this.listRoundLogs(profileId);
    const payload = buildExportPayload(profileId, rounds);
    return JSON.stringify(payload, null, pretty ? 2 : undefined);
  }

  async importLogsJson(
    rawJson: string,
    options: { profileIdOverride?: string } = {}
  ): Promise<{ importedCount: number; skippedDuplicates: number }> {
    const records = parseImportPayload(rawJson, options.profileIdOverride);
    const dbP = this.getDb();
    if (!dbP) {
      return this.fallback.importLogsJson(rawJson, options);
    }
    try {
      const db = await dbP;
      let importedCount = 0;
      let skippedDuplicates = 0;
      const tx = db.transaction('roundLogs', 'readwrite');
      for (const rec of records) {
        const existing = await tx.store.get(rec.id);
        if (existing) {
          skippedDuplicates += 1;
          continue;
        }
        await tx.store.put(rec);
        importedCount += 1;
      }
      await tx.done;
      return { importedCount, skippedDuplicates };
    } catch {
      return this.fallback.importLogsJson(rawJson, options);
    }
  }

  async clearLogs(profileId?: string): Promise<void> {
    const dbP = this.getDb();
    if (!dbP) {
      return this.fallback.clearLogs(profileId);
    }
    try {
      const db = await dbP;
      if (!profileId) {
        await db.clear('roundLogs');
        return;
      }
      const tx = db.transaction('roundLogs', 'readwrite');
      const keys = await tx.store.index('by-profile').getAllKeys(profileId);
      for (const k of keys) {
        await tx.store.delete(k);
      }
      await tx.done;
    } catch {
      await this.fallback.clearLogs(profileId);
    }
  }
}
