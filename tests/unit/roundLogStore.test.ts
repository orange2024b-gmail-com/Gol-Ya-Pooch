import { describe, expect, it } from 'vitest';
import {
  InMemoryRoundLogStore,
  PrivacySchemaError,
  assertNumericOnlyPayload,
  type RoundLogRecord,
} from '../../src/storage/roundLogStore';
import {
  TEMPORAL_FEATURE_DIM,
  TEMPORAL_SEQ_LEN,
} from '../../src/tracking/temporalFeatures';

function makeValidRoundLog(id = 'rl-1', profileId = 'player-1'): RoundLogRecord {
  return {
    schemaVersion: 1,
    id,
    profileId,
    recordedAtMs: 1_700_000_000_000,
    levelId: 2,
    ruleSetId: 'classic-1v1',
    initialShownHand: 'LEFT',
    resampledSequence: Array.from({ length: TEMPORAL_SEQ_LEN }, () =>
      new Array<number>(TEMPORAL_FEATURE_DIM).fill(0.1)
    ),
    rawFrameFeatures: [
      { tMs: 0, v: new Array<number>(TEMPORAL_FEATURE_DIM).fill(0.1) },
      { tMs: 100, v: new Array<number>(TEMPORAL_FEATURE_DIM).fill(0.2) },
    ],
    eventTimeline: [
      { type: 'CONTACT_START', timestampMs: 1200, relativeSec: 1.2, minDistanceNorm: 0.31 },
    ],
    beliefTrace: [
      {
        timestampMs: 0,
        pLeft: 0.99,
        pRight: 0.01,
        confidence: 0.98,
        mostLikely: 'LEFT',
        inContact: false,
        lowQuality: false,
      },
    ],
    finalGuess: {
      guess: 'RIGHT',
      probability: 0.87,
      pLeft: 0.13,
      hmmPLeft: 0.25,
      temporalPLeft: 0.08,
      behaviorPLeft: 0.5,
      dominantSource: 'TRACKING',
    },
    verifiedGroundTruth: 'RIGHT',
    timings: {
      hidingDurationMs: 3200,
      frameCount: 96,
      sampleRateHz: 10,
      temporalInferenceMs: 0.65,
      totalProcessingMs: 4.2,
    },
  };
}

describe('Consent-gated numeric round logging (src/storage/roundLogStore.ts)', () => {
  it('defaults consent to false and refuses to persist logs until opted in', async () => {
    const store = new InMemoryRoundLogStore();
    const initialConsent = await store.getConsentSettings('player-1');
    expect(initialConsent.consentEnabled).toBe(false);

    const savedWhenOff = await store.appendRoundLog(makeValidRoundLog('rl-off', 'player-1'));
    expect(savedWhenOff).toBe(false);
    expect(await store.getRoundLogCount('player-1')).toBe(0);

    await store.setConsentEnabled(true, 'player-1');
    const savedWhenOn = await store.appendRoundLog(makeValidRoundLog('rl-on', 'player-1'));
    expect(savedWhenOn).toBe(true);
    expect(await store.getRoundLogCount('player-1')).toBe(1);
  });

  it('exports valid numeric JSON, imports cleanly without duplicates, and clears logs', async () => {
    const store = new InMemoryRoundLogStore();
    await store.setConsentEnabled(true, 'player-1');
    await store.appendRoundLog(makeValidRoundLog('rl-1', 'player-1'));
    await store.appendRoundLog(makeValidRoundLog('rl-2', 'player-1'));

    const exportedJson = await store.exportLogsJson('player-1', false);
    const parsed = JSON.parse(exportedJson) as {
      schemaVersion: number;
      consentEnabled: boolean;
      privacyNotice: string;
      rounds: RoundLogRecord[];
    };
    expect(parsed.schemaVersion).toBe(1);
    expect(parsed.consentEnabled).toBe(true);
    expect(parsed.privacyNotice).toContain('no-images-no-video');
    expect(parsed.rounds.length).toBe(2);

    const store2 = new InMemoryRoundLogStore();
    const res1 = await store2.importLogsJson(exportedJson);
    expect(res1.importedCount).toBe(2);
    expect(res1.skippedDuplicates).toBe(0);

    const res2 = await store2.importLogsJson(exportedJson);
    expect(res2.importedCount).toBe(0);
    expect(res2.skippedDuplicates).toBe(2);

    await store2.clearLogs('player-1');
    expect(await store2.getRoundLogCount('player-1')).toBe(0);
  });

  it('rejects any image, video, pixel array, canvas, or base64 data URI at runtime and on import', async () => {
    const store = new InMemoryRoundLogStore();
    await store.setConsentEnabled(true, 'player-1');

    expect(() =>
      assertNumericOnlyPayload({ frameImage: 'data:image/png;base64,AAAA' })
    ).toThrow(PrivacySchemaError);
    expect(() =>
      assertNumericOnlyPayload({ nested: { videoBlob: 'blob:http://localhost/123' } })
    ).toThrow(PrivacySchemaError);
    expect(() =>
      assertNumericOnlyPayload({ pixels: new Uint8ClampedArray([255, 0, 0, 255]) })
    ).toThrow(PrivacySchemaError);
    expect(() =>
      assertNumericOnlyPayload({ note: 'data:video/webm;base64,BBBB' })
    ).toThrow(PrivacySchemaError);

    const badRecord = {
      ...makeValidRoundLog('rl-bad', 'player-1'),
      imageData: [1, 2, 3],
    };
    await expect(
      store.appendRoundLog(badRecord as unknown as RoundLogRecord)
    ).rejects.toThrow(PrivacySchemaError);

    const badExportJson = JSON.stringify({
      schemaVersion: 1,
      rounds: [{ ...makeValidRoundLog('rl-bad2', 'player-1'), screenshotUrl: 'data:image/jpeg;base64,abc' }],
    });
    await expect(store.importLogsJson(badExportJson)).rejects.toThrow(PrivacySchemaError);
  });
});
