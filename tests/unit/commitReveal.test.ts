import { describe, expect, it } from 'vitest';
import {
  buildCommitPreimage,
  createCommitEnvelope,
  formatChoiceToken,
  formatCommitFingerprint,
  generateSaltHex,
  sha256Hex,
  verifyCommitReveal,
} from '../../src/game/commitReveal';

describe('Cryptographic Commit–Reveal (src/game/commitReveal.ts)', () => {
  it('generates 16-byte random salts, SHA-256 commitments, and verifies valid reveals', async () => {
    const salt = generateSaltHex(16);
    expect(salt).toMatch(/^[0-9a-f]{32}$/);

    const env2 = await createCommitEnvelope(1, 2);
    expect(env2.secret.choice).toBe('RIGHT');
    expect(env2.secret.saltHex).toMatch(/^[0-9a-f]{32}$/);
    expect(env2.publicInfo.commitHex).toMatch(/^[0-9a-f]{64}$/);
    expect(env2.publicInfo.fingerprint).toBe(
      formatCommitFingerprint(env2.publicInfo.commitHex, 12)
    );
    expect(env2.secret.preimage).toBe(
      buildCommitPreimage(env2.secret.choice, env2.secret.saltHex)
    );

    const directHash = await sha256Hex(`${env2.secret.choice}:${env2.secret.saltHex}`);
    expect(directHash).toBe(env2.publicInfo.commitHex);

    const verified = await verifyCommitReveal(
      env2.publicInfo.commitHex,
      env2.secret.choice,
      env2.secret.saltHex
    );
    expect(verified.valid).toBe(true);
    expect(verified.statusFa).toBe('✓ تأیید شد');
    expect(verified.recomputedCommitHex).toBe(env2.publicInfo.commitHex);
  });

  it('supports 4-hand and 6-hand choice tokens and deterministic custom salt verification', async () => {
    expect(formatChoiceToken(0, 2)).toBe('LEFT');
    expect(formatChoiceToken(1, 2)).toBe('RIGHT');
    expect(formatChoiceToken(2, 4)).toBe('HAND_3');
    expect(formatChoiceToken(5, 6)).toBe('HAND_6');

    const fixedSalt = new Uint8Array(16).fill(0xab);
    const env = await createCommitEnvelope(3, 6, fixedSalt);
    expect(env.secret.choice).toBe('HAND_4');
    expect(env.secret.saltHex).toBe('ab'.repeat(16));

    const res = await verifyCommitReveal(
      env.publicInfo.commitHex,
      env.secret.choice,
      env.secret.saltHex
    );
    expect(res.valid).toBe(true);
  });

  it('detects tampering with choice, salt, or commit hash', async () => {
    const env = await createCommitEnvelope(0, 4);

    // 1. Tampered choice
    const wrongChoice = await verifyCommitReveal(
      env.publicInfo.commitHex,
      'HAND_2',
      env.secret.saltHex
    );
    expect(wrongChoice.valid).toBe(false);
    expect(wrongChoice.reason).toBe('HASH_MISMATCH');

    // 2. Tampered salt
    const tamperedSalt =
      (env.secret.saltHex[0] === '0' ? '1' : '0') + env.secret.saltHex.slice(1);
    const wrongSalt = await verifyCommitReveal(
      env.publicInfo.commitHex,
      env.secret.choice,
      tamperedSalt
    );
    expect(wrongSalt.valid).toBe(false);
    expect(wrongSalt.reason).toBe('HASH_MISMATCH');

    // 3. Tampered commit hash
    const tamperedCommit =
      (env.publicInfo.commitHex[0] === 'a' ? 'b' : 'a') +
      env.publicInfo.commitHex.slice(1);
    const wrongCommit = await verifyCommitReveal(
      tamperedCommit,
      env.secret.choice,
      env.secret.saltHex
    );
    expect(wrongCommit.valid).toBe(false);
    expect(wrongCommit.reason).toBe('HASH_MISMATCH');

    // 4. Malformed inputs
    expect((await verifyCommitReveal('not-a-hash', 'LEFT', env.secret.saltHex)).valid).toBe(
      false
    );
    expect((await verifyCommitReveal(env.publicInfo.commitHex, '', env.secret.saltHex)).valid).toBe(
      false
    );
    expect((await verifyCommitReveal(env.publicInfo.commitHex, 'LEFT', 'short')).valid).toBe(
      false
    );
  });
});
