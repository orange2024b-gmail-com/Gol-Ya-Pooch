import { describe, expect, it, vi } from 'vitest';
import {
  ProceduralSoundEngine,
  type SoundCue,
  type StorageLike,
} from '../../src/audio/soundEngine';

function makeStorage(initial: Record<string, string> = {}): StorageLike {
  const map = new Map<string, string>(Object.entries(initial));
  return {
    getItem: (k) => map.get(k) ?? null,
    setItem: (k, v) => {
      map.set(k, v);
    },
  };
}

describe('Procedural Web Audio Sound Engine (src/audio/)', () => {
  it('persists mute state and safely returns false without throwing when AudioContext is unavailable', () => {
    const storage = makeStorage();
    const engine = new ProceduralSoundEngine({
      storage,
      audioContextCtor: null,
    });

    expect(engine.isMuted()).toBe(false);
    expect(engine.play('tap')).toBe(false);

    expect(engine.toggleMute()).toBe(true);
    expect(engine.isMuted()).toBe(true);
    expect(storage.getItem('gol-ya-pooch:audio-muted')).toBe('true');

    const reloaded = new ProceduralSoundEngine({
      storage,
      audioContextCtor: null,
    });
    expect(reloaded.isMuted()).toBe(true);
  });

  it('never creates AudioContext or oscillators when muted, and synthesizes all cues when unmuted', () => {
    const startSpy = vi.fn();
    const stopSpy = vi.fn();
    const ctorSpy = vi.fn();

    class MockAudioContext {
      state: AudioContextState = 'running';
      currentTime = 1.0;
      destination = {} as AudioDestinationNode;

      constructor() {
        ctorSpy();
      }

      resume = vi.fn().mockResolvedValue(undefined);

      createOscillator() {
        return {
          type: 'sine' as OscillatorType,
          frequency: {
            setValueAtTime: vi.fn(),
            exponentialRampToValueAtTime: vi.fn(),
          },
          connect: vi.fn(),
          start: startSpy,
          stop: stopSpy,
        } as unknown as OscillatorNode;
      }

      createGain() {
        return {
          gain: {
            setValueAtTime: vi.fn(),
            linearRampToValueAtTime: vi.fn(),
            exponentialRampToValueAtTime: vi.fn(),
          },
          connect: vi.fn(),
        } as unknown as GainNode;
      }
    }

    const storage = makeStorage({ 'gol-ya-pooch:audio-muted': 'true' });
    const engine = new ProceduralSoundEngine({
      storage,
      audioContextCtor: MockAudioContext as unknown as new () => AudioContext,
    });

    expect(engine.play('win')).toBe(false);
    expect(ctorSpy).not.toHaveBeenCalled();

    engine.setMuted(false);
    const cues: SoundCue[] = ['tap', 'pinchConfirm', 'shuffle', 'reveal', 'win', 'lose', 'void'];
    for (const cue of cues) {
      expect(engine.play(cue)).toBe(true);
    }

    expect(ctorSpy).toHaveBeenCalledTimes(1);
    expect(startSpy.mock.calls.length).toBeGreaterThanOrEqual(cues.length);
    expect(stopSpy.mock.calls.length).toBe(startSpy.mock.calls.length);
  });
});
