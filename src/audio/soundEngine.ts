export type SoundCue =
  | 'tap'
  | 'pinchConfirm'
  | 'shuffle'
  | 'reveal'
  | 'win'
  | 'lose'
  | 'void';

const MUTE_STORAGE_KEY = 'gol-ya-pooch:audio-muted';

export interface StorageLike {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

function getSafeStorage(override?: StorageLike): StorageLike | null {
  if (override) return override;
  try {
    if (typeof globalThis.localStorage !== 'undefined') {
      return globalThis.localStorage;
    }
  } catch {
    return null;
  }
  return null;
}

type AudioContextCtor = new () => AudioContext;

function getAudioContextCtor(override?: AudioContextCtor | null): AudioContextCtor | null {
  if (override !== undefined) return override;
  const w = globalThis as typeof globalThis & {
    AudioContext?: AudioContextCtor;
    webkitAudioContext?: AudioContextCtor;
  };
  return w.AudioContext ?? w.webkitAudioContext ?? null;
}

export class ProceduralSoundEngine {
  private muted: boolean;
  private ctx: AudioContext | null = null;
  private readonly storage: StorageLike | null;
  private readonly audioCtorOverride: AudioContextCtor | null | undefined;

  constructor(options: {
    storage?: StorageLike;
    audioContextCtor?: AudioContextCtor | null;
    defaultMuted?: boolean;
  } = {}) {
    this.storage = getSafeStorage(options.storage);
    this.audioCtorOverride = options.audioContextCtor;
    this.muted = this.readInitialMuted(options.defaultMuted ?? false);
  }

  private readInitialMuted(fallback: boolean): boolean {
    try {
      const saved = this.storage?.getItem(MUTE_STORAGE_KEY);
      if (saved === 'true') return true;
      if (saved === 'false') return false;
    } catch {
      // Ignore storage access errors in restricted contexts
    }
    return fallback;
  }

  isMuted(): boolean {
    return this.muted;
  }

  setMuted(muted: boolean): void {
    this.muted = Boolean(muted);
    try {
      this.storage?.setItem(MUTE_STORAGE_KEY, this.muted ? 'true' : 'false');
    } catch {
      // Ignore storage write errors
    }
  }

  toggleMute(): boolean {
    this.setMuted(!this.muted);
    return this.muted;
  }

  private ensureContext(): AudioContext | null {
    if (this.muted) return null;
    if (!this.ctx) {
      const Ctor = getAudioContextCtor(this.audioCtorOverride);
      if (!Ctor) return null;
      try {
        this.ctx = new Ctor();
      } catch {
        return null;
      }
    }
    try {
      if (this.ctx.state === 'suspended' && typeof this.ctx.resume === 'function') {
        void this.ctx.resume().catch(() => {});
      }
    } catch {
      // Ignore resume errors before user gesture
    }
    return this.ctx;
  }

  private playTone(
    ctx: AudioContext,
    options: {
      freqStart: number;
      freqEnd?: number;
      startOffsetSec?: number;
      durationSec: number;
      type: OscillatorType;
      gainPeak: number;
    }
  ): void {
    const now = ctx.currentTime + (options.startOffsetSec ?? 0);
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.type = options.type;
    osc.frequency.setValueAtTime(options.freqStart, now);
    if (options.freqEnd !== undefined && options.freqEnd !== options.freqStart) {
      osc.frequency.exponentialRampToValueAtTime(
        Math.max(40, options.freqEnd),
        now + options.durationSec
      );
    }

    gain.gain.setValueAtTime(0.0001, now);
    gain.gain.linearRampToValueAtTime(options.gainPeak, now + 0.01);
    gain.gain.exponentialRampToValueAtTime(0.0001, now + options.durationSec);

    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.start(now);
    osc.stop(now + options.durationSec + 0.015);
  }

  play(cue: SoundCue): boolean {
    if (this.muted) return false;
    const ctx = this.ensureContext();
    if (!ctx) return false;

    try {
      switch (cue) {
        case 'tap':
          this.playTone(ctx, {
            freqStart: 640,
            freqEnd: 420,
            durationSec: 0.045,
            type: 'triangle',
            gainPeak: 0.06,
          });
          return true;

        case 'pinchConfirm':
          this.playTone(ctx, {
            freqStart: 523.25,
            durationSec: 0.11,
            type: 'sine',
            gainPeak: 0.09,
          });
          this.playTone(ctx, {
            freqStart: 783.99,
            startOffsetSec: 0.085,
            durationSec: 0.17,
            type: 'triangle',
            gainPeak: 0.1,
          });
          return true;

        case 'shuffle':
          this.playTone(ctx, {
            freqStart: 280,
            freqEnd: 440,
            durationSec: 0.12,
            type: 'sine',
            gainPeak: 0.055,
          });
          this.playTone(ctx, {
            freqStart: 420,
            freqEnd: 290,
            startOffsetSec: 0.11,
            durationSec: 0.12,
            type: 'sine',
            gainPeak: 0.05,
          });
          return true;

        case 'reveal':
          this.playTone(ctx, {
            freqStart: 440,
            durationSec: 0.18,
            type: 'triangle',
            gainPeak: 0.08,
          });
          this.playTone(ctx, {
            freqStart: 554.37,
            startOffsetSec: 0.05,
            durationSec: 0.2,
            type: 'sine',
            gainPeak: 0.075,
          });
          this.playTone(ctx, {
            freqStart: 659.25,
            startOffsetSec: 0.1,
            durationSec: 0.24,
            type: 'sine',
            gainPeak: 0.08,
          });
          return true;

        case 'win':
          this.playTone(ctx, {
            freqStart: 523.25,
            durationSec: 0.13,
            type: 'triangle',
            gainPeak: 0.1,
          });
          this.playTone(ctx, {
            freqStart: 659.25,
            startOffsetSec: 0.09,
            durationSec: 0.14,
            type: 'triangle',
            gainPeak: 0.1,
          });
          this.playTone(ctx, {
            freqStart: 783.99,
            startOffsetSec: 0.18,
            durationSec: 0.16,
            type: 'triangle',
            gainPeak: 0.11,
          });
          this.playTone(ctx, {
            freqStart: 1046.5,
            startOffsetSec: 0.28,
            durationSec: 0.26,
            type: 'sine',
            gainPeak: 0.11,
          });
          return true;

        case 'lose':
          this.playTone(ctx, {
            freqStart: 360,
            freqEnd: 240,
            durationSec: 0.18,
            type: 'sine',
            gainPeak: 0.08,
          });
          this.playTone(ctx, {
            freqStart: 250,
            freqEnd: 185,
            startOffsetSec: 0.14,
            durationSec: 0.22,
            type: 'triangle',
            gainPeak: 0.075,
          });
          return true;

        case 'void':
          this.playTone(ctx, {
            freqStart: 310,
            durationSec: 0.09,
            type: 'sine',
            gainPeak: 0.06,
          });
          this.playTone(ctx, {
            freqStart: 260,
            startOffsetSec: 0.11,
            durationSec: 0.1,
            type: 'sine',
            gainPeak: 0.055,
          });
          return true;
      }
    } catch {
      return false;
    }
  }
}
