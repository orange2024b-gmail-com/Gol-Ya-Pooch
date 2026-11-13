import type { FrameFeatures } from '../vision/features';
import type { HandSide } from '../vision/types';

export type TrackingEvent =
  | {
      type: 'CONTACT_START';
      timestampMs: number;
      relativeSec: number;
      minDistanceNorm: number;
    }
  | {
      type: 'CONTACT_END';
      timestampMs: number;
      relativeSec: number;
      durationMs: number;
      partialOpeningsDuringContact: HandSide[];
    }
  | {
      type: 'PARTIAL_OPENING';
      timestampMs: number;
      relativeSec: number;
      side: HandSide;
      strength: number;
      duringContact: boolean;
    }
  | {
      type: 'GOL_SIGHTING';
      timestampMs: number;
      relativeSec: number;
      side: HandSide;
      score: number;
    }
  | {
      type: 'OCCLUSION_START';
      timestampMs: number;
      relativeSec: number;
      side: HandSide | 'BOTH';
    }
  | {
      type: 'OCCLUSION_END';
      timestampMs: number;
      relativeSec: number;
      durationMs: number;
    };

export interface TimelineSummary {
  totalFrames: number;
  contactCount: number;
  totalContactMs: number;
  partialOpeningCount: number;
  sightingCount: number;
  occludedFrames: number;
  lowQualityFrames: number;
  events: TrackingEvent[];
}

export interface EventDetectorConfig {
  minSightingScore: number;
  sightingCooldownMs: number;
}

export const DEFAULT_EVENT_DETECTOR_CONFIG: EventDetectorConfig = {
  minSightingScore: 0.35,
  sightingCooldownMs: 300,
};

/**
 * Converts per-frame kinematic & sighting features into a structured timeline
 * of hiding-phase events (contacts, partial openings, gol sightings, occlusions).
 */
export class EventDetector {
  private readonly config: EventDetectorConfig;
  private startTimestampMs: number | null = null;
  private events: TrackingEvent[] = [];
  private prevInContact = false;
  private contactStartMs = 0;
  private contactOpenings: HandSide[] = [];
  private prevOccluded = false;
  private occlusionStartMs = 0;
  private lastLeftSightingMs = -Infinity;
  private lastRightSightingMs = -Infinity;

  private totalFrames = 0;
  private contactCount = 0;
  private totalContactMs = 0;
  private partialOpeningCount = 0;
  private sightingCount = 0;
  private occludedFrames = 0;
  private lowQualityFrames = 0;

  constructor(config: Partial<EventDetectorConfig> = {}) {
    this.config = { ...DEFAULT_EVENT_DETECTOR_CONFIG, ...config };
  }

  reset(startTimestampMs?: number): void {
    this.startTimestampMs = startTimestampMs ?? null;
    this.events = [];
    this.prevInContact = false;
    this.contactStartMs = 0;
    this.contactOpenings = [];
    this.prevOccluded = false;
    this.occlusionStartMs = 0;
    this.lastLeftSightingMs = -Infinity;
    this.lastRightSightingMs = -Infinity;
    this.totalFrames = 0;
    this.contactCount = 0;
    this.totalContactMs = 0;
    this.partialOpeningCount = 0;
    this.sightingCount = 0;
    this.occludedFrames = 0;
    this.lowQualityFrames = 0;
  }

  private relSec(ts: number): number {
    if (this.startTimestampMs === null) {
      this.startTimestampMs = ts;
    }
    return Math.max(0, (ts - this.startTimestampMs) / 1000);
  }

  update(features: FrameFeatures): TrackingEvent[] {
    const ts = features.timestampMs;
    const relSec = this.relSec(ts);
    const emitted: TrackingEvent[] = [];
    this.totalFrames += 1;

    if (features.quality.anyOccluded) {
      this.occludedFrames += 1;
    }
    if (features.quality.lowQuality) {
      this.lowQualityFrames += 1;
    }

    // Occlusion start / end
    if (!this.prevOccluded && features.quality.anyOccluded) {
      this.prevOccluded = true;
      this.occlusionStartMs = ts;
      const side: HandSide | 'BOTH' =
        features.quality.leftOccluded && features.quality.rightOccluded
          ? 'BOTH'
          : features.quality.leftOccluded
            ? 'LEFT'
            : 'RIGHT';
      const ev: TrackingEvent = {
        type: 'OCCLUSION_START',
        timestampMs: ts,
        relativeSec: relSec,
        side,
      };
      this.events.push(ev);
      emitted.push(ev);
    } else if (this.prevOccluded && !features.quality.anyOccluded) {
      this.prevOccluded = false;
      const ev: TrackingEvent = {
        type: 'OCCLUSION_END',
        timestampMs: ts,
        relativeSec: relSec,
        durationMs: Math.max(0, ts - this.occlusionStartMs),
      };
      this.events.push(ev);
      emitted.push(ev);
    }

    // Inter-hand contact start / end
    const inContact = features.interHand.inContact;
    if (!this.prevInContact && inContact) {
      this.prevInContact = true;
      this.contactStartMs = ts;
      this.contactOpenings = [];
      this.contactCount += 1;
      const ev: TrackingEvent = {
        type: 'CONTACT_START',
        timestampMs: ts,
        relativeSec: relSec,
        minDistanceNorm: features.interHand.minLandmarkDistanceNorm,
      };
      this.events.push(ev);
      emitted.push(ev);
    } else if (this.prevInContact && !inContact) {
      this.prevInContact = false;
      const durationMs = Math.max(0, ts - this.contactStartMs);
      this.totalContactMs += durationMs;
      const ev: TrackingEvent = {
        type: 'CONTACT_END',
        timestampMs: ts,
        relativeSec: relSec,
        durationMs,
        partialOpeningsDuringContact: [...this.contactOpenings],
      };
      this.events.push(ev);
      emitted.push(ev);
    }

    // Partial openings
    for (const handFeat of [features.left, features.right]) {
      if (handFeat.partialOpening) {
        this.partialOpeningCount += 1;
        if (inContact && !this.contactOpenings.includes(handFeat.side)) {
          this.contactOpenings.push(handFeat.side);
        }
        const ev: TrackingEvent = {
          type: 'PARTIAL_OPENING',
          timestampMs: ts,
          relativeSec: relSec,
          side: handFeat.side,
          strength: handFeat.partialOpeningStrength,
          duringContact: inContact,
        };
        this.events.push(ev);
        emitted.push(ev);
      }
    }

    // Gol sightings (debounced per hand)
    if (
      features.left.golSightingScore >= this.config.minSightingScore &&
      ts - this.lastLeftSightingMs >= this.config.sightingCooldownMs
    ) {
      this.lastLeftSightingMs = ts;
      this.sightingCount += 1;
      const ev: TrackingEvent = {
        type: 'GOL_SIGHTING',
        timestampMs: ts,
        relativeSec: relSec,
        side: 'LEFT',
        score: features.left.golSightingScore,
      };
      this.events.push(ev);
      emitted.push(ev);
    }
    if (
      features.right.golSightingScore >= this.config.minSightingScore &&
      ts - this.lastRightSightingMs >= this.config.sightingCooldownMs
    ) {
      this.lastRightSightingMs = ts;
      this.sightingCount += 1;
      const ev: TrackingEvent = {
        type: 'GOL_SIGHTING',
        timestampMs: ts,
        relativeSec: relSec,
        side: 'RIGHT',
        score: features.right.golSightingScore,
      };
      this.events.push(ev);
      emitted.push(ev);
    }

    return emitted;
  }

  getTimeline(): TrackingEvent[] {
    return [...this.events];
  }

  getSummary(): TimelineSummary {
    return {
      totalFrames: this.totalFrames,
      contactCount: this.contactCount,
      totalContactMs:
        this.totalContactMs +
        (this.prevInContact && this.events.length > 0
          ? Math.max(0, this.events[this.events.length - 1]!.timestampMs - this.contactStartMs)
          : 0),
      partialOpeningCount: this.partialOpeningCount,
      sightingCount: this.sightingCount,
      occludedFrames: this.occludedFrames,
      lowQualityFrames: this.lowQualityFrames,
      events: [...this.events],
    };
  }
}
