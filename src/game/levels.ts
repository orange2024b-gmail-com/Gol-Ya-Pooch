import {
  CLASSIC_1V1_RULES,
  TRADITIONAL_TEAM_RULES,
  loadRuleSet,
  type RuleSetDefinition,
} from './rules/engine';

export type LevelId = 1 | 2 | 3 | 4 | 5;

export type ComputerHidingStyle =
  | 'simple-shuffle'
  | 'faster-single-feint'
  | 'multi-feint-uneven-tempo'
  | 'team-misdirection'
  | 'master-adaptive-balanced';

export interface LevelAnimationProfile {
  shuffleSteps: number;
  feintCount: number;
  unevenTempo: boolean;
  misdirectionSwaps: number;
  shuffleDurationMs: number;
  speedMultiplier: number;
}

export interface LevelDefinition {
  id: LevelId;
  nameFa: string;
  nameEn: string;
  subtitleFa: string;
  humanHideDurationMs: number;
  humanGuessDurationMs: number;
  minHandContactsRequired: number;
  handsCountDefault: number;
  supportedHandsCounts: readonly number[];
  ruleSetIds: readonly ('classic-1v1' | 'traditional-team')[];
  bestOfMatches: 1 | 3;
  seriesRuleOrder: readonly ('classic-1v1' | 'traditional-team')[];
  hidingStyle: ComputerHidingStyle;
  allowBiasExploitation: boolean;
  maxExploitDeviation: number;
  animation: LevelAnimationProfile;
  descriptionFa: string;
}

export const LEVELS: readonly LevelDefinition[] = [
  {
    id: 1,
    nameFa: 'تازه‌کار',
    nameEn: 'Beginner',
    subtitleFa: '۲ دست، ۱۵ ثانیه مخفی‌سازی، حرکت ساده رایانه',
    humanHideDurationMs: 15_000,
    humanGuessDurationMs: 20_000,
    minHandContactsRequired: 0,
    handsCountDefault: 2,
    supportedHandsCounts: [2],
    ruleSetIds: ['classic-1v1'],
    bestOfMatches: 1,
    seriesRuleOrder: ['classic-1v1'],
    hidingStyle: 'simple-shuffle',
    allowBiasExploitation: false,
    maxExploitDeviation: 0,
    animation: {
      shuffleSteps: 3,
      feintCount: 0,
      unevenTempo: false,
      misdirectionSwaps: 0,
      shuffleDurationMs: 2200,
      speedMultiplier: 1.0,
    },
    descriptionFa:
      'قوانین کلاسیک ۱ به ۱ با ۲ دست. زمان مخفی‌سازی ۱۵ ثانیه بدون محدودیت حرکت؛ رایانه جابه‌جایی ساده و یکنواخت انجام می‌دهد.',
  },
  {
    id: 2,
    nameFa: 'آماتور',
    nameEn: 'Amateur',
    subtitleFa: '۲ دست، ۱۰ ثانیه مخفی‌سازی، جابه‌جایی سریع‌تر + ۱ فریب',
    humanHideDurationMs: 10_000,
    humanGuessDurationMs: 16_000,
    minHandContactsRequired: 0,
    handsCountDefault: 2,
    supportedHandsCounts: [2],
    ruleSetIds: ['classic-1v1'],
    bestOfMatches: 1,
    seriesRuleOrder: ['classic-1v1'],
    hidingStyle: 'faster-single-feint',
    allowBiasExploitation: false,
    maxExploitDeviation: 0,
    animation: {
      shuffleSteps: 5,
      feintCount: 1,
      unevenTempo: false,
      misdirectionSwaps: 0,
      shuffleDurationMs: 2600,
      speedMultiplier: 1.35,
    },
    descriptionFa:
      'قوانین کلاسیک ۱ به ۱ با ۱۰ ثانیه زمان مخفی‌سازی. رایانه سریع‌تر دست‌ها را جابه‌جا می‌کند و یک حرکت فریب (Feint) انجام می‌دهد.',
  },
  {
    id: 3,
    nameFa: 'حرفه‌ای',
    nameEn: 'Pro',
    subtitleFa: '۲ دست، ۷ ثانیه، الزام حداقل ۱ تماس دست، چند فریب و ریتم نامنظم',
    humanHideDurationMs: 7_000,
    humanGuessDurationMs: 14_000,
    minHandContactsRequired: 1,
    handsCountDefault: 2,
    supportedHandsCounts: [2],
    ruleSetIds: ['classic-1v1'],
    bestOfMatches: 1,
    seriesRuleOrder: ['classic-1v1'],
    hidingStyle: 'multi-feint-uneven-tempo',
    allowBiasExploitation: false,
    maxExploitDeviation: 0,
    animation: {
      shuffleSteps: 6,
      feintCount: 3,
      unevenTempo: true,
      misdirectionSwaps: 0,
      shuffleDurationMs: 3200,
      speedMultiplier: 1.65,
    },
    descriptionFa:
      'زمان مخفی‌سازی ۷ ثانیه و در حین مخفی‌سازی حداقل یک بار تماس دست‌ها الزامی است. رایانه با چند فریب و سرعت نامنظم پنهان‌کاری می‌کند.',
  },
  {
    id: 4,
    nameFa: 'تیمی',
    nameEn: 'Team',
    subtitleFa: 'قوانین تیمی (۴ تا ۶ دست)، پوچ و گل، زمان‌بندی‌شده با انحراف توجه',
    humanHideDurationMs: 10_000,
    humanGuessDurationMs: 22_000,
    minHandContactsRequired: 0,
    handsCountDefault: 4,
    supportedHandsCounts: [4, 6],
    ruleSetIds: ['traditional-team'],
    bestOfMatches: 1,
    seriesRuleOrder: ['traditional-team'],
    hidingStyle: 'team-misdirection',
    allowBiasExploitation: false,
    maxExploitDeviation: 0,
    animation: {
      shuffleSteps: 7,
      feintCount: 3,
      unevenTempo: true,
      misdirectionSwaps: 3,
      shuffleDurationMs: 3500,
      speedMultiplier: 1.5,
    },
    descriptionFa:
      'بازی تیمی سنتی با ۴ یا ۶ دست متحرک و حداکثر ۳ حدس (فراخوانی «پوچ» و «گل»). رایانه از حرکات انحرافی بین چند دست استفاده می‌کند.',
  },
  {
    id: 5,
    nameFa: 'استاد گل‌گیر',
    nameEn: 'Master Gol-Gir',
    subtitleFa: 'بهترین از ۳ مسابقه (کلاسیک + تیمی)، ۵ ثانیه مخفی‌سازی، پنهان‌کاری تطبیقی متعادل',
    humanHideDurationMs: 5_000,
    humanGuessDurationMs: 15_000,
    minHandContactsRequired: 0,
    handsCountDefault: 2,
    supportedHandsCounts: [2, 4, 6],
    ruleSetIds: ['classic-1v1', 'traditional-team'],
    bestOfMatches: 3,
    seriesRuleOrder: ['classic-1v1', 'traditional-team', 'classic-1v1'],
    hidingStyle: 'master-adaptive-balanced',
    allowBiasExploitation: true,
    maxExploitDeviation: 0.18,
    animation: {
      shuffleSteps: 8,
      feintCount: 4,
      unevenTempo: true,
      misdirectionSwaps: 4,
      shuffleDurationMs: 3600,
      speedMultiplier: 1.85,
    },
    descriptionFa:
      'نبرد نهایی بهترین از ۳ مسابقه (ترکیبی از کلاسیک دو نفره و تیمی) با ۵ ثانیه زمان مخفی‌سازی. رایانه ضمن حفظ تصادفی‌بودن متعادل، از عادت‌های حدس شما بهره محدود می‌برد.',
  },
] as const;

export function getLevelDefinition(levelId: LevelId): LevelDefinition {
  const found = LEVELS.find((l) => l.id === levelId);
  if (!found) {
    throw new RangeError(`Unknown levelId: ${String(levelId)}`);
  }
  return found;
}

export function buildRuleSetForLevel(
  levelId: LevelId,
  options: {
    seriesMatchIndex?: number;
    targetScore?: number;
    teamHandsCount?: 4 | 6;
  } = {}
): RuleSetDefinition {
  const level = getLevelDefinition(levelId);
  const seriesIdx = Math.max(
    0,
    Math.min(level.seriesRuleOrder.length - 1, options.seriesMatchIndex ?? 0)
  );
  const ruleKey = level.seriesRuleOrder[seriesIdx] ?? 'classic-1v1';
  const baseRule = ruleKey === 'traditional-team' ? TRADITIONAL_TEAM_RULES : CLASSIC_1V1_RULES;

  const handsCount =
    ruleKey === 'traditional-team'
      ? (options.teamHandsCount ?? (level.id === 4 ? (level.handsCountDefault as 4 | 6) : 4))
      : 2;

  return loadRuleSet(baseRule, {
    ...(options.targetScore !== undefined ? { targetScore: options.targetScore } : {}),
    handsCount,
    hidingTimeoutMs: level.humanHideDurationMs,
  });
}

export function isLevelUnlocked(levelId: LevelId, unlockedMaxLevel: LevelId): boolean {
  return levelId <= unlockedMaxLevel;
}

export function computeNextUnlockedLevel(
  currentUnlockedMax: LevelId,
  wonLevelId: LevelId
): LevelId {
  if (wonLevelId < 5 && wonLevelId >= currentUnlockedMax) {
    return (wonLevelId + 1) as LevelId;
  }
  return currentUnlockedMax;
}
