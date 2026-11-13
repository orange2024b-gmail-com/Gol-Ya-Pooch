export interface RuleSetScoring {
  correctGolNoWrongCalls: number;
  correctGolAfterPoochCalls: number;
  wrongGuessOrOutOfGuessesHiderPoints: number;
  voidRoundPoints: 0;
}

export interface RuleSetTurnPolicy {
  passTurnOnCorrectGuess: boolean;
  keepTurnOnWrongGuess: boolean;
}

export interface RuleSetTimeoutsMs {
  showGol: number;
  hiding: number;
  guessDisplay: number;
  revealVerify: number;
}

export interface RuleSetVoidConditions {
  maxHandLostMs: number;
  voidOnRevealNeither: boolean;
  voidOnRevealBoth: boolean;
  voidOnTimeout: boolean;
}

export interface RuleSetDefinition {
  id: string;
  version: number;
  nameFa: string;
  nameEn: string;
  descriptionFa: string;
  handsCount: number;
  targetScore: number;
  maxGuessesPerRound: number;
  allowPoochCalls: boolean;
  scoring: RuleSetScoring;
  turnPolicy: RuleSetTurnPolicy;
  timeoutsMs: RuleSetTimeoutsMs;
  voidConditions: RuleSetVoidConditions;
}

export class RuleSchemaError extends Error {
  constructor(message: string) {
    super(`Invalid rule set schema: ${message}`);
    this.name = 'RuleSchemaError';
  }
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

function assertNonEmptyString(obj: Record<string, unknown>, key: string): string {
  const val = obj[key];
  if (typeof val !== 'string' || val.trim().length === 0) {
    throw new RuleSchemaError(`"${key}" must be a non-empty string`);
  }
  return val;
}

function assertIntRange(
  obj: Record<string, unknown>,
  key: string,
  min: number,
  max: number
): number {
  const val = obj[key];
  if (typeof val !== 'number' || !Number.isInteger(val) || val < min || val > max) {
    throw new RuleSchemaError(`"${key}" must be an integer in [${min}, ${max}]`);
  }
  return val;
}

function assertBoolean(obj: Record<string, unknown>, key: string): boolean {
  const val = obj[key];
  if (typeof val !== 'boolean') {
    throw new RuleSchemaError(`"${key}" must be a boolean`);
  }
  return val;
}

function assertObject(obj: Record<string, unknown>, key: string): Record<string, unknown> {
  const val = obj[key];
  if (!isRecord(val)) {
    throw new RuleSchemaError(`"${key}" must be an object`);
  }
  return val;
}

export function validateRuleSet(raw: unknown): RuleSetDefinition {
  if (!isRecord(raw)) {
    throw new RuleSchemaError('root must be a JSON object');
  }

  const id = assertNonEmptyString(raw, 'id');
  const version = assertIntRange(raw, 'version', 1, 100);
  const nameFa = assertNonEmptyString(raw, 'nameFa');
  const nameEn = assertNonEmptyString(raw, 'nameEn');
  const descriptionFa = assertNonEmptyString(raw, 'descriptionFa');
  const handsCount = assertIntRange(raw, 'handsCount', 2, 8);
  const targetScore = assertIntRange(raw, 'targetScore', 1, 50);
  const maxGuessesPerRound = assertIntRange(raw, 'maxGuessesPerRound', 1, 8);
  const allowPoochCalls = assertBoolean(raw, 'allowPoochCalls');

  const scoringObj = assertObject(raw, 'scoring');
  const correctGolNoWrongCalls = assertIntRange(scoringObj, 'correctGolNoWrongCalls', 0, 10);
  const correctGolAfterPoochCalls = assertIntRange(scoringObj, 'correctGolAfterPoochCalls', 0, 10);
  const wrongGuessOrOutOfGuessesHiderPoints = assertIntRange(
    scoringObj,
    'wrongGuessOrOutOfGuessesHiderPoints',
    0,
    10
  );
  if (scoringObj.voidRoundPoints !== 0) {
    throw new RuleSchemaError('"scoring.voidRoundPoints" must be 0');
  }

  const turnObj = assertObject(raw, 'turnPolicy');
  const passTurnOnCorrectGuess = assertBoolean(turnObj, 'passTurnOnCorrectGuess');
  const keepTurnOnWrongGuess = assertBoolean(turnObj, 'keepTurnOnWrongGuess');

  const timeoutsObj = assertObject(raw, 'timeoutsMs');
  const showGol = assertIntRange(timeoutsObj, 'showGol', 1000, 120_000);
  const hiding = assertIntRange(timeoutsObj, 'hiding', 1000, 120_000);
  const guessDisplay = assertIntRange(timeoutsObj, 'guessDisplay', 500, 30_000);
  const revealVerify = assertIntRange(timeoutsObj, 'revealVerify', 1000, 120_000);

  const voidObj = assertObject(raw, 'voidConditions');
  const maxHandLostMs = assertIntRange(voidObj, 'maxHandLostMs', 100, 5000);
  const voidOnRevealNeither = assertBoolean(voidObj, 'voidOnRevealNeither');
  const voidOnRevealBoth = assertBoolean(voidObj, 'voidOnRevealBoth');
  const voidOnTimeout = assertBoolean(voidObj, 'voidOnTimeout');

  return {
    id,
    version,
    nameFa,
    nameEn,
    descriptionFa,
    handsCount,
    targetScore,
    maxGuessesPerRound,
    allowPoochCalls,
    scoring: {
      correctGolNoWrongCalls,
      correctGolAfterPoochCalls,
      wrongGuessOrOutOfGuessesHiderPoints,
      voidRoundPoints: 0,
    },
    turnPolicy: {
      passTurnOnCorrectGuess,
      keepTurnOnWrongGuess,
    },
    timeoutsMs: {
      showGol,
      hiding,
      guessDisplay,
      revealVerify,
    },
    voidConditions: {
      maxHandLostMs,
      voidOnRevealNeither,
      voidOnRevealBoth,
      voidOnTimeout,
    },
  };
}

export function isValidRuleSet(raw: unknown): raw is RuleSetDefinition {
  try {
    validateRuleSet(raw);
    return true;
  } catch {
    return false;
  }
}
