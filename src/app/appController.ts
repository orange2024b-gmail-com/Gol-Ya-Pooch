import type { RoundBehaviorRecord } from '../ai/behavior/types';
import { HedgeEnsemble } from '../ai/ensemble';
import { makeFinalGuess, type FinalGuessResult } from '../ai/guesser';
import {
  chooseComputerHidingHand,
  generateHidingChoreography,
} from '../ai/hider';
import { CameraManager, type CameraDeviceSummary } from '../camera/cameraManager';
import {
  COMMIT_REVEAL_EXPLANATION_FA,
  createCommitEnvelope,
  verifyCommitReveal,
  type CommitVerificationResult,
} from '../game/commitReveal';
import {
  createInitialGameContext,
  transitionGameFsm,
  type AiGuessPayload,
  type ComputerHideSetupPayload,
  type GameContext,
  type GameEvent,
} from '../game/fsm';
import {
  LEVELS,
  getLevelDefinition,
  isLevelUnlocked,
} from '../game/levels';
import {
  RULE_PRESETS,
  type GuessCallType,
} from '../game/rules/engine';
import { renderBeliefGraphCanvas, renderVisionOverlay } from '../render/canvasOverlay';
import { PixiHidingScene } from '../render/pixiHidingScene';
import { DEFAULT_PROFILE_ID, IndexedDbProfileStore } from '../storage/profileStore';
import {
  IndexedDbProgressionStore,
  applyMatchCompletionToProgression,
  applyRoundCompletionToProgression,
  createDefaultProgressionState,
  type PlayerProgressionState,
} from '../storage/progressionStore';
import {
  IndexedDbRoundLogStore,
  createDefaultConsentSettings,
  type ConsentSettings,
  type RoundLogRecord,
} from '../storage/roundLogStore';
import { BeliefFilter, type BeliefState } from '../tracking/beliefFilter';
import {
  TEMPORAL_SAMPLE_RATE_HZ,
  extractFrameFeatureVector,
  resampleHidingFeatures,
} from '../tracking/temporalFeatures';
import {
  TemporalModelRunner,
  type TemporalPredictionResult,
} from '../tracking/temporalModel';
import { FeatureExtractor, type FrameFeatures } from '../vision/features';
import {
  PinchConfirmationTracker,
  calibrateGolColorFromRgba,
  detectLightingShift,
  evaluatePinchForHand,
  scanHandRoisForGol,
  verifyRevealHands,
} from '../vision/golDetector';
import { MediaPipeHandTracker } from '../vision/handTracker';
import { CalibrationStabilityTracker, assessFrameQuality } from '../vision/quality';
import {
  SessionRecorder,
  parseRecordedSession,
  replayRecordedSession,
} from '../vision/recorder';
import {
  createBuiltInSampleSession,
  createPhase2TransferSampleSession,
} from '../vision/sampleSession';
import type {
  GolColorModel,
  PinchConfirmationState,
  PinchObservation,
  QualityReport,
  RecordedSession,
  RevealVerificationResult,
  VisionFrameHands,
} from '../vision/types';
import {
  FA,
  formatSecondsFa,
  handIndexLabelFa,
  sideLabelFa,
  toPersianDigits,
} from './i18n';

const CALIBRATION_CIRCLE = { x: 0.5, y: 0.5, radiusNorm: 0.085 };

export class GolYaPoochApp {
  private readonly root: HTMLElement;
  private ctx: GameContext = createInitialGameContext();
  private readonly camera = new CameraManager();
  private readonly tracker = new MediaPipeHandTracker();
  private readonly stabilityTracker = new CalibrationStabilityTracker(1000, 0.7);
  private readonly pinchTracker = new PinchConfirmationTracker(15);
  private readonly featureExtractor = new FeatureExtractor();
  private readonly beliefFilter = new BeliefFilter();
  private readonly ensemble = new HedgeEnsemble();
  private readonly profileStore = new IndexedDbProfileStore();
  private readonly progressionStore = new IndexedDbProgressionStore();
  private readonly roundLogStore = new IndexedDbRoundLogStore();
  private readonly temporalModel = new TemporalModelRunner();
  private readonly recorder = new SessionRecorder();

  private progression: PlayerProgressionState = createDefaultProgressionState();
  private consentSettings: ConsentSettings = createDefaultConsentSettings();
  private loggedRoundsCount = 0;
  private lastLogExportJson: string | null = null;
  private logNoticeFa: string | null = null;
  private logNoticeError = false;
  private behaviorHistory: RoundBehaviorRecord[] = [];
  private hidingFeatureFrames: FrameFeatures[] = [];
  private hidingStartedAtMs = 0;
  private lastGuessResult: FinalGuessResult | null = null;
  private lastTemporalPred: TemporalPredictionResult | null = null;
  private resolvingHidingGuess = false;
  private videoEl!: HTMLVideoElement;
  private overlayCanvas!: HTMLCanvasElement;
  private overlayCtx!: CanvasRenderingContext2D;
  private sampleCanvas!: HTMLCanvasElement;
  private sampleCtx!: CanvasRenderingContext2D;
  private beliefGraphCanvas: HTMLCanvasElement | null = null;

  private stageContainer!: HTMLElement;
  private stageBannerEl!: HTMLElement;
  private pixiContainer!: HTMLElement;
  private pixiScene: PixiHidingScene | null = null;
  private panelContainer!: HTMLElement;
  private debugContainer!: HTMLElement;
  private topbarStatusEl!: HTMLElement;

  private mirrored = true;
  private showLandmarks = true;
  private showDebug = false;
  private reducedMotion = false;
  private permissionError: string | null = null;
  private cameraDevices: CameraDeviceSummary[] = [];
  private lowFpsDegraded = false;
  private lowFpsCounter = 0;

  private golColorModel: GolColorModel | null = null;
  private lightingShiftDetected = false;
  private lastHands: VisionFrameHands | null = null;
  private lastQuality: QualityReport | null = null;
  private lastFeatures: FrameFeatures | null = null;
  private lastBelief: BeliefState | null = null;
  private lastPinchObs: PinchObservation[] = [];
  private lastPinchState: PinchConfirmationState | null = null;
  private lastRevealResult: RevealVerificationResult | null = null;
  private stabilityPassed = false;
  private stabilityProgress = 0;

  private activeTeamCallMode: GuessCallType = 'POOCH';
  private selectedHandIndex: number | null = null;
  private lastCommitVerification: CommitVerificationResult | null = null;
  private showHowItWorks = false;
  private lastUnlockedNoticeFa: string | null = null;

  private fps = 0;
  private inferenceMs = 0;
  private lastFrameTs = 0;
  private lastDebugRenderTs = 0;
  private rafId: number | null = null;
  private uiTimerRafId: number | null = null;
  private lastUiTimerTs = 0;
  private replayTimerId: number | null = null;
  private replayActive = false;
  private replayStatusText: string | null = null;

  constructor(root: HTMLElement) {
    this.root = root;
  }

  mount(): void {
    this.buildShell();
    this.bindGlobalListeners();
    this.renderUi();
    void this.loadStoredState();
    void this.initTemporalModel();
  }

  private async initTemporalModel(): Promise<void> {
    await this.temporalModel.init();
    if (this.showDebug) {
      this.renderDebugPanel();
    }
  }

  private async loadStoredState(): Promise<void> {
    const [profile, prog, consent, logCount] = await Promise.all([
      this.profileStore.loadProfile(DEFAULT_PROFILE_ID),
      this.progressionStore.load(DEFAULT_PROFILE_ID),
      this.roundLogStore.getConsentSettings(DEFAULT_PROFILE_ID),
      this.roundLogStore.getRoundLogCount(DEFAULT_PROFILE_ID),
    ]);
    if (profile) {
      this.ensemble.restore(profile.ensemble);
      this.behaviorHistory = profile.history;
    }
    this.progression = prog;
    this.consentSettings = consent;
    this.loggedRoundsCount = logCount;
    this.ctx = transitionGameFsm(this.ctx, {
      type: 'SELECT_LEVEL',
      levelId: prog.selectedLevelId,
      teamHandsCount: prog.teamHandsCount,
    });
    if (
      this.ctx.state === 'MENU' ||
      this.ctx.state === 'LEVEL_SELECT' ||
      this.ctx.state === 'STATS' ||
      this.ctx.state === 'CONSENT' ||
      this.showDebug
    ) {
      this.renderUi();
    }
  }

  private async saveBehaviorProfile(): Promise<void> {
    await this.profileStore.saveProfile({
      profileId: DEFAULT_PROFILE_ID,
      updatedAtMs: Date.now(),
      roundsPlayed: this.behaviorHistory.length,
      ensemble: this.ensemble.serialize(),
      history: this.behaviorHistory,
    });
  }

  private buildShell(): void {
    this.root.innerHTML = '';

    const shell = document.createElement('div');
    shell.className = 'app-shell';

    const topbar = document.createElement('header');
    topbar.className = 'topbar';

    const brand = document.createElement('div');
    brand.className = 'brand';

    const mark = document.createElement('div');
    mark.className = 'brand-mark';
    mark.textContent = 'گ';

    const titleWrap = document.createElement('div');
    const title = document.createElement('h1');
    title.className = 'brand-title';
    title.textContent = FA.appTitle;
    const phaseNote = document.createElement('div');
    phaseNote.className = 'brand-phase';
    phaseNote.textContent = FA.appSubtitle;
    titleWrap.append(title, phaseNote);
    brand.append(mark, titleWrap);

    const actions = document.createElement('div');
    actions.className = 'topbar-actions';

    this.topbarStatusEl = document.createElement('span');
    this.topbarStatusEl.className = 'brand-phase';

    const debugBtn = document.createElement('button');
    debugBtn.type = 'button';
    debugBtn.className = 'btn btn-secondary btn-compact';
    debugBtn.textContent = 'دیباگ (D)';
    debugBtn.setAttribute('data-testid', 'toggle-debug-btn');
    debugBtn.addEventListener('click', () => this.toggleDebug());

    actions.append(this.topbarStatusEl, debugBtn);
    topbar.append(brand, actions);

    const mainGrid = document.createElement('main');
    mainGrid.className = 'main-grid';

    this.stageContainer = document.createElement('section');
    this.stageContainer.className = 'camera-stage hidden';
    this.stageContainer.setAttribute('aria-label', 'نمای زنده دوربین');

    this.videoEl = document.createElement('video');
    this.videoEl.className = 'camera-video mirrored';
    this.videoEl.playsInline = true;
    this.videoEl.muted = true;

    this.overlayCanvas = document.createElement('canvas');
    this.overlayCanvas.className = 'camera-overlay';
    this.overlayCanvas.dir = 'ltr';
    const oCtx = this.overlayCanvas.getContext('2d');
    if (!oCtx) throw new Error('2D canvas context unavailable');
    this.overlayCtx = oCtx;

    this.sampleCanvas = document.createElement('canvas');
    this.sampleCanvas.width = 320;
    this.sampleCanvas.height = 240;
    const sCtx = this.sampleCanvas.getContext('2d', { willReadFrequently: true });
    if (!sCtx) throw new Error('2D offscreen canvas context unavailable');
    this.sampleCtx = sCtx;

    this.stageBannerEl = document.createElement('div');
    this.stageBannerEl.className = 'stage-banner hidden';

    this.stageContainer.append(this.videoEl, this.overlayCanvas, this.stageBannerEl);

    this.pixiContainer = document.createElement('section');
    this.pixiContainer.className = 'pixi-stage hidden';
    this.pixiContainer.setAttribute('aria-label', 'صحنه انیمیشن پنهان‌کاری رایانه');
    this.pixiContainer.setAttribute('data-testid', 'pixi-stage');

    this.pixiScene = new PixiHidingScene(this.pixiContainer);
    this.pixiScene.setSelectHandler((handIndex) => {
      if (this.ctx.state !== 'HUMAN_GUESS') return;
      if (!this.ctx.commitPublic) return;
      this.selectedHandIndex = handIndex;
      const callType: GuessCallType = this.ctx.ruleSet.allowPoochCalls
        ? this.activeTeamCallMode
        : 'GOL';
      this.dispatch({ type: 'HUMAN_GUESS_CALL', handIndex, callType });
    });

    this.panelContainer = document.createElement('section');
    this.panelContainer.setAttribute('data-testid', 'screen-panel');

    mainGrid.append(this.stageContainer, this.pixiContainer, this.panelContainer);

    this.debugContainer = document.createElement('section');
    this.debugContainer.className = 'debug-panel hidden';
    this.debugContainer.setAttribute('data-testid', 'debug-panel');

    shell.append(topbar, mainGrid, this.debugContainer);
    this.root.append(shell);
  }

  private bindGlobalListeners(): void {
    document.addEventListener('visibilitychange', () => {
      if (document.hidden) {
        this.dispatch({ type: 'TAB_HIDDEN' });
      } else {
        this.dispatch({ type: 'TAB_VISIBLE' });
      }
    });

    window.addEventListener('keydown', (ev) => {
      if (ev.key === 'd' || ev.key === 'D' || ev.key === 'د') {
        const tag = (ev.target as HTMLElement | null)?.tagName;
        if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
        this.toggleDebug();
      }
    });

    this.camera.setLossHandler((reasonFa) => {
      this.dispatch({ type: 'CAMERA_LOST', messageFa: reasonFa });
    });
  }

  private toggleDebug(): void {
    this.showDebug = !this.showDebug;
    this.renderDebugPanel();
    if (this.shouldShowCameraStage()) {
      this.updateLiveOverlay();
    }
  }

  private computeAiGuess(temporalPred: TemporalPredictionResult | null = null): AiGuessPayload {
    const snapshot = this.beliefFilter.getSnapshot();
    const obsQuality = this.beliefFilter.getObservationQuality();
    const behaviorPred = this.ensemble.predict(this.behaviorHistory);
    const modelStatus = this.temporalModel.getStatus();
    const result = makeFinalGuess({
      belief: snapshot,
      observationQuality: obsQuality,
      behavior: behaviorPred,
      temporal: temporalPred,
      temporalFusionWeights: modelStatus.fusionConfig.fusionWeights,
    });
    this.lastGuessResult = result;
    return {
      guess: result.guess,
      probability: result.probability,
      explanationFa: result.explanationFa,
      timelineEvents: result.timelineDescriptionsFa,
    };
  }

  private async computeAiGuessAsync(): Promise<AiGuessPayload> {
    const initialSide = this.ctx.initialShownHand ?? 'LEFT';
    if (!this.temporalModel.isReady) {
      await this.temporalModel.init();
    }
    const temporalPred = await this.temporalModel.predict(
      this.hidingFeatureFrames,
      initialSide
    );
    this.lastTemporalPred = temporalPred;
    return this.computeAiGuess(temporalPred);
  }

  private async resolveHidingAndFinish(
    contactCount: number,
    proceedToRevealAfter = false
  ): Promise<void> {
    if (this.resolvingHidingGuess) return;
    this.resolvingHidingGuess = true;
    try {
      const aiGuess = await this.computeAiGuessAsync();
      if (this.ctx.state === 'HIDING') {
        this.dispatch({
          type: 'FINISH_HIDING',
          aiGuess,
          contactCount,
        });
        if (proceedToRevealAfter) {
          this.dispatch({ type: 'PROCEED_TO_REVEAL' });
        }
      }
    } finally {
      this.resolvingHidingGuess = false;
    }
  }

  private persistVerifiedHumanHidesRound(verifiedHand: 'LEFT' | 'RIGHT'): void {
    if (!this.consentSettings.consentEnabled) return;

    const initialSide = this.ctx.initialShownHand ?? 'LEFT';
    const frames = this.hidingFeatureFrames;
    const resampled = resampleHidingFeatures(frames, initialSide);
    const rawFrameFeatures = frames.map((f, idx) => ({
      tMs: f.timestampMs,
      v: extractFrameFeatureVector(
        f,
        initialSide,
        frames.length <= 1 ? 0 : idx / (frames.length - 1)
      ),
    }));

    const snap = this.beliefFilter.getSnapshot();
    const guessRes = this.lastGuessResult;
    const hidingDurationMs = Math.max(
      resampled.durationMs,
      performance.now() - this.hidingStartedAtMs
    );

    const record: RoundLogRecord = {
      schemaVersion: 1,
      id: `rl-${Date.now()}-${this.ctx.roundNumber}`,
      profileId: DEFAULT_PROFILE_ID,
      recordedAtMs: Date.now(),
      levelId: this.ctx.levelId,
      ruleSetId: this.ctx.ruleSet.id,
      initialShownHand: initialSide,
      resampledSequence: resampled.matrix,
      rawFrameFeatures,
      eventTimeline: snap.timeline,
      beliefTrace: snap.trace,
      finalGuess: {
        guess: guessRes?.guess ?? this.ctx.computerGuess ?? 'LEFT',
        probability: guessRes?.probability ?? this.ctx.computerGuessProb ?? 0.5,
        pLeft: guessRes?.pLeft ?? 0.5,
        hmmPLeft: guessRes?.hmmPLeft ?? snap.pLeft,
        temporalPLeft: guessRes?.temporalPLeft ?? null,
        behaviorPLeft: guessRes?.behaviorPLeft ?? 0.5,
        dominantSource: guessRes?.dominantSource ?? 'TRACKING',
      },
      verifiedGroundTruth: verifiedHand,
      timings: {
        hidingDurationMs,
        frameCount: frames.length,
        sampleRateHz: TEMPORAL_SAMPLE_RATE_HZ,
        temporalInferenceMs: guessRes?.temporalInferenceMs ?? null,
        totalProcessingMs: this.inferenceMs,
      },
    };

    void this.roundLogStore.appendRoundLog(record).then(async (saved) => {
      if (saved) {
        this.loggedRoundsCount = await this.roundLogStore.getRoundLogCount(DEFAULT_PROFILE_ID);
        if (this.ctx.state === 'ROUND_RESULT' || this.ctx.state === 'STATS' || this.ctx.state === 'CONSENT') {
          this.renderPanel();
        }
      }
    });
  }

  private async handleToggleConsent(enabled: boolean): Promise<void> {
    this.consentSettings = await this.roundLogStore.setConsentEnabled(
      enabled,
      DEFAULT_PROFILE_ID
    );
    this.logNoticeError = false;
    this.logNoticeFa = enabled
      ? 'ثبت محلی داده‌های عددی فعال شد. تنها ویژگی‌های عددی و رویدادها ذخیره می‌شوند.'
      : 'ثبت داده غیرفعال شد. هیچ دور جدیدی ذخیره نمی‌شود.';
    this.renderPanel();
  }

  private async handleExportLogs(): Promise<void> {
    try {
      const json = await this.roundLogStore.exportLogsJson(DEFAULT_PROFILE_ID, true);
      this.lastLogExportJson = json;
      const blob = new Blob([json], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `gol-ya-pooch-logs-${Date.now()}.json`;
      a.click();
      URL.revokeObjectURL(url);

      const count = await this.roundLogStore.getRoundLogCount(DEFAULT_PROFILE_ID);
      this.loggedRoundsCount = count;
      this.logNoticeError = false;
      this.logNoticeFa = `خروجی JSON با موفقیت آماده شد (${toPersianDigits(count)} دور عددی، بدون تصویر/ویدیو).`;
    } catch (err) {
      this.logNoticeError = true;
      this.logNoticeFa = `خطا در خروجی JSON: ${err instanceof Error ? err.message : String(err)}`;
    }
    this.renderPanel();
  }

  private async handleImportLogs(file: File): Promise<void> {
    try {
      const text = await file.text();
      const res = await this.roundLogStore.importLogsJson(text, {
        profileIdOverride: DEFAULT_PROFILE_ID,
      });
      this.loggedRoundsCount = await this.roundLogStore.getRoundLogCount(DEFAULT_PROFILE_ID);
      this.logNoticeError = false;
      this.logNoticeFa = `${toPersianDigits(res.importedCount)} دور وارد شد (${toPersianDigits(res.skippedDuplicates)} تکراری نادیده گرفته شد).`;
    } catch (err) {
      this.logNoticeError = true;
      this.logNoticeFa = `فایل نامعتبر یا دارای داده غیرعددی رد شد: ${err instanceof Error ? err.message : String(err)}`;
    }
    this.renderPanel();
  }

  private async handleClearLogs(): Promise<void> {
    await this.roundLogStore.clearLogs(DEFAULT_PROFILE_ID);
    this.loggedRoundsCount = 0;
    this.lastLogExportJson = null;
    this.logNoticeError = false;
    this.logNoticeFa = 'تمام لاگ‌های عددی ثبت‌شده پاک شدند.';
    this.renderPanel();
  }

  private async runDebugOnnxBenchmark(): Promise<void> {
    await this.temporalModel.init();
    let frames = this.hidingFeatureFrames;
    let initialSide = this.ctx.initialShownHand ?? 'LEFT';
    if (frames.length === 0) {
      const sample = replayRecordedSession(createPhase2TransferSampleSession(), {
        requiredPinchFrames: 15,
      });
      frames = sample.frames.slice(18, -1).map((f) => f.features);
      initialSide = sample.confirmedSide ?? 'LEFT';
    }
    this.lastTemporalPred = await this.temporalModel.predict(frames, initialSide);
    this.renderDebugPanel();
  }

  private async prepareComputerHidesRound(): Promise<void> {
    const level = getLevelDefinition(this.ctx.levelId);
    const handsCount = this.ctx.ruleSet.handsCount;
    const decision = chooseComputerHidingHand(
      level,
      handsCount,
      this.progression.humanGuessHistory
    );
    const env = await createCommitEnvelope(decision.chosenHandIndex, handsCount);
    const setup: ComputerHideSetupPayload = {
      secretGolHandIndex: decision.chosenHandIndex,
      commitPublic: env.publicInfo,
      commitSecret: env.secret,
      shuffleDurationMs: this.reducedMotion ? 600 : level.animation.shuffleDurationMs,
    };

    if (this.ctx.state === 'COMPUTER_HIDES' || this.ctx.state === 'HUMAN_GUESS') {
      this.dispatch({ type: 'SET_COMPUTER_COMMIT', setup });
    }
  }

  private startPixiComputerHides(): void {
    const level = getLevelDefinition(this.ctx.levelId);
    const handsCount = this.ctx.ruleSet.handsCount;
    const choreo = generateHidingChoreography(level, handsCount, this.reducedMotion);
    this.lastCommitVerification = null;
    this.selectedHandIndex = null;
    this.activeTeamCallMode = this.ctx.ruleSet.allowPoochCalls ? 'POOCH' : 'GOL';

    const commitP = this.prepareComputerHidesRound();

    if (!this.pixiScene) return;
    void this.pixiScene.init().then(() => {
      if (this.ctx.state !== 'COMPUTER_HIDES' || !this.pixiScene) return;
      this.pixiScene.updateState({
        phase: 'COMPUTER_HIDES',
        handsCount,
        selectedHandIndex: null,
        openedEmptyHands: [],
        revealedGolHandIndex: null,
        allowPoochCalls: this.ctx.ruleSet.allowPoochCalls,
        reducedMotion: this.reducedMotion,
      });
      this.pixiScene.startChoreography(choreo, () => {
        void commitP.then(() => {
          if (this.ctx.state === 'COMPUTER_HIDES') {
            this.dispatch({ type: 'COMPUTER_SHUFFLE_DONE' });
          }
        });
      });
    });

    this.ensureUiTimerLoop();
  }

  private syncPixiSceneState(): void {
    if (!this.pixiScene) return;
    if (!this.shouldShowPixiStage()) return;
    const phase: 'COMPUTER_HIDES' | 'HUMAN_GUESS' | 'REVEALED' =
      this.ctx.state === 'COMPUTER_HIDES'
        ? 'COMPUTER_HIDES'
        : this.ctx.state === 'HUMAN_GUESS'
          ? 'HUMAN_GUESS'
          : 'REVEALED';

    void this.pixiScene.init().then(() => {
      this.pixiScene?.updateState({
        phase,
        handsCount: this.ctx.ruleSet.handsCount,
        selectedHandIndex: this.selectedHandIndex,
        openedEmptyHands: this.ctx.teamGuessProgress.openedEmptyHands,
        revealedGolHandIndex:
          phase === 'REVEALED' ? this.ctx.computerSecretGolHandIndex : null,
        allowPoochCalls: this.ctx.ruleSet.allowPoochCalls,
        reducedMotion: this.reducedMotion,
      });
    });
  }

  private verifyAndRecordRoundResult(prevState: GameContext['state']): void {
    // Verify commit-reveal when a computer-hides round finishes
    if (
      prevState === 'HUMAN_GUESS' &&
      this.ctx.commitPublic &&
      this.ctx.commitSecret
    ) {
      const pub = this.ctx.commitPublic;
      const sec = this.ctx.commitSecret;
      void verifyCommitReveal(pub.commitHex, sec.choice, sec.saltHex).then((ver) => {
        this.lastCommitVerification = ver;
        if (this.ctx.state === 'ROUND_RESULT') {
          this.renderPanel();
        }
      });
    }

    // Update progression & stats
    if (this.ctx.lastRoundWinner) {
      const wasCompHiding = prevState === 'HUMAN_GUESS';
      const guessRec =
        wasCompHiding && this.ctx.lastHumanCalledHandIndex !== null
          ? {
              roundIndex: this.ctx.roundNumber,
              handsCount: this.ctx.ruleSet.handsCount,
              guessedHandIndex: this.ctx.lastHumanCalledHandIndex,
              humanFoundGol: this.ctx.lastRoundWinner === 'HUMAN',
            }
          : undefined;

      this.progression = applyRoundCompletionToProgression(this.progression, {
        winner: this.ctx.lastRoundWinner,
        wasComputerHiding: wasCompHiding,
        reason: this.ctx.lastComputerHidesOutcomeReason,
        ...(guessRec ? { guessRecord: guessRec } : {}),
      });

      const levelResolved =
        this.ctx.bestOfMatches === 3
          ? this.ctx.seriesWinner !== null
          : this.ctx.matchWinner !== null;

      if (levelResolved) {
        const wonLevel =
          (this.ctx.bestOfMatches === 3
            ? this.ctx.seriesWinner
            : this.ctx.matchWinner) === 'HUMAN';
        const prevMax = this.progression.unlockedMaxLevel;
        this.progression = applyMatchCompletionToProgression(this.progression, {
          levelId: this.ctx.levelId,
          humanWonLevel: wonLevel,
          scoreDiff: this.ctx.humanScore - this.ctx.computerScore,
        });
        if (this.progression.unlockedMaxLevel > prevMax) {
          const nextLvl = getLevelDefinition(this.progression.unlockedMaxLevel);
          this.lastUnlockedNoticeFa = `سطح ${toPersianDigits(nextLvl.id)} (${nextLvl.nameFa}) باز شد!`;
        } else {
          this.lastUnlockedNoticeFa = null;
        }
      } else {
        this.lastUnlockedNoticeFa = null;
      }

      void this.progressionStore.save(this.progression);
    }
  }

  private dispatch(event: GameEvent): void {
    const prevState = this.ctx.state;

    // Record behavioral feedback & consent-gated numeric round log when a human-hides round completes
    if (
      prevState === 'REVEAL_VERIFY' &&
      event.type === 'REVEAL_VERIFIED' &&
      (event.outcome === 'LEFT' || event.outcome === 'RIGHT') &&
      this.ctx.computerGuess !== null
    ) {
      const priorHistory = [...this.behaviorHistory];
      this.ensemble.update(priorHistory, event.outcome);
      this.behaviorHistory.push({
        roundIndex: this.ctx.roundNumber,
        timestampMs: Date.now(),
        chosenHand: event.outcome,
        computerGuess: this.ctx.computerGuess,
        humanWon: event.outcome !== this.ctx.computerGuess,
      });
      void this.saveBehaviorProfile();
      this.persistVerifiedHumanHidesRound(event.outcome);
    }

    if (event.type === 'SELECT_LEVEL') {
      this.progression = {
        ...this.progression,
        selectedLevelId: event.levelId,
        ...(event.teamHandsCount ? { teamHandsCount: event.teamHandsCount } : {}),
      };
      void this.progressionStore.save(this.progression);
    } else if (event.type === 'SET_TEAM_HANDS_COUNT') {
      this.progression = { ...this.progression, teamHandsCount: event.handsCount };
      void this.progressionStore.save(this.progression);
    }

    this.ctx = transitionGameFsm(this.ctx, event);

    if (prevState !== this.ctx.state) {
      if (this.ctx.state === 'SHOW_GOL') {
        this.pinchTracker.reset();
        this.featureExtractor.reset();
        this.lastPinchState = null;
        this.lastPinchObs = [];
        this.lastRevealResult = null;
        this.lastBelief = null;
        this.lastCommitVerification = null;
      } else if (this.ctx.state === 'HIDING') {
        this.featureExtractor.reset();
        this.hidingFeatureFrames = [];
        this.hidingStartedAtMs = performance.now();
        this.lastGuessResult = null;
        this.resolvingHidingGuess = false;
        this.lastBelief = this.beliefFilter.init(
          this.ctx.initialShownHand ?? 'LEFT',
          this.lastHands?.timestampMs ?? performance.now()
        );
      } else if (this.ctx.state === 'REVEAL_VERIFY') {
        this.lastRevealResult = null;
      } else if (this.ctx.state === 'CALIBRATION') {
        this.stabilityTracker.reset();
        this.stabilityPassed = false;
        this.stabilityProgress = 0;
      } else if (this.ctx.state === 'COMPUTER_HIDES') {
        this.lastUnlockedNoticeFa = null;
        this.showHowItWorks = false;
        this.renderUi();
        this.startPixiComputerHides();
        return;
      } else if (this.ctx.state === 'HUMAN_GUESS') {
        this.ensureUiTimerLoop();
        this.syncPixiSceneState();
      } else if (this.ctx.state === 'ROUND_RESULT') {
        this.verifyAndRecordRoundResult(prevState);
        this.syncPixiSceneState();
      } else {
        this.pixiScene?.stopChoreography();
      }

      this.renderUi();
      return;
    }

    if (event.type === 'TICK') {
      this.updateDynamicFieldsInPlace();
      if (this.shouldShowCameraStage()) {
        this.updateLiveOverlay();
      }
      if (this.showDebug && performance.now() - this.lastDebugRenderTs > 140) {
        this.updateDebugMetricsInPlace();
      }
      return;
    }

    if (event.type === 'HUMAN_GUESS_CALL') {
      this.syncPixiSceneState();
    }

    this.renderUi();
  }

  private ensureUiTimerLoop(): void {
    if (this.uiTimerRafId !== null) return;
    this.lastUiTimerTs = performance.now();

    const step = (now: number) => {
      if (
        this.ctx.state !== 'COMPUTER_HIDES' &&
        this.ctx.state !== 'HUMAN_GUESS'
      ) {
        this.uiTimerRafId = null;
        return;
      }

      const dtMs = Math.max(1, Math.min(100, now - this.lastUiTimerTs));
      this.lastUiTimerTs = now;

      // Avoid double-ticking if camera loop is already driving ticks
      if (!this.camera.isStreaming || this.replayActive || this.ctx.practiceMode) {
        if (this.ctx.state === 'COMPUTER_HIDES') {
          if (this.ctx.phaseRemainingMs - dtMs <= 0 && !this.ctx.commitPublic) {
            // Wait for Web Crypto commit envelope before advancing
          } else {
            this.dispatch({ type: 'TICK', dtMs });
          }
        } else if (this.ctx.state === 'HUMAN_GUESS') {
          this.dispatch({ type: 'TICK', dtMs });
        }
      }

      this.uiTimerRafId = requestAnimationFrame(step);
    };

    this.uiTimerRafId = requestAnimationFrame(step);
  }

  private shouldShowCameraStage(): boolean {
    if (this.shouldShowPixiStage()) return false;
    if (this.ctx.practiceMode && !this.replayActive) return false;
    return (
      this.ctx.state !== 'SPLASH' &&
      this.ctx.state !== 'PERMISSION' &&
      this.ctx.state !== 'CAMERA_ERROR' &&
      this.ctx.state !== 'LEVEL_SELECT' &&
      this.ctx.state !== 'STATS' &&
      this.ctx.state !== 'RULES' &&
      this.ctx.state !== 'CONSENT'
    );
  }

  private shouldShowPixiStage(): boolean {
    return (
      this.ctx.state === 'COMPUTER_HIDES' ||
      this.ctx.state === 'HUMAN_GUESS' ||
      (this.ctx.state === 'ROUND_RESULT' && this.ctx.commitPublic !== null)
    );
  }

  private async handleRequestCamera(): Promise<void> {
    this.permissionError = null;
    try {
      await this.camera.start(this.videoEl);
      this.cameraDevices = await this.camera.listCameras();
      await this.tracker.init();
      this.startLoop();
      this.dispatch({ type: 'CAMERA_READY' });
    } catch (err) {
      this.permissionError =
        err instanceof Error && err.message
          ? `${FA.permissionDeniedHint} (${err.message})`
          : FA.permissionDeniedHint;
      this.renderUi();
    }
  }

  private startLoop(): void {
    if (this.rafId !== null) cancelAnimationFrame(this.rafId);
    this.lastFrameTs = performance.now();

    const step = (now: number) => {
      this.rafId = requestAnimationFrame(step);
      if (this.replayActive) return;

      const dtMs = Math.max(1, Math.min(100, now - this.lastFrameTs));
      this.lastFrameTs = now;
      const instantFps = 1000 / dtMs;
      this.fps = this.fps === 0 ? instantFps : this.fps * 0.85 + instantFps * 0.15;

      if (this.fps < 22 && this.camera.isStreaming && !this.lowFpsDegraded) {
        this.lowFpsCounter += 1;
        if (this.lowFpsCounter > 45) {
          this.lowFpsDegraded = true;
          void this.camera.start(this.videoEl, { preferLowRes: true }).catch(() => {});
        }
      } else {
        this.lowFpsCounter = 0;
      }

      if (this.ctx.state === 'PAUSED' || this.ctx.state === 'CAMERA_ERROR') {
        return;
      }

      if (
        (this.ctx.state === 'COMPUTER_HIDES' || this.ctx.state === 'HUMAN_GUESS') &&
        !this.ctx.practiceMode
      ) {
        if (this.ctx.state === 'COMPUTER_HIDES' && this.ctx.phaseRemainingMs - dtMs <= 0 && !this.ctx.commitPublic) {
          return;
        }
        this.dispatch({ type: 'TICK', dtMs });
        return;
      }

      if (!this.camera.isStreaming || this.videoEl.readyState < 2) {
        return;
      }

      this.processCameraFrame(now, dtMs);
    };

    this.rafId = requestAnimationFrame(step);
  }

  private getCurrentContactCount(): number {
    const events = this.beliefFilter.getSnapshot().timeline;
    let count = 0;
    for (const ev of events) {
      if (ev.type === 'CONTACT_START') count += 1;
    }
    if (this.lastFeatures?.interHand.inContact) {
      count = Math.max(count, 1);
    }
    return count;
  }

  private processCameraFrame(now: number, dtMs: number): void {
    const t0 = performance.now();
    const vw = this.videoEl.videoWidth || 640;
    const vh = this.videoEl.videoHeight || 480;
    if (this.overlayCanvas.width !== vw || this.overlayCanvas.height !== vh) {
      this.overlayCanvas.width = vw;
      this.overlayCanvas.height = vh;
    }

    const sw = this.sampleCanvas.width;
    const sh = this.sampleCanvas.height;
    this.sampleCtx.drawImage(this.videoEl, 0, 0, sw, sh);
    const frameImageData = this.sampleCtx.getImageData(0, 0, sw, sh);

    let rawDetections: ReturnType<MediaPipeHandTracker['detectFrame']>['rawDetections'] = [];
    if (this.tracker.isReady) {
      const res = this.tracker.detectFrame(this.videoEl, now);
      rawDetections = res.rawDetections;
      this.lastHands = res.hands;
    }

    if (this.lastHands) {
      this.lastQuality = assessFrameQuality(frameImageData.data, sw, sh, this.lastHands);
      if (this.golColorModel) {
        this.lightingShiftDetected = detectLightingShift(
          this.lastQuality.meanLuminance,
          this.golColorModel
        );
      }

      let golSightings = { left: 0, right: 0 };
      if (this.ctx.state === 'HIDING' && this.golColorModel) {
        golSightings = {
          left: scanHandRoisForGol(this.lastHands.left, frameImageData.data, sw, sh, this.golColorModel),
          right: scanHandRoisForGol(this.lastHands.right, frameImageData.data, sw, sh, this.golColorModel),
        };
      }

      this.lastFeatures = this.featureExtractor.update({
        hands: this.lastHands,
        quality: this.lastQuality,
        golSightings,
      });

      if (this.ctx.state === 'CALIBRATION') {
        const st = this.stabilityTracker.update(this.lastHands, now);
        this.stabilityProgress = st.progress;
        if (st.passed) {
          this.stabilityPassed = true;
        }
      }

      if (this.ctx.state === 'SHOW_GOL' && this.golColorModel) {
        const obs: PinchObservation[] = [];
        for (const hand of [this.lastHands.left, this.lastHands.right]) {
          const p = evaluatePinchForHand(hand, frameImageData.data, sw, sh, this.golColorModel);
          if (p) obs.push(p);
        }
        this.lastPinchObs = obs;
        this.lastPinchState = this.pinchTracker.update(obs);
        if (this.lastPinchState.confirmed && this.lastPinchState.confirmedSide) {
          this.dispatch({ type: 'PINCH_CONFIRMED', side: this.lastPinchState.confirmedSide });
        }
      } else if (this.ctx.state === 'HIDING') {
        this.lastBelief = this.beliefFilter.update(this.lastFeatures);
        if (this.hidingFeatureFrames.length < 900) {
          this.hidingFeatureFrames.push(this.lastFeatures);
        }
        this.lastPinchObs = [];
      } else if (this.ctx.state === 'REVEAL_VERIFY' && this.golColorModel) {
        const reveal = verifyRevealHands(
          frameImageData.data,
          sw,
          sh,
          this.lastHands.left,
          this.lastHands.right,
          this.golColorModel
        );
        this.lastRevealResult = reveal;
        if (reveal.bothOpen) {
          this.dispatch({ type: 'REVEAL_VERIFIED', outcome: reveal.outcome });
        }
      } else {
        this.lastPinchObs = [];
      }

      if (this.recorder.isRecording) {
        this.recorder.recordFrame({
          timestampMs: now,
          rawDetections,
          hands: this.lastHands,
          ...(this.lastQuality ? { quality: this.lastQuality } : {}),
          ...(this.lastFeatures ? { features: this.lastFeatures } : {}),
          ...(this.lastPinchObs.length > 0 ? { pinch: this.lastPinchObs } : {}),
          ...(this.lastRevealResult
            ? {
                reveal: {
                  bothOpen: this.lastRevealResult.bothOpen,
                  outcome: this.lastRevealResult.outcome,
                  leftGolDetected: Boolean(this.lastRevealResult.left?.golDetected),
                  rightGolDetected: Boolean(this.lastRevealResult.right?.golDetected),
                },
              }
            : {}),
        });
      }
    }

    this.inferenceMs = performance.now() - t0;

    if (
      this.ctx.state === 'SHOW_GOL' ||
      this.ctx.state === 'HIDING' ||
      this.ctx.state === 'COMPUTER_GUESS' ||
      this.ctx.state === 'REVEAL_VERIFY'
    ) {
      const willTimeoutHide =
        this.ctx.state === 'HIDING' && this.ctx.phaseRemainingMs - dtMs <= 0;
      const contactCount = this.ctx.state === 'HIDING' ? this.getCurrentContactCount() : 0;
      if (willTimeoutHide) {
        void this.resolveHidingAndFinish(contactCount);
        return;
      }
      this.dispatch({
        type: 'TICK',
        dtMs,
        maxLostDurationMs: this.lastHands?.maxLostDurationMs ?? 0,
        ...(this.ctx.state === 'HIDING' ? { contactCount } : {}),
      });
    } else {
      this.updateLiveOverlay();
      if (this.ctx.state === 'CALIBRATION') {
        this.updateDynamicFieldsInPlace();
      }
      if (this.showDebug && performance.now() - this.lastDebugRenderTs > 140) {
        this.updateDebugMetricsInPlace();
      }
    }
  }

  private sampleGolColorFromCurrentFrame(): void {
    const sw = this.sampleCanvas.width;
    const sh = this.sampleCanvas.height;
    if (this.videoEl.readyState >= 2) {
      this.sampleCtx.drawImage(this.videoEl, 0, 0, sw, sh);
    }
    const img = this.sampleCtx.getImageData(0, 0, sw, sh);
    const luma = this.lastQuality?.meanLuminance ?? 128;
    const model = calibrateGolColorFromRgba(
      img.data,
      sw,
      sh,
      { x: CALIBRATION_CIRCLE.x, y: CALIBRATION_CIRCLE.y },
      CALIBRATION_CIRCLE.radiusNorm * 0.75,
      luma,
      performance.now()
    );
    if (model) {
      this.golColorModel = model;
      this.lightingShiftDetected = false;
      this.recorder.setGolColorModel(model);
      this.renderUi();
    }
  }

  private stopReplay(): void {
    if (this.replayTimerId !== null) {
      window.clearInterval(this.replayTimerId);
      this.replayTimerId = null;
    }
    this.replayActive = false;
  }

  private startSessionReplay(session: RecordedSession): void {
    this.stopReplay();
    this.replayActive = true;
    this.showDebug = true;
    this.tracker.resetTracking();
    this.pinchTracker.reset();
    this.featureExtractor.reset();

    if (session.golColorModel) {
      this.golColorModel = session.golColorModel;
    }

    const summary = replayRecordedSession(session, { requiredPinchFrames: 15 });
    this.replayStatusText = `بازپخش ${toPersianDigits(summary.totalFrames)} فریم آغاز شد…`;

    this.ctx = {
      ...createInitialGameContext({
        targetScore: this.ctx.config.targetScore,
        enableTurnAlternation: false,
      }),
      state: 'SHOW_GOL',
      calibrated: true,
      practiceMode: false,
      phaseRemainingMs: 12_000,
    };
    this.renderUi();

    let idx = 0;
    this.replayTimerId = window.setInterval(() => {
      if (this.resolvingHidingGuess) return;
      const frame = summary.frames[idx];
      if (!frame) {
        this.stopReplay();
        this.replayStatusText = `بازپخش کامل شد: تأیید گل=${summary.pinchConfirmed ? 'بله' : 'خیر'} (${summary.confirmedSide ?? '-'}) | نتیجه بازگشایی=${summary.finalRevealOutcome ?? '-'}`;
        this.renderUi();
        return;
      }

      this.lastHands = frame.hands;
      this.lastFeatures = frame.features;
      this.lastPinchState = frame.pinchState;
      this.lastPinchObs = (session.frames[idx]?.pinch ?? []).map((p) => ({
        side: p.side,
        pinchPostureOk: p.pinchPostureOk,
        pinchDistanceNorm: p.pinchDistanceNorm,
        roiCenter: p.side === 'LEFT' ? { x: 0.28, y: 0.55 } : { x: 0.72, y: 0.55 },
        roiRadiusNorm: 0.05,
        blob: {
          matchRatio: p.blobMatchRatio,
          compactness: p.blobCompactness,
          matchedPixels: 40,
          totalPixels: 100,
          detected: p.golPresentInFrame,
        },
        golPresentInFrame: p.golPresentInFrame,
      }));

      if (this.ctx.state === 'SHOW_GOL' && frame.pinchState.confirmed && frame.pinchState.confirmedSide) {
        this.dispatch({ type: 'PINCH_CONFIRMED', side: frame.pinchState.confirmedSide });
      } else if (this.ctx.state === 'HIDING') {
        this.lastBelief = this.beliefFilter.update(frame.features);
        if (this.hidingFeatureFrames.length < 900) {
          this.hidingFeatureFrames.push(frame.features);
        }
        if (idx >= summary.frames.length - 2) {
          idx += 1;
          void this.resolveHidingAndFinish(this.getCurrentContactCount(), true);
          return;
        }
      }

      if (this.ctx.state === 'REVEAL_VERIFY' && frame.reveal) {
        this.dispatch({ type: 'REVEAL_VERIFIED', outcome: frame.reveal.outcome });
      } else if (
        this.ctx.state === 'SHOW_GOL' ||
        this.ctx.state === 'HIDING' ||
        this.ctx.state === 'REVEAL_VERIFY'
      ) {
        this.dispatch({
          type: 'TICK',
          dtMs: 33,
          maxLostDurationMs: frame.hands.maxLostDurationMs,
          ...(this.ctx.state === 'HIDING' ? { contactCount: this.getCurrentContactCount() } : {}),
        });
      }

      if (this.overlayCanvas.width !== 640) {
        this.overlayCanvas.width = 640;
        this.overlayCanvas.height = 480;
      }
      this.updateLiveOverlay();
      if (this.showDebug) {
        this.updateDebugMetricsInPlace();
      }
      idx += 1;
    }, 35);
  }

  private updateLiveOverlay(): void {
    const w = this.overlayCanvas.width || 640;
    const h = this.overlayCanvas.height || 480;
    renderVisionOverlay(this.overlayCtx, w, h, {
      mirrored: this.mirrored,
      showLandmarks: this.showLandmarks,
      showDebug: this.showDebug,
      showCalibrationCircle: this.ctx.state === 'CALIBRATION',
      calibrationCircle: CALIBRATION_CIRCLE,
      hands: this.lastHands,
      quality: this.lastQuality,
      pinchObs: this.lastPinchObs,
      pinchState: this.lastPinchState,
      revealResult: this.lastRevealResult,
      debugBelief: this.showDebug ? this.lastBelief : null,
      fps: this.fps,
      inferenceMs: this.inferenceMs,
    });

    // Player-facing stage banner: NEVER reveal belief during HIDING
    this.stageBannerEl.className = 'stage-banner';
    if (this.ctx.state === 'SHOW_GOL') {
      if (this.lastPinchState?.confirmed) {
        this.stageBannerEl.classList.add('success');
        this.stageBannerEl.textContent = FA.golConfirmedBadge;
      } else {
        const progressPct = Math.round((this.lastPinchState?.progress ?? 0) * 100);
        this.stageBannerEl.textContent = `${FA.showGolPrompt} (${toPersianDigits(progressPct)}٪)`;
      }
    } else if (this.ctx.state === 'HIDING') {
      this.stageBannerEl.textContent = `${FA.hidingIndicator} ${formatSecondsFa(this.ctx.phaseRemainingMs)}`;
    } else if (this.ctx.state === 'REVEAL_VERIFY') {
      this.stageBannerEl.textContent = FA.revealPrompt;
    } else if (this.lowFpsDegraded) {
      this.stageBannerEl.classList.add('warn');
      this.stageBannerEl.textContent = FA.lowFpsWarning;
    } else {
      this.stageBannerEl.classList.add('hidden');
    }
  }

  private isCalibrationReady(): boolean {
    return Boolean(
      this.lastQuality?.lightingOk &&
        this.stabilityPassed &&
        this.lastQuality?.distanceOk &&
        this.golColorModel !== null
    );
  }

  private updateDynamicFieldsInPlace(): void {
    const timerEl = this.panelContainer.querySelector('[data-hud-timer]');
    if (timerEl) {
      timerEl.textContent = formatSecondsFa(this.ctx.phaseRemainingMs);
    }

    const pinchEl = this.panelContainer.querySelector('[data-pinch-progress]');
    if (pinchEl) {
      const framesDone = this.lastPinchState?.consecutiveFrames ?? 0;
      const framesReq = this.lastPinchState?.requiredFrames ?? 15;
      pinchEl.textContent = `تثبیت پینچ: ${toPersianDigits(framesDone)} از ${toPersianDigits(framesReq)} فریم پیاپی`;
    }

    const contactEl = this.panelContainer.querySelector<HTMLElement>(
      '[data-testid="contact-requirement-status"]'
    );
    if (contactEl && this.ctx.config.minHandContactsRequired > 0) {
      const met = this.ctx.hidingContactCount >= this.ctx.config.minHandContactsRequired;
      contactEl.className = `contact-req-pill ${met ? 'done' : 'pending'}`;
      contactEl.textContent = met
        ? 'الزام سطح حرفه‌ای: تماس دست‌ها ثبت شد ✓'
        : 'الزام سطح حرفه‌ای: حداقل یک بار دست‌ها را به هم بزنید (در انتظار تماس…)';
    }

    if (this.ctx.state === 'CALIBRATION') {
      const q = this.lastQuality;
      this.updateCheckItemInPlace('lighting', Boolean(q?.lightingOk), q?.lightingHintFa ?? null);
      this.updateCheckItemInPlace(
        'hands',
        this.stabilityPassed,
        this.stabilityPassed
          ? 'هر دو دست با اطمینان کافی تثبیت شدند.'
          : `پیشرفت پایداری: ${toPersianDigits(Math.round(this.stabilityProgress * 100))}٪`
      );
      this.updateCheckItemInPlace('distance', Boolean(q?.distanceOk), q?.distanceHintFa ?? null);
      this.updateCheckItemInPlace(
        'gol',
        this.golColorModel !== null,
        this.golColorModel
          ? `مدل رنگ ثبت شد (${toPersianDigits(this.golColorModel.sampleCount)} نمونه پیکسل).`
          : FA.calGolInstruction
      );

      const doneBtn = this.panelContainer.querySelector<HTMLButtonElement>(
        '[data-testid="calibration-done-btn"]'
      );
      if (doneBtn) {
        doneBtn.disabled = !this.isCalibrationReady();
      }
    }
  }

  private updateCheckItemInPlace(key: string, ok: boolean, hint: string | null): void {
    const li = this.panelContainer.querySelector<HTMLElement>(`[data-cal-key="${key}"]`);
    if (!li) return;
    li.className = `check-item ${ok ? 'ok' : 'fail'}`;
    const badge = li.querySelector<HTMLElement>('[data-cal-status]');
    if (badge) {
      badge.className = `check-status ${ok ? 'ok' : 'pending'}`;
      badge.textContent = ok ? 'تأیید ✓' : 'در انتظار…';
    }
    const hintEl = li.querySelector<HTMLElement>('[data-cal-hint]');
    if (hintEl) {
      hintEl.textContent = hint ?? '';
      hintEl.classList.toggle('hidden', !hint);
    }
  }

  private renderUi(): void {
    const showCam = this.shouldShowCameraStage();
    const showPixi = this.shouldShowPixiStage();

    this.stageContainer.classList.toggle('hidden', !showCam);
    this.pixiContainer.classList.toggle('hidden', !showPixi);

    const mainGrid = this.panelContainer.parentElement;
    mainGrid?.classList.toggle('with-camera', showCam);
    mainGrid?.classList.toggle('with-pixi', showPixi);

    this.videoEl.classList.toggle('mirrored', this.mirrored);
    document.body.classList.toggle('reduced-motion', this.reducedMotion);

    const lvl = getLevelDefinition(this.ctx.levelId);
    const inGame =
      this.ctx.state === 'SHOW_GOL' ||
      this.ctx.state === 'HIDING' ||
      this.ctx.state === 'COMPUTER_GUESS' ||
      this.ctx.state === 'REVEAL_VERIFY' ||
      this.ctx.state === 'COMPUTER_HIDES' ||
      this.ctx.state === 'HUMAN_GUESS' ||
      this.ctx.state === 'ROUND_RESULT';

    this.topbarStatusEl.textContent = inGame
      ? `سطح ${toPersianDigits(lvl.id)} (${lvl.nameFa}) | دور ${toPersianDigits(this.ctx.roundNumber)} | شما ${toPersianDigits(this.ctx.humanScore)} - ${toPersianDigits(this.ctx.computerScore)} رایانه`
      : `فاز ۴ — سطح ${toPersianDigits(lvl.id)}: ${lvl.nameFa} (${this.ctx.ruleSet.nameFa})`;

    this.renderPanel();
    this.renderDebugPanel();
    if (showCam) {
      this.updateLiveOverlay();
    }
    if (showPixi) {
      this.syncPixiSceneState();
    }
  }

  private renderPanel(): void {
    this.panelContainer.innerHTML = '';
    const card = document.createElement('div');
    card.className = 'card';

    switch (this.ctx.state) {
      case 'SPLASH': {
        card.classList.add('hero-card');
        const badge = document.createElement('div');
        badge.className = 'hero-badge';
        badge.textContent = FA.splashPrivacyBadge;

        const h2 = document.createElement('h2');
        h2.textContent = FA.appTitle;

        const p = document.createElement('p');
        p.textContent = FA.appSubtitle;

        const startBtn = document.createElement('button');
        startBtn.type = 'button';
        startBtn.className = 'btn btn-primary';
        startBtn.textContent = FA.splashStartBtn;
        startBtn.setAttribute('data-testid', 'splash-start-btn');
        startBtn.addEventListener('click', () => this.dispatch({ type: 'GO_PERMISSION' }));

        const practiceBtn = document.createElement('button');
        practiceBtn.type = 'button';
        practiceBtn.className = 'btn btn-emerald';
        practiceBtn.textContent = FA.splashPracticeBtn;
        practiceBtn.setAttribute('data-testid', 'splash-practice-btn');
        practiceBtn.addEventListener('click', () =>
          this.dispatch({ type: 'ENTER_PRACTICE_MENU' })
        );

        const replayRow = document.createElement('div');
        replayRow.className = 'btn-row';

        const replayBtn = document.createElement('button');
        replayBtn.type = 'button';
        replayBtn.className = 'btn btn-secondary';
        replayBtn.textContent = FA.debugReplaySample;
        replayBtn.setAttribute('data-testid', 'replay-sample-btn');
        replayBtn.addEventListener('click', () =>
          this.startSessionReplay(createBuiltInSampleSession())
        );

        const transferReplayBtn = document.createElement('button');
        transferReplayBtn.type = 'button';
        transferReplayBtn.className = 'btn btn-secondary';
        transferReplayBtn.textContent = FA.debugReplayTransfer;
        transferReplayBtn.setAttribute('data-testid', 'replay-transfer-btn');
        transferReplayBtn.addEventListener('click', () =>
          this.startSessionReplay(createPhase2TransferSampleSession())
        );

        replayRow.append(replayBtn, transferReplayBtn);
        card.append(badge, h2, p, startBtn, practiceBtn, replayRow);
        break;
      }

      case 'PERMISSION': {
        const h2 = document.createElement('h2');
        h2.textContent = FA.permissionTitle;

        const body = document.createElement('p');
        body.textContent = FA.permissionBody;

        const ul = document.createElement('ul');
        ul.className = 'checklist';
        for (const itemText of FA.permissionChecklist) {
          const li = document.createElement('li');
          li.className = 'check-item ok';
          li.textContent = `✓ ${itemText}`;
          ul.append(li);
        }

        const btnRow = document.createElement('div');
        btnRow.className = 'btn-row';

        const grantBtn = document.createElement('button');
        grantBtn.type = 'button';
        grantBtn.className = 'btn btn-primary';
        grantBtn.textContent = FA.permissionGrantBtn;
        grantBtn.setAttribute('data-testid', 'grant-camera-btn');
        grantBtn.addEventListener('click', () => {
          void this.handleRequestCamera();
        });

        const practiceBtn = document.createElement('button');
        practiceBtn.type = 'button';
        practiceBtn.className = 'btn btn-emerald';
        practiceBtn.textContent = FA.splashPracticeBtn;
        practiceBtn.addEventListener('click', () =>
          this.dispatch({ type: 'ENTER_PRACTICE_MENU' })
        );

        const replayBtn = document.createElement('button');
        replayBtn.type = 'button';
        replayBtn.className = 'btn btn-secondary';
        replayBtn.textContent = FA.debugReplaySample;
        replayBtn.addEventListener('click', () =>
          this.startSessionReplay(createBuiltInSampleSession())
        );

        btnRow.append(grantBtn, practiceBtn, replayBtn);
        card.append(h2, body, ul, btnRow);

        if (this.permissionError) {
          const errBox = document.createElement('div');
          errBox.className = 'void-box';
          errBox.textContent = this.permissionError;
          card.append(errBox);
        }
        break;
      }

      case 'CALIBRATION': {
        const h2 = document.createElement('h2');
        h2.textContent = FA.calibrationTitle;

        const intro = document.createElement('p');
        intro.textContent = FA.calibrationIntro;

        const q = this.lastQuality;
        const list = document.createElement('ul');
        list.className = 'checklist';
        list.setAttribute('data-cal-checklist', 'true');

        const addCheck = (key: string, label: string, ok: boolean, hint?: string | null) => {
          const li = document.createElement('li');
          li.className = `check-item ${ok ? 'ok' : 'fail'}`;
          li.setAttribute('data-cal-key', key);
          const left = document.createElement('div');
          const title = document.createElement('div');
          title.textContent = label;
          const p = document.createElement('p');
          p.className = `check-hint${hint ? '' : ' hidden'}`;
          p.setAttribute('data-cal-hint', 'true');
          p.textContent = hint ?? '';
          left.append(title, p);
          const badge = document.createElement('span');
          badge.className = `check-status ${ok ? 'ok' : 'pending'}`;
          badge.setAttribute('data-cal-status', 'true');
          badge.textContent = ok ? 'تأیید ✓' : 'در انتظار…';
          li.append(left, badge);
          list.append(li);
        };

        addCheck('lighting', FA.calCheckLighting, Boolean(q?.lightingOk), q?.lightingHintFa);
        addCheck(
          'hands',
          FA.calCheckHands,
          this.stabilityPassed,
          this.stabilityPassed
            ? 'هر دو دست با اطمینان کافی تثبیت شدند.'
            : `پیشرفت پایداری: ${toPersianDigits(Math.round(this.stabilityProgress * 100))}٪`
        );
        addCheck('distance', FA.calCheckDistance, Boolean(q?.distanceOk), q?.distanceHintFa);
        addCheck(
          'gol',
          FA.calCheckGolColor,
          this.golColorModel !== null,
          this.golColorModel
            ? `مدل رنگ ثبت شد (${toPersianDigits(this.golColorModel.sampleCount)} نمونه پیکسل).`
            : FA.calGolInstruction
        );

        const btnRow = document.createElement('div');
        btnRow.className = 'btn-row';

        const sampleBtn = document.createElement('button');
        sampleBtn.type = 'button';
        sampleBtn.className = 'btn btn-secondary';
        sampleBtn.textContent = this.golColorModel ? FA.calRecalibrateGolBtn : FA.calSampleGolBtn;
        sampleBtn.setAttribute('data-testid', 'sample-gol-btn');
        sampleBtn.addEventListener('click', () => this.sampleGolColorFromCurrentFrame());

        const doneBtn = document.createElement('button');
        doneBtn.type = 'button';
        doneBtn.className = 'btn btn-primary';
        doneBtn.textContent = FA.calCompleteBtn;
        doneBtn.disabled = !this.isCalibrationReady();
        doneBtn.setAttribute('data-testid', 'calibration-done-btn');
        doneBtn.addEventListener('click', () => this.dispatch({ type: 'CALIBRATION_COMPLETE' }));

        btnRow.append(sampleBtn, doneBtn);

        if (this.cameraDevices.length > 1) {
          const selectWrap = document.createElement('div');
          selectWrap.style.marginTop = '12px';
          const lbl = document.createElement('label');
          lbl.textContent = FA.cameraSelectLabel;
          const select = document.createElement('select');
          select.className = 'field-select';
          for (const dev of this.cameraDevices) {
            const opt = document.createElement('option');
            opt.value = dev.deviceId;
            opt.textContent = dev.label;
            opt.selected = dev.deviceId === this.camera.currentDeviceId;
            select.append(opt);
          }
          select.addEventListener('change', () => {
            void this.camera.switchDevice(select.value);
          });
          selectWrap.append(lbl, select);
          card.append(h2, intro, list, btnRow, selectWrap);
        } else {
          card.append(h2, intro, list, btnRow);
        }

        if (this.lightingShiftDetected) {
          const warn = document.createElement('div');
          warn.className = 'void-box';
          warn.textContent = FA.calLightingShiftWarn;
          card.append(warn);
        }
        break;
      }

      case 'MENU': {
        const lvl = getLevelDefinition(this.ctx.levelId);
        const h2 = document.createElement('h2');
        h2.textContent = FA.menuTitle;

        const note = document.createElement('p');
        note.textContent = FA.menuPhaseNote;

        const levelBanner = document.createElement('div');
        levelBanner.className = 'check-item ok';
        levelBanner.setAttribute('data-testid', 'menu-active-level');
        levelBanner.innerHTML = '';
        const lblDiv = document.createElement('div');
        lblDiv.textContent = `سطح فعلی: ${toPersianDigits(lvl.id)}. ${lvl.nameFa} — ${lvl.subtitleFa}`;
        const modeSpan = document.createElement('span');
        modeSpan.className = 'check-status ok';
        modeSpan.textContent = this.ctx.practiceMode ? 'حالت تمرینی' : 'دوربین فعال';
        levelBanner.append(lblDiv, modeSpan);

        const ruleSummary = document.createElement('div');
        ruleSummary.className = 'check-item';
        ruleSummary.textContent = `${this.ctx.ruleSet.nameFa} (${toPersianDigits(this.ctx.ruleSet.handsCount)} دست): ${this.ctx.ruleSet.descriptionFa}`;

        const btnRow = document.createElement('div');
        btnRow.className = 'btn-row';

        const playBtn = document.createElement('button');
        playBtn.type = 'button';
        playBtn.className = 'btn btn-primary';
        playBtn.textContent = FA.menuPlayBtn;
        playBtn.setAttribute('data-testid', 'menu-play-btn');
        playBtn.addEventListener('click', () => this.dispatch({ type: 'START_GAME' }));

        const levelsBtn = document.createElement('button');
        levelsBtn.type = 'button';
        levelsBtn.className = 'btn btn-secondary';
        levelsBtn.textContent = FA.menuLevelsBtn;
        levelsBtn.setAttribute('data-testid', 'menu-levels-btn');
        levelsBtn.addEventListener('click', () => this.dispatch({ type: 'OPEN_LEVEL_SELECT' }));

        const statsBtn = document.createElement('button');
        statsBtn.type = 'button';
        statsBtn.className = 'btn btn-secondary';
        statsBtn.textContent = FA.menuStatsBtn;
        statsBtn.setAttribute('data-testid', 'menu-stats-btn');
        statsBtn.addEventListener('click', () => this.dispatch({ type: 'OPEN_STATS' }));

        const rulesBtn = document.createElement('button');
        rulesBtn.type = 'button';
        rulesBtn.className = 'btn btn-secondary';
        rulesBtn.textContent = FA.menuRulesBtn;
        rulesBtn.setAttribute('data-testid', 'menu-rules-btn');
        rulesBtn.addEventListener('click', () => this.dispatch({ type: 'OPEN_RULES' }));

        const consentBtn = document.createElement('button');
        consentBtn.type = 'button';
        consentBtn.className = 'btn btn-secondary';
        consentBtn.textContent = FA.menuConsentBtn;
        consentBtn.setAttribute('data-testid', 'menu-consent-btn');
        consentBtn.addEventListener('click', () => this.dispatch({ type: 'OPEN_CONSENT' }));

        const calBtn = document.createElement('button');
        calBtn.type = 'button';
        calBtn.className = 'btn btn-secondary';
        calBtn.textContent = FA.menuRecalibrateBtn;
        calBtn.addEventListener('click', () => this.dispatch({ type: 'OPEN_CALIBRATION' }));

        btnRow.append(playBtn, levelsBtn, statsBtn, rulesBtn, consentBtn, calBtn);

        const targetWrap = document.createElement('div');
        targetWrap.style.marginTop = '14px';
        const targetLbl = document.createElement('label');
        targetLbl.textContent = FA.targetScoreLabel;
        const targetSelect = document.createElement('select');
        targetSelect.className = 'field-select';
        targetSelect.setAttribute('data-testid', 'target-score-select');
        for (const scoreOpt of [3, 5, 7, 10]) {
          const opt = document.createElement('option');
          opt.value = String(scoreOpt);
          opt.textContent = `${toPersianDigits(scoreOpt)} امتیاز`;
          opt.selected = this.ctx.ruleSet.targetScore === scoreOpt;
          targetSelect.append(opt);
        }
        targetSelect.addEventListener('change', () => {
          this.dispatch({ type: 'SET_TARGET_SCORE', targetScore: Number(targetSelect.value) });
        });
        targetWrap.append(targetLbl, targetSelect);

        const settingsHeading = document.createElement('h3');
        settingsHeading.style.marginTop = '18px';
        settingsHeading.textContent = FA.menuSettingsHeading;

        const settingsBox = document.createElement('div');
        const consentRow = this.makeToggleRow(
          FA.consentToggleLabel,
          this.consentSettings.consentEnabled,
          (val) => {
            void this.handleToggleConsent(val);
          }
        );
        consentRow
          .querySelector('input')
          ?.setAttribute('data-testid', 'settings-consent-toggle');

        settingsBox.append(
          this.makeToggleRow(FA.calMirrorToggle, this.mirrored, (val) => {
            this.mirrored = val;
            this.renderUi();
          }),
          this.makeToggleRow(FA.calLandmarkToggle, this.showLandmarks, (val) => {
            this.showLandmarks = val;
            this.renderUi();
          }),
          this.makeToggleRow(FA.reducedMotionLabel, this.reducedMotion, (val) => {
            this.reducedMotion = val;
            this.renderUi();
          }),
          this.makeToggleRow(FA.debugToggleLabel, this.showDebug, (val) => {
            this.showDebug = val;
            this.renderDebugPanel();
          }),
          consentRow
        );

        const memRow = document.createElement('div');
        memRow.className = 'btn-row';
        const memStat = document.createElement('span');
        memStat.className = 'check-hint';
        memStat.textContent = `دورهای ثبت‌شده: ${toPersianDigits(this.progression.totalRoundsPlayed)} | لاگ عددی (فاز ۴): ${toPersianDigits(this.loggedRoundsCount)} (${this.consentSettings.consentEnabled ? 'فعال' : 'پیش‌فرض خاموش'})`;

        const exportLogsBtn = document.createElement('button');
        exportLogsBtn.type = 'button';
        exportLogsBtn.className = 'btn btn-secondary btn-compact';
        exportLogsBtn.textContent = FA.exportLogsBtn;
        exportLogsBtn.setAttribute('data-testid', 'settings-export-logs-btn');
        exportLogsBtn.addEventListener('click', () => {
          void this.handleExportLogs();
        });

        const clearMemBtn = document.createElement('button');
        clearMemBtn.type = 'button';
        clearMemBtn.className = 'btn btn-secondary btn-compact';
        clearMemBtn.textContent = FA.resetProfileMemoryBtn;
        clearMemBtn.addEventListener('click', () => {
          this.behaviorHistory = [];
          this.ensemble.restore({ weights: {}, cumulativeLosses: {}, roundsUpdated: 0 });
          void this.profileStore.clearProfile(DEFAULT_PROFILE_ID);
          void this.progressionStore.reset(DEFAULT_PROFILE_ID).then((fresh) => {
            this.progression = fresh;
            this.renderUi();
          });
        });
        memRow.append(memStat, exportLogsBtn, clearMemBtn);

        card.append(
          h2,
          note,
          levelBanner,
          ruleSummary,
          btnRow,
          targetWrap,
          settingsHeading,
          settingsBox,
          memRow
        );
        if (this.logNoticeFa) {
          const n = document.createElement('div');
          n.className = `status-notice${this.logNoticeError ? ' error' : ''}`;
          n.setAttribute('data-testid', 'log-status-notice');
          n.textContent = this.logNoticeFa;
          card.append(n);
        }
        break;
      }

      case 'LEVEL_SELECT': {
        const h2 = document.createElement('h2');
        h2.textContent = FA.levelSelectTitle;

        const unlockedMax = this.progression.unlockedMaxLevel;
        const intro = document.createElement('p');
        intro.textContent = `با پیروزی در هر سطح، سطح بعدی به‌صورت خودکار باز می‌شود. سطوح باز: تا سطح ${toPersianDigits(unlockedMax)}.`;

        const grid = document.createElement('div');
        grid.className = 'level-grid';
        grid.setAttribute('data-testid', 'level-select-grid');

        for (const lvl of LEVELS) {
          const unlocked = isLevelUnlocked(lvl.id, unlockedMax);
          const completed = this.progression.completedLevels.includes(lvl.id);
          const selected = this.ctx.levelId === lvl.id;

          const item = document.createElement('div');
          item.className = `level-card${selected ? ' selected' : ''}${unlocked ? '' : ' locked'}`;
          item.setAttribute('data-testid', `level-card-${lvl.id}`);

          const header = document.createElement('div');
          header.className = 'level-card-header';
          const title = document.createElement('strong');
          title.textContent = `سطح ${toPersianDigits(lvl.id)}: ${lvl.nameFa}`;
          const badge = document.createElement('span');
          badge.className = `check-status ${unlocked ? 'ok' : 'pending'}`;
          badge.textContent = completed
            ? 'تکمیل‌شده 🏆'
            : unlocked
              ? selected
                ? 'انتخاب‌شده ✓'
                : 'باز'
              : 'قفل 🔒';
          header.append(title, badge);

          const desc = document.createElement('p');
          desc.className = 'check-hint';
          desc.textContent = lvl.descriptionFa;

          const actions = document.createElement('div');
          actions.className = 'btn-row';

          const pickBtn = document.createElement('button');
          pickBtn.type = 'button';
          pickBtn.className = selected ? 'btn btn-primary btn-compact' : 'btn btn-secondary btn-compact';
          pickBtn.textContent = selected ? 'شروع این سطح' : 'انتخاب سطح';
          pickBtn.disabled = !unlocked;
          pickBtn.setAttribute('data-testid', `select-level-btn-${lvl.id}`);
          pickBtn.addEventListener('click', () => {
            if (!unlocked) return;
            this.dispatch({
              type: 'SELECT_LEVEL',
              levelId: lvl.id,
              teamHandsCount: this.ctx.teamHandsCount,
            });
            this.dispatch({ type: 'START_GAME' });
          });

          actions.append(pickBtn);
          item.append(header, desc, actions);
          grid.append(item);
        }

        const backBtn = document.createElement('button');
        backBtn.type = 'button';
        backBtn.className = 'btn btn-secondary';
        backBtn.textContent = FA.exitToMenuBtn;
        backBtn.addEventListener('click', () => this.dispatch({ type: 'EXIT_TO_MENU' }));

        card.append(h2, intro, grid, backBtn);
        break;
      }

      case 'STATS': {
        const h2 = document.createElement('h2');
        h2.textContent = FA.statsTitle;

        const p = this.progression;
        const winRatePct =
          p.totalRoundsPlayed > 0
            ? Math.round((p.humanRoundsWon / p.totalRoundsPlayed) * 100)
            : 0;

        const grid = document.createElement('div');
        grid.className = 'stats-grid';
        grid.setAttribute('data-testid', 'stats-grid');

        const tiles: Array<[string, string]> = [
          ['بالاترین سطح باز', `${toPersianDigits(p.unlockedMaxLevel)} از ۵`],
          ['مسابقات برده / کل', `${toPersianDigits(p.totalMatchesWon)} / ${toPersianDigits(p.totalMatchesPlayed)}`],
          ['دورهای برده / کل', `${toPersianDigits(p.humanRoundsWon)} / ${toPersianDigits(p.totalRoundsPlayed)}`],
          ['نرخ برد دورها', `${toPersianDigits(winRatePct)}٪`],
          ['بیشترین برد پیاپی', toPersianDigits(p.humanBestWinStreak)],
          ['گل مستقیم / پس از پوچ', `${toPersianDigits(p.humanDirectGolCount)} / ${toPersianDigits(p.humanCorrectAfterPoochCount)}`],
        ];

        for (const [label, val] of tiles) {
          const tile = document.createElement('div');
          tile.className = 'stat-tile';
          const s = document.createElement('strong');
          s.textContent = val;
          const lbl = document.createElement('span');
          lbl.className = 'check-hint';
          lbl.textContent = label;
          tile.append(s, lbl);
          grid.append(tile);
        }

        const levelList = document.createElement('ul');
        levelList.className = 'checklist';
        for (const lvl of LEVELS) {
          const st = p.perLevelStats[lvl.id] ?? { played: 0, won: 0, bestScoreDiff: 0 };
          const li = document.createElement('li');
          li.className = 'check-item';
          li.textContent = `سطح ${toPersianDigits(lvl.id)} (${lvl.nameFa}): ${toPersianDigits(st.won)} برد از ${toPersianDigits(st.played)} مسابقه`;
          levelList.append(li);
        }

        const logCard = document.createElement('div');
        logCard.className = 'consent-box';
        logCard.setAttribute('data-testid', 'stats-logs-section');
        const logTitle = document.createElement('strong');
        logTitle.textContent = `لاگ‌های عددی یادگیری (فاز ۴): ${toPersianDigits(this.loggedRoundsCount)} دور ثبت‌شده`;
        const logHint = document.createElement('p');
        logHint.className = 'check-hint';
        logHint.textContent = this.consentSettings.consentEnabled
          ? FA.consentActiveBadge
          : FA.consentDefaultOffBadge;

        const logBtns = document.createElement('div');
        logBtns.className = 'btn-row';

        const expBtn = document.createElement('button');
        expBtn.type = 'button';
        expBtn.className = 'btn btn-primary btn-compact';
        expBtn.textContent = FA.exportLogsBtn;
        expBtn.setAttribute('data-testid', 'stats-export-logs-btn');
        expBtn.addEventListener('click', () => {
          void this.handleExportLogs();
        });

        const impInput = document.createElement('input');
        impInput.type = 'file';
        impInput.accept = 'application/json,.json';
        impInput.className = 'hidden';
        impInput.addEventListener('change', () => {
          const f = impInput.files?.[0];
          if (f) void this.handleImportLogs(f);
        });

        const impBtn = document.createElement('button');
        impBtn.type = 'button';
        impBtn.className = 'btn btn-secondary btn-compact';
        impBtn.textContent = FA.importLogsBtn;
        impBtn.setAttribute('data-testid', 'stats-import-logs-btn');
        impBtn.addEventListener('click', () => impInput.click());

        const clrBtn = document.createElement('button');
        clrBtn.type = 'button';
        clrBtn.className = 'btn btn-danger btn-compact';
        clrBtn.textContent = FA.clearLogsBtn;
        clrBtn.setAttribute('data-testid', 'stats-clear-logs-btn');
        clrBtn.addEventListener('click', () => {
          void this.handleClearLogs();
        });

        const openConsentBtn = document.createElement('button');
        openConsentBtn.type = 'button';
        openConsentBtn.className = 'btn btn-secondary btn-compact';
        openConsentBtn.textContent = FA.menuConsentBtn;
        openConsentBtn.setAttribute('data-testid', 'stats-open-consent-btn');
        openConsentBtn.addEventListener('click', () =>
          this.dispatch({ type: 'OPEN_CONSENT' })
        );

        logBtns.append(expBtn, impBtn, clrBtn, openConsentBtn, impInput);
        logCard.append(logTitle, logHint, logBtns);

        if (this.logNoticeFa) {
          const n = document.createElement('div');
          n.className = `status-notice${this.logNoticeError ? ' error' : ''}`;
          n.setAttribute('data-testid', 'log-status-notice');
          n.textContent = this.logNoticeFa;
          logCard.append(n);
        }

        const backBtn = document.createElement('button');
        backBtn.type = 'button';
        backBtn.className = 'btn btn-secondary';
        backBtn.textContent = FA.exitToMenuBtn;
        backBtn.addEventListener('click', () => this.dispatch({ type: 'EXIT_TO_MENU' }));

        card.append(h2, grid, levelList, logCard, backBtn);
        break;
      }

      case 'CONSENT': {
        card.setAttribute('data-testid', 'consent-screen');
        const h2 = document.createElement('h2');
        h2.textContent = FA.consentTitle;

        const statusBadge = document.createElement('div');
        statusBadge.className = this.consentSettings.consentEnabled
          ? 'contact-req-pill done'
          : 'contact-req-pill pending';
        statusBadge.setAttribute('data-testid', 'consent-status-badge');
        statusBadge.textContent = this.consentSettings.consentEnabled
          ? FA.consentActiveBadge
          : FA.consentDefaultOffBadge;

        const body = document.createElement('p');
        body.textContent = FA.consentExplanation;

        const checklist = document.createElement('ul');
        checklist.className = 'checklist';
        for (const text of FA.consentChecklist) {
          const li = document.createElement('li');
          li.className = 'check-item ok';
          li.textContent = `✓ ${text}`;
          checklist.append(li);
        }

        const toggleBox = document.createElement('div');
        toggleBox.className = 'consent-box';
        const toggleRow = this.makeToggleRow(
          FA.consentToggleLabel,
          this.consentSettings.consentEnabled,
          (val) => {
            void this.handleToggleConsent(val);
          }
        );
        toggleRow
          .querySelector('input')
          ?.setAttribute('data-testid', 'consent-toggle-checkbox');

        const countInfo = document.createElement('p');
        countInfo.className = 'check-hint';
        countInfo.setAttribute('data-testid', 'consent-log-count');
        countInfo.textContent = `تعداد دورهای عددی ذخیره‌شده در IndexedDB: ${toPersianDigits(this.loggedRoundsCount)}`;
        toggleBox.append(toggleRow, countInfo);

        const btnRow = document.createElement('div');
        btnRow.className = 'btn-row';

        const expBtn = document.createElement('button');
        expBtn.type = 'button';
        expBtn.className = 'btn btn-primary';
        expBtn.textContent = FA.exportLogsBtn;
        expBtn.setAttribute('data-testid', 'consent-export-btn');
        expBtn.addEventListener('click', () => {
          void this.handleExportLogs();
        });

        const impInput = document.createElement('input');
        impInput.type = 'file';
        impInput.accept = 'application/json,.json';
        impInput.className = 'hidden';
        impInput.setAttribute('data-testid', 'consent-import-input');
        impInput.addEventListener('change', () => {
          const f = impInput.files?.[0];
          if (f) void this.handleImportLogs(f);
        });

        const impBtn = document.createElement('button');
        impBtn.type = 'button';
        impBtn.className = 'btn btn-secondary';
        impBtn.textContent = FA.importLogsBtn;
        impBtn.setAttribute('data-testid', 'consent-import-btn');
        impBtn.addEventListener('click', () => impInput.click());

        const clrBtn = document.createElement('button');
        clrBtn.type = 'button';
        clrBtn.className = 'btn btn-danger';
        clrBtn.textContent = FA.clearLogsBtn;
        clrBtn.setAttribute('data-testid', 'consent-clear-btn');
        clrBtn.addEventListener('click', () => {
          void this.handleClearLogs();
        });

        const backBtn = document.createElement('button');
        backBtn.type = 'button';
        backBtn.className = 'btn btn-secondary';
        backBtn.textContent = FA.exitToMenuBtn;
        backBtn.setAttribute('data-testid', 'consent-back-btn');
        backBtn.addEventListener('click', () => this.dispatch({ type: 'EXIT_TO_MENU' }));

        btnRow.append(expBtn, impBtn, clrBtn, backBtn, impInput);
        card.append(h2, statusBadge, body, checklist, toggleBox, btnRow);

        if (this.logNoticeFa) {
          const n = document.createElement('div');
          n.className = `status-notice${this.logNoticeError ? ' error' : ''}`;
          n.setAttribute('data-testid', 'log-status-notice');
          n.textContent = this.logNoticeFa;
          card.append(n);
        }
        if (this.lastLogExportJson) {
          const pre = document.createElement('pre');
          pre.className = 'how-it-works-box ltr-inline';
          pre.style.maxHeight = '120px';
          pre.style.overflow = 'auto';
          pre.style.width = '100%';
          pre.setAttribute('data-testid', 'export-json-preview');
          pre.textContent = this.lastLogExportJson.slice(0, 600);
          card.append(pre);
        }
        break;
      }

      case 'RULES': {
        const h2 = document.createElement('h2');
        h2.textContent = FA.rulesTitle;

        const teamCountWrap = document.createElement('div');
        teamCountWrap.style.marginBottom = '12px';
        const teamLbl = document.createElement('label');
        teamLbl.textContent = 'تعداد دست‌ها در حالت تیمی (traditional-team):';
        const teamSel = document.createElement('select');
        teamSel.className = 'field-select';
        teamSel.setAttribute('data-testid', 'team-hands-select');
        for (const countOpt of [4, 6] as const) {
          const opt = document.createElement('option');
          opt.value = String(countOpt);
          opt.textContent = `${toPersianDigits(countOpt)} دست`;
          opt.selected = this.ctx.teamHandsCount === countOpt;
          teamSel.append(opt);
        }
        teamSel.addEventListener('change', () => {
          this.dispatch({
            type: 'SET_TEAM_HANDS_COUNT',
            handsCount: Number(teamSel.value) === 6 ? 6 : 4,
          });
        });
        teamCountWrap.append(teamLbl, teamSel);

        const list = document.createElement('ul');
        list.className = 'checklist';
        for (const rule of Object.values(RULE_PRESETS)) {
          const li = document.createElement('li');
          li.className = `check-item${this.ctx.ruleSet.id === rule.id ? ' ok' : ''}`;
          const left = document.createElement('div');
          const title = document.createElement('strong');
          title.textContent = `${rule.nameFa} (${rule.id})`;
          const desc = document.createElement('p');
          desc.className = 'check-hint';
          desc.textContent = `${rule.descriptionFa} | دست‌ها: ${toPersianDigits(rule.handsCount)} | سقف حدس: ${toPersianDigits(rule.maxGuessesPerRound)}`;
          left.append(title, desc);

          const useBtn = document.createElement('button');
          useBtn.type = 'button';
          useBtn.className = 'btn btn-secondary btn-compact';
          useBtn.textContent = this.ctx.ruleSet.id === rule.id ? 'فعال ✓' : 'اعمال قانون';
          useBtn.setAttribute('data-testid', `apply-rule-${rule.id}`);
          useBtn.addEventListener('click', () => {
            this.dispatch({ type: 'SET_CUSTOM_RULES', ruleSet: rule });
          });

          li.append(left, useBtn);
          list.append(li);
        }

        const backBtn = document.createElement('button');
        backBtn.type = 'button';
        backBtn.className = 'btn btn-secondary';
        backBtn.textContent = FA.exitToMenuBtn;
        backBtn.addEventListener('click', () => this.dispatch({ type: 'EXIT_TO_MENU' }));

        card.append(h2, teamCountWrap, list, backBtn);
        break;
      }

      case 'SHOW_GOL':
      case 'HIDING':
      case 'COMPUTER_GUESS':
      case 'REVEAL_VERIFY':
      case 'COMPUTER_HIDES':
      case 'HUMAN_GUESS': {
        card.append(this.makeHudRow());
        if (this.ctx.bestOfMatches === 3) {
          const sb = document.createElement('div');
          sb.className = 'series-banner';
          sb.setAttribute('data-testid', 'series-banner');
          sb.textContent = `سری بهترین از ۳ (سطح ۵) — مسابقه ${toPersianDigits(this.ctx.seriesMatchIndex + 1)} (${this.ctx.ruleSet.nameFa}) | برد سری: شما ${toPersianDigits(this.ctx.seriesHumanWins)} - ${toPersianDigits(this.ctx.seriesComputerWins)} رایانه`;
          card.append(sb);
        }

        if (this.ctx.state === 'SHOW_GOL') {
          const h2 = document.createElement('h2');
          h2.textContent = FA.showGolPrompt;
          const p = document.createElement('p');
          p.setAttribute('data-pinch-progress', 'true');
          const framesDone = this.lastPinchState?.consecutiveFrames ?? 0;
          const framesReq = this.lastPinchState?.requiredFrames ?? 15;
          p.textContent = `تثبیت پینچ: ${toPersianDigits(framesDone)} از ${toPersianDigits(framesReq)} فریم پیاپی`;
          card.append(h2, p);
        } else if (this.ctx.state === 'HIDING') {
          // IMPORTANT: Never show tracking belief to player during HIDING!
          const h2 = document.createElement('h2');
          h2.textContent = FA.hidingIndicator;
          const p = document.createElement('p');
          p.textContent = FA.hidingInstruction;
          card.append(h2, p);

          if (this.ctx.config.minHandContactsRequired > 0) {
            const met = this.ctx.hidingContactCount >= this.ctx.config.minHandContactsRequired;
            const cPill = document.createElement('div');
            cPill.className = `contact-req-pill ${met ? 'done' : 'pending'}`;
            cPill.setAttribute('data-testid', 'contact-requirement-status');
            cPill.textContent = met
              ? 'الزام سطح حرفه‌ای: تماس دست‌ها ثبت شد ✓'
              : 'الزام سطح حرفه‌ای: حداقل یک بار دست‌ها را به هم بزنید (در انتظار تماس…)';
            card.append(cPill);
          }

          const doneBtn = document.createElement('button');
          doneBtn.type = 'button';
          doneBtn.className = 'btn btn-primary';
          doneBtn.textContent = FA.finishHidingBtn;
          doneBtn.setAttribute('data-testid', 'finish-hiding-btn');
          doneBtn.addEventListener('click', () => {
            void this.resolveHidingAndFinish(this.getCurrentContactCount());
          });
          card.append(doneBtn);
        } else if (this.ctx.state === 'COMPUTER_GUESS') {
          const h2 = document.createElement('h2');
          h2.textContent = FA.guessPhaseTitle;

          const guessBox = document.createElement('div');
          guessBox.className = 'guess-card';
          const lbl = document.createElement('div');
          lbl.textContent = 'حدس رایانه:';
          const handEl = document.createElement('div');
          handEl.className = 'guess-hand-big';
          handEl.setAttribute('data-testid', 'computer-guess-hand');
          handEl.textContent = this.ctx.computerGuess ? sideLabelFa(this.ctx.computerGuess) : '-';

          const probPct = Math.round((this.ctx.computerGuessProb ?? 0.5) * 100);
          const probEl = document.createElement('div');
          probEl.className = 'guess-prob-pill';
          probEl.setAttribute('data-testid', 'guess-probability');
          probEl.textContent = `اطمینان حدس: ${toPersianDigits(probPct)}٪`;

          guessBox.append(lbl, handEl, probEl);

          const nextBtn = document.createElement('button');
          nextBtn.type = 'button';
          nextBtn.className = 'btn btn-primary';
          nextBtn.textContent = FA.proceedToRevealBtn;
          nextBtn.setAttribute('data-testid', 'proceed-reveal-btn');
          nextBtn.addEventListener('click', () => this.dispatch({ type: 'PROCEED_TO_REVEAL' }));

          card.append(h2, guessBox, nextBtn);
        } else if (this.ctx.state === 'REVEAL_VERIFY') {
          const h2 = document.createElement('h2');
          h2.textContent = FA.revealPrompt;
          const p = document.createElement('p');
          p.textContent = FA.revealWaitingHands;
          card.append(h2, p);
        } else if (this.ctx.state === 'COMPUTER_HIDES') {
          const h2 = document.createElement('h2');
          h2.textContent = FA.computerHidesTitle;
          const p = document.createElement('p');
          p.textContent = this.reducedMotion ? FA.computerHidesReducedDesc : FA.computerHidesDesc;

          const commitBox = this.makeCommitFingerprintBox();
          card.append(h2, p, commitBox);
        } else {
          // HUMAN_GUESS
          const h2 = document.createElement('h2');
          h2.textContent = FA.humanGuessTitle;
          const p = document.createElement('p');
          p.textContent = this.ctx.ruleSet.allowPoochCalls
            ? FA.humanGuessPromptTeam
            : FA.humanGuessPromptClassic;

          const commitBox = this.makeCommitFingerprintBox();
          card.append(h2, p, commitBox);

          if (this.ctx.ruleSet.allowPoochCalls) {
            const remGuesses = Math.max(
              0,
              this.ctx.ruleSet.maxGuessesPerRound - this.ctx.teamGuessProgress.guessesUsed
            );
            const modeBar = document.createElement('div');
            modeBar.className = 'call-mode-bar';

            const infoSpan = document.createElement('span');
            infoSpan.setAttribute('data-testid', 'remaining-guesses-info');
            infoSpan.textContent = `حدس‌های باقیمانده: ${toPersianDigits(remGuesses)} از ${toPersianDigits(this.ctx.ruleSet.maxGuessesPerRound)} | پوچ‌های درست: ${toPersianDigits(this.ctx.teamGuessProgress.poochCallsCount)}`;

            const toggleRow = document.createElement('div');
            toggleRow.className = 'btn-row';
            toggleRow.style.marginTop = '0';

            const poochModeBtn = document.createElement('button');
            poochModeBtn.type = 'button';
            poochModeBtn.className =
              this.activeTeamCallMode === 'POOCH'
                ? 'btn btn-sky btn-compact'
                : 'btn btn-secondary btn-compact';
            poochModeBtn.textContent = 'حالت ضربه: پوچ';
            poochModeBtn.setAttribute('data-testid', 'mode-pooch-btn');
            poochModeBtn.addEventListener('click', () => {
              this.activeTeamCallMode = 'POOCH';
              this.renderPanel();
            });

            const golModeBtn = document.createElement('button');
            golModeBtn.type = 'button';
            golModeBtn.className =
              this.activeTeamCallMode === 'GOL'
                ? 'btn btn-primary btn-compact'
                : 'btn btn-secondary btn-compact';
            golModeBtn.textContent = 'حالت ضربه: گل';
            golModeBtn.setAttribute('data-testid', 'mode-gol-btn');
            golModeBtn.addEventListener('click', () => {
              this.activeTeamCallMode = 'GOL';
              this.renderPanel();
            });

            toggleRow.append(poochModeBtn, golModeBtn);
            modeBar.append(infoSpan, toggleRow);
            card.append(modeBar);
          }

          const actionsGrid = document.createElement('div');
          actionsGrid.className = 'hand-actions-grid';
          actionsGrid.setAttribute('data-testid', 'hand-actions-grid');
          const count = this.ctx.ruleSet.handsCount;

          for (let i = 0; i < count; i++) {
            const isOpened = this.ctx.teamGuessProgress.openedEmptyHands.includes(i);
            const hCard = document.createElement('div');
            hCard.className = `hand-action-card${isOpened ? ' opened' : ''}${this.selectedHandIndex === i ? ' selected' : ''}`;

            const lbl = document.createElement('strong');
            lbl.textContent = handIndexLabelFa(i, count);
            hCard.append(lbl);

            if (isOpened) {
              const tag = document.createElement('span');
              tag.className = 'check-hint';
              tag.textContent = 'باز شد: پوچ ✓';
              hCard.append(tag);
            } else if (this.ctx.ruleSet.allowPoochCalls) {
              const btnPair = document.createElement('div');
              btnPair.className = 'hand-action-btns';

              const poochBtn = document.createElement('button');
              poochBtn.type = 'button';
              poochBtn.className = 'btn btn-sky btn-compact';
              poochBtn.textContent = FA.poochActionBtn;
              poochBtn.disabled = !this.ctx.commitPublic;
              poochBtn.setAttribute('data-testid', `hand-pooch-btn-${i}`);
              poochBtn.addEventListener('click', () => {
                this.selectedHandIndex = i;
                this.dispatch({ type: 'HUMAN_GUESS_CALL', handIndex: i, callType: 'POOCH' });
              });

              const golBtn = document.createElement('button');
              golBtn.type = 'button';
              golBtn.className = 'btn btn-primary btn-compact';
              golBtn.textContent = FA.golActionBtn;
              golBtn.disabled = !this.ctx.commitPublic;
              golBtn.setAttribute('data-testid', `hand-gol-btn-${i}`);
              golBtn.addEventListener('click', () => {
                this.selectedHandIndex = i;
                this.dispatch({ type: 'HUMAN_GUESS_CALL', handIndex: i, callType: 'GOL' });
              });

              btnPair.append(poochBtn, golBtn);
              hCard.append(btnPair);
            } else {
              const pickBtn = document.createElement('button');
              pickBtn.type = 'button';
              pickBtn.className = 'btn btn-primary btn-compact';
              pickBtn.textContent = 'انتخاب این دست';
              pickBtn.disabled = !this.ctx.commitPublic;
              pickBtn.setAttribute('data-testid', `hand-guess-btn-${i}`);
              pickBtn.addEventListener('click', () => {
                this.selectedHandIndex = i;
                this.dispatch({ type: 'HUMAN_GUESS_CALL', handIndex: i, callType: 'GOL' });
              });
              hCard.append(pickBtn);
            }

            actionsGrid.append(hCard);
          }

          card.append(actionsGrid);
        }

        const exitRow = document.createElement('div');
        exitRow.className = 'btn-row';
        const exitBtn = document.createElement('button');
        exitBtn.type = 'button';
        exitBtn.className = 'btn btn-secondary';
        exitBtn.textContent = FA.exitToMenuBtn;
        exitBtn.addEventListener('click', () => {
          this.stopReplay();
          this.pixiScene?.stopChoreography();
          this.dispatch({ type: 'EXIT_TO_MENU' });
        });
        exitRow.append(exitBtn);
        card.append(exitRow);
        break;
      }

      case 'ROUND_RESULT': {
        card.append(this.makeHudRow());
        const h2 = document.createElement('h2');
        h2.textContent =
          this.ctx.lastRoundWinner === 'COMPUTER'
            ? FA.roundResultComputerWon
            : FA.roundResultHumanWon;

        if (this.ctx.bestOfMatches === 3) {
          const sb = document.createElement('div');
          sb.className = 'series-banner';
          sb.textContent = `سری بهترین از ۳ — برد سری: شما ${toPersianDigits(this.ctx.seriesHumanWins)} - ${toPersianDigits(this.ctx.seriesComputerWins)} رایانه`;
          card.append(sb);
        }

        const effectiveMatchWinner =
          this.ctx.bestOfMatches === 3 ? this.ctx.seriesWinner : this.ctx.matchWinner;

        if (effectiveMatchWinner !== null) {
          const winBanner = document.createElement('div');
          winBanner.className = 'hero-badge';
          winBanner.setAttribute('data-testid', 'match-winner-banner');
          winBanner.textContent =
            effectiveMatchWinner === 'HUMAN' ? FA.matchWinnerHuman : FA.matchWinnerComputer;
          card.append(winBanner);
        } else if (this.ctx.matchWinner !== null && this.ctx.bestOfMatches === 3) {
          const subBanner = document.createElement('div');
          subBanner.className = 'hero-badge';
          subBanner.textContent =
            this.ctx.matchWinner === 'HUMAN'
              ? 'شما این مسابقه از سری ۳ تایی را بردید!'
              : 'رایانه این مسابقه از سری ۳ تایی را برد.';
          card.append(subBanner);
        }

        if (this.lastUnlockedNoticeFa) {
          const unlockBanner = document.createElement('div');
          unlockBanner.className = 'hero-badge';
          unlockBanner.setAttribute('data-testid', 'level-unlocked-banner');
          unlockBanner.textContent = `🔓 ${this.lastUnlockedNoticeFa}`;
          card.append(unlockBanner);
        }

        const targetProgress = document.createElement('p');
        targetProgress.className = 'check-hint';
        targetProgress.setAttribute('data-testid', 'match-target-progress');
        targetProgress.textContent = `امتیاز مسابقه (هدف ${toPersianDigits(this.ctx.ruleSet.targetScore)}): شما ${toPersianDigits(this.ctx.humanScore)} — رایانه ${toPersianDigits(this.ctx.computerScore)}`;

        card.append(h2, targetProgress);

        // Computer-hides round details + cryptographic commit-reveal verification
        if (this.ctx.commitPublic && this.ctx.commitSecret) {
          const secretIdx = this.ctx.commitSecret.handIndex;
          const reasonFa = this.describeComputerHidesReasonFa();
          const detail = document.createElement('p');
          detail.setAttribute('data-testid', 'computer-hides-result-detail');
          detail.textContent = `گل در ${handIndexLabelFa(secretIdx, this.ctx.ruleSet.handsCount)} بود | ${reasonFa}`;
          card.append(detail, this.makeCommitVerificationBox());
        } else {
          // Human-hides round details & timeline explanation
          const probPct = Math.round((this.ctx.computerGuessProb ?? 0.5) * 100);
          const detail = document.createElement('p');
          detail.textContent = `گل در ${this.ctx.revealedHand ? sideLabelFa(this.ctx.revealedHand) : '-'} بود | حدس رایانه: ${this.ctx.computerGuess ? sideLabelFa(this.ctx.computerGuess) : '-'} (${toPersianDigits(probPct)}٪)`;

          const explainBox = document.createElement('div');
          explainBox.className = 'explanation-box';
          explainBox.setAttribute('data-testid', 'round-explanation');
          const explainTitle = document.createElement('strong');
          explainTitle.textContent = 'تحلیل رایانه:';
          const explainBody = document.createElement('p');
          explainBody.textContent =
            this.ctx.explanationFa ?? 'بر اساس نشانه‌های حرکتی تصمیم‌گیری شد.';
          explainBox.append(explainTitle, explainBody);

          if (this.ctx.timelineEvents.length > 0) {
            const tlList = document.createElement('ul');
            tlList.className = 'timeline-list';
            tlList.setAttribute('data-testid', 'round-timeline');
            for (const item of this.ctx.timelineEvents) {
              const li = document.createElement('li');
              li.textContent = item;
              tlList.append(li);
            }
            explainBox.append(tlList);
          }

          card.append(detail, explainBox);
        }

        const btnRow = document.createElement('div');
        btnRow.className = 'btn-row';

        if (effectiveMatchWinner === null) {
          const nextBtn = document.createElement('button');
          nextBtn.type = 'button';
          nextBtn.className = 'btn btn-primary';
          nextBtn.setAttribute('data-testid', 'next-round-btn');
          nextBtn.textContent =
            this.ctx.matchWinner !== null && this.ctx.bestOfMatches === 3
              ? FA.nextSeriesMatchBtn
              : FA.nextRoundBtn;
          nextBtn.addEventListener('click', () => this.dispatch({ type: 'NEXT_ROUND' }));
          btnRow.append(nextBtn);
        }

        const resetBtn = document.createElement('button');
        resetBtn.type = 'button';
        resetBtn.className =
          effectiveMatchWinner !== null ? 'btn btn-primary' : 'btn btn-secondary';
        resetBtn.textContent = FA.resetMatchBtn;
        resetBtn.setAttribute('data-testid', 'reset-match-btn');
        resetBtn.addEventListener('click', () => this.dispatch({ type: 'RESET_MATCH' }));

        const exitBtn = document.createElement('button');
        exitBtn.type = 'button';
        exitBtn.className = 'btn btn-secondary';
        exitBtn.textContent = FA.exitToMenuBtn;
        exitBtn.addEventListener('click', () => this.dispatch({ type: 'EXIT_TO_MENU' }));

        btnRow.append(resetBtn, exitBtn);
        card.append(btnRow);
        break;
      }

      case 'ROUND_VOID': {
        card.append(this.makeHudRow());
        const h2 = document.createElement('h2');
        h2.textContent = 'دور باطل شد';
        const box = document.createElement('div');
        box.className = 'void-box';
        box.setAttribute('data-testid', 'void-reason-box');
        box.textContent = this.ctx.voidMessageFa ?? 'دور به دلیل شرایط ردیابی باطل شد.';

        const btnRow = document.createElement('div');
        btnRow.className = 'btn-row';

        const replayBtn = document.createElement('button');
        replayBtn.type = 'button';
        replayBtn.className = 'btn btn-primary';
        replayBtn.textContent = FA.replayVoidBtn;
        replayBtn.addEventListener('click', () => this.dispatch({ type: 'NEXT_ROUND' }));

        const exitBtn = document.createElement('button');
        exitBtn.type = 'button';
        exitBtn.className = 'btn btn-secondary';
        exitBtn.textContent = FA.exitToMenuBtn;
        exitBtn.addEventListener('click', () => this.dispatch({ type: 'EXIT_TO_MENU' }));

        btnRow.append(replayBtn, exitBtn);
        card.append(h2, box, btnRow);
        break;
      }

      case 'PAUSED': {
        const h2 = document.createElement('h2');
        h2.textContent = FA.pausedTitle;
        const p = document.createElement('p');
        p.textContent = FA.pausedBody;
        const resumeBtn = document.createElement('button');
        resumeBtn.type = 'button';
        resumeBtn.className = 'btn btn-primary';
        resumeBtn.textContent = 'ادامه بازی';
        resumeBtn.addEventListener('click', () => this.dispatch({ type: 'TAB_VISIBLE' }));
        card.append(h2, p, resumeBtn);
        break;
      }

      case 'CAMERA_ERROR': {
        const h2 = document.createElement('h2');
        h2.textContent = FA.cameraErrorTitle;
        const box = document.createElement('div');
        box.className = 'void-box';
        box.textContent = this.ctx.cameraErrorMessageFa ?? FA.permissionDeniedHint;

        const btnRow = document.createElement('div');
        btnRow.className = 'btn-row';

        const recoverBtn = document.createElement('button');
        recoverBtn.type = 'button';
        recoverBtn.className = 'btn btn-primary';
        recoverBtn.textContent = FA.cameraRecoverBtn;
        recoverBtn.addEventListener('click', () => {
          void this.camera
            .start(this.videoEl)
            .then(() => this.dispatch({ type: 'CAMERA_RECOVERED' }))
            .catch(() => this.renderUi());
        });

        const permBtn = document.createElement('button');
        permBtn.type = 'button';
        permBtn.className = 'btn btn-secondary';
        permBtn.textContent = FA.cameraRetryPermissionBtn;
        permBtn.addEventListener('click', () => this.dispatch({ type: 'RETRY_PERMISSION' }));

        btnRow.append(recoverBtn, permBtn);
        card.append(h2, box, btnRow);
        break;
      }
    }

    this.panelContainer.append(card);
  }

  private describeComputerHidesReasonFa(): string {
    switch (this.ctx.lastComputerHidesOutcomeReason) {
      case 'GOL_DIRECT':
        return `گل مستقیم پیدا شد (+${toPersianDigits(this.ctx.ruleSet.scoring.correctGolNoWrongCalls)} امتیاز و تغییر نوبت)`;
      case 'GOL_AFTER_POOCH':
        return `گل پس از فراخوانی پوچ پیدا شد (+${toPersianDigits(this.ctx.ruleSet.scoring.correctGolAfterPoochCalls)} امتیاز و تغییر نوبت)`;
      case 'POOCH_HIT_GOL':
        return `فراخوانی «پوچ» روی دستِ دارای گل انجام شد (+${toPersianDigits(this.ctx.ruleSet.scoring.wrongGuessOrOutOfGuessesHiderPoints)} امتیاز رایانه)`;
      case 'OUT_OF_GUESSES':
        return `تعداد حدس‌ها یا زمان به پایان رسید (+${toPersianDigits(this.ctx.ruleSet.scoring.wrongGuessOrOutOfGuessesHiderPoints)} امتیاز رایانه)`;
      default:
        return `حدس اشتباه بود (+${toPersianDigits(this.ctx.ruleSet.scoring.wrongGuessOrOutOfGuessesHiderPoints)} امتیاز رایانه)`;
    }
  }

  private makeCommitFingerprintBox(): HTMLElement {
    const box = document.createElement('div');
    box.className = 'commit-box';
    box.setAttribute('data-testid', 'commit-fingerprint-box');

    const title = document.createElement('div');
    title.textContent = FA.commitLabel;

    const fp = document.createElement('div');
    fp.className = 'commit-fingerprint ltr-inline';
    fp.setAttribute('data-testid', 'commit-fingerprint-value');
    fp.textContent = this.ctx.commitPublic?.fingerprint ?? 'در حال تولید هش SHA-256…';

    box.append(title, fp);
    return box;
  }

  private makeCommitVerificationBox(): HTMLElement {
    const box = document.createElement('div');
    box.className = 'commit-box';
    box.setAttribute('data-testid', 'commit-verification-box');

    const topRow = document.createElement('div');
    topRow.style.display = 'flex';
    topRow.style.flexWrap = 'wrap';
    topRow.style.alignItems = 'center';
    topRow.style.justifyContent = 'space-between';
    topRow.style.gap = '8px';

    const title = document.createElement('strong');
    title.textContent = 'راستی‌آزمایی رمزنگاری (Commit–Reveal)';

    const badge = document.createElement('span');
    badge.setAttribute('data-testid', 'commit-verify-badge');
    if (this.lastCommitVerification) {
      badge.className = `commit-verified-badge${this.lastCommitVerification.valid ? '' : ' invalid'}`;
      badge.textContent = this.lastCommitVerification.statusFa;
    } else {
      badge.className = 'commit-verified-badge';
      badge.textContent = FA.commitVerifying;
    }

    topRow.append(title, badge);

    const details = document.createElement('div');
    details.className = 'check-hint ltr-inline';
    details.style.marginTop = '8px';
    details.style.wordBreak = 'break-all';
    if (this.ctx.commitPublic && this.ctx.commitSecret) {
      details.textContent = `choice=${this.ctx.commitSecret.choice} | salt=${this.ctx.commitSecret.saltHex} | SHA-256=${this.ctx.commitPublic.commitHex.slice(0, 24)}…`;
    }

    const howBtn = document.createElement('button');
    howBtn.type = 'button';
    howBtn.className = 'btn btn-secondary btn-compact';
    howBtn.style.marginTop = '10px';
    howBtn.textContent = FA.howItWorksBtn;
    howBtn.setAttribute('data-testid', 'how-it-works-btn');
    howBtn.addEventListener('click', () => {
      this.showHowItWorks = !this.showHowItWorks;
      this.renderPanel();
    });

    box.append(topRow, details, howBtn);

    if (this.showHowItWorks) {
      const howBox = document.createElement('div');
      howBox.className = 'how-it-works-box';
      howBox.setAttribute('data-testid', 'how-it-works-box');
      const hTitle = document.createElement('strong');
      hTitle.textContent = COMMIT_REVEAL_EXPLANATION_FA.title;
      const hSummary = document.createElement('p');
      hSummary.textContent = COMMIT_REVEAL_EXPLANATION_FA.summary;
      const ul = document.createElement('ul');
      for (const stepText of COMMIT_REVEAL_EXPLANATION_FA.steps) {
        const li = document.createElement('li');
        li.textContent = stepText;
        ul.append(li);
      }
      howBox.append(hTitle, hSummary, ul);
      box.append(howBox);
    }

    return box;
  }

  private makeHudRow(): HTMLElement {
    const row = document.createElement('div');
    row.className = 'hud-row';
    row.setAttribute('data-testid', 'game-hud');

    const makeStat = (label: string, val: string, testId?: string, isTimer = false) => {
      const box = document.createElement('div');
      box.className = 'hud-stat';
      const l = document.createElement('div');
      l.className = 'hud-stat-label';
      l.textContent = label;
      const v = document.createElement('div');
      v.className = 'hud-stat-value';
      if (isTimer) v.setAttribute('data-hud-timer', 'true');
      if (testId) v.setAttribute('data-testid', testId);
      v.textContent = val;
      box.append(l, v);
      return box;
    };

    const lvl = getLevelDefinition(this.ctx.levelId);
    row.append(
      makeStat('سطح', `${toPersianDigits(lvl.id)}. ${lvl.nameFa}`, 'hud-level'),
      makeStat(
        `شما (هدف ${toPersianDigits(this.ctx.ruleSet.targetScore)})`,
        toPersianDigits(this.ctx.humanScore),
        'hud-human-score'
      ),
      makeStat('زمان مرحله', formatSecondsFa(this.ctx.phaseRemainingMs), 'hud-timer', true),
      makeStat(
        `رایانه (هدف ${toPersianDigits(this.ctx.ruleSet.targetScore)})`,
        toPersianDigits(this.ctx.computerScore),
        'hud-computer-score'
      )
    );
    return row;
  }

  private makeToggleRow(
    label: string,
    checked: boolean,
    onChange: (val: boolean) => void
  ): HTMLElement {
    const row = document.createElement('label');
    row.className = 'toggle-row';
    const span = document.createElement('span');
    span.textContent = label;
    const input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = checked;
    input.addEventListener('change', () => onChange(input.checked));
    row.append(span, input);
    return row;
  }

  private getDebugMetricValues(): Array<[string, string]> {
    const belief = this.lastBelief ?? this.beliefFilter.getBelief();
    const behav = this.ensemble.predict(this.behaviorHistory);
    const ortStatus = this.temporalModel.getStatus();
    const lastMs = this.lastTemporalPred?.available
      ? this.lastTemporalPred.inferenceMs
      : ortStatus.lastInferenceMs;
    const onnxStr = ortStatus.ready
      ? `READY (${ortStatus.backend}, ${ortStatus.modelVersion ?? 'v1'})${lastMs !== null ? ` | ${lastMs.toFixed(2)} ms` : ''}${this.lastTemporalPred?.pLeft !== null && this.lastTemporalPred?.pLeft !== undefined ? ` | P(L)=${(this.lastTemporalPred.pLeft * 100).toFixed(1)}%` : ''}`
      : `FALLBACK (${ortStatus.fallbackReason ?? ortStatus.status})`;
    const fusionStr = this.lastGuessResult
      ? `${this.lastGuessResult.temporalUsed ? 'HMM+ONNX' : 'HMM-only'} -> P(L)=${(this.lastGuessResult.trackingPLeft * 100).toFixed(1)}%`
      : `wHMM=${ortStatus.fusionConfig.fusionWeights.hmmWeight.toFixed(2)}, wONNX=${ortStatus.fusionConfig.fusionWeights.temporalWeight.toFixed(2)}`;

    return [
      ['FSM State', `${this.ctx.state} (L${this.ctx.levelId})`],
      ['FPS / Infer', `${this.fps.toFixed(1)} fps / ${this.inferenceMs.toFixed(1)} ms`],
      [
        'Left Hand',
        this.lastHands?.left.detected
          ? `OK (curl:${this.lastFeatures?.left.totalCurl.toFixed(2) ?? '-'}, comp:${this.lastFeatures?.left.fistCompactness.toFixed(2) ?? '-'})`
          : `LOST (${this.lastHands?.left.lostFrames ?? 0}, occ:${Boolean(this.lastHands?.left.occluded)})`,
      ],
      [
        'Right Hand',
        this.lastHands?.right.detected
          ? `OK (curl:${this.lastFeatures?.right.totalCurl.toFixed(2) ?? '-'}, comp:${this.lastFeatures?.right.fistCompactness.toFixed(2) ?? '-'})`
          : `LOST (${this.lastHands?.right.lostFrames ?? 0}, occ:${Boolean(this.lastHands?.right.occluded)})`,
      ],
      [
        'Inter-Hand',
        this.lastFeatures
          ? `${this.lastFeatures.interHand.inContact ? 'CONTACT' : 'APART'} (d=${this.lastFeatures.interHand.minLandmarkDistanceNorm.toFixed(2)}, contacts=${this.ctx.hidingContactCount})`
          : 'N/A',
      ],
      [
        'HMM Belief',
        `P(L)=${(belief.pLeft * 100).toFixed(0)}% | P(R)=${(belief.pRight * 100).toFixed(0)}% | conf=${belief.confidence.toFixed(2)}`,
      ],
      ['ONNX Temporal', onnxStr],
      ['Fused Tracking', fusionStr],
      [
        'Behavior Prior',
        `P(L)=${(behav.pLeft * 100).toFixed(0)}% (${this.behaviorHistory.length} rounds) | Logs=${this.loggedRoundsCount} (${this.consentSettings.consentEnabled ? 'ON' : 'OFF'})`,
      ],
      [
        'Commit Hash',
        this.ctx.commitPublic ? this.ctx.commitPublic.fingerprint : 'N/A',
      ],
    ];
  }

  private updateDebugMetricsInPlace(): void {
    this.lastDebugRenderTs = performance.now();
    const metrics = this.getDebugMetricValues();
    for (const [label, value] of metrics) {
      const el = this.debugContainer.querySelector<HTMLElement>(
        `[data-debug-metric="${label}"]`
      );
      if (el) {
        el.textContent = value;
      }
    }

    if (this.beliefGraphCanvas) {
      renderBeliefGraphCanvas(this.beliefGraphCanvas, this.beliefFilter.getSnapshot().trace);
    }

    const tlList = this.debugContainer.querySelector<HTMLElement>(
      '[data-testid="debug-timeline-list"]'
    );
    if (tlList) {
      const snap = this.beliefFilter.getSnapshot();
      tlList.innerHTML = '';
      if (snap.timeline.length === 0) {
        const li = document.createElement('li');
        li.textContent = 'No tracking events recorded yet in current round.';
        tlList.append(li);
      } else {
        for (const ev of snap.timeline.slice(-6)) {
          const li = document.createElement('li');
          if (ev.type === 'CONTACT_START') {
            li.textContent = `[${ev.relativeSec.toFixed(2)}s] CONTACT_START (d=${ev.minDistanceNorm.toFixed(2)})`;
          } else if (ev.type === 'CONTACT_END') {
            li.textContent = `[${ev.relativeSec.toFixed(2)}s] CONTACT_END (${ev.durationMs.toFixed(0)}ms)`;
          } else if (ev.type === 'PARTIAL_OPENING') {
            li.textContent = `[${ev.relativeSec.toFixed(2)}s] PARTIAL_OPENING ${ev.side} (str=${ev.strength.toFixed(2)}, contact=${ev.duringContact})`;
          } else if (ev.type === 'GOL_SIGHTING') {
            li.textContent = `[${ev.relativeSec.toFixed(2)}s] GOL_SIGHTING ${ev.side} (score=${ev.score.toFixed(2)})`;
          } else if (ev.type === 'OCCLUSION_START') {
            li.textContent = `[${ev.relativeSec.toFixed(2)}s] OCCLUSION_START (${ev.side})`;
          } else {
            li.textContent = `[${ev.relativeSec.toFixed(2)}s] OCCLUSION_END (${ev.durationMs.toFixed(0)}ms)`;
          }
          tlList.append(li);
        }
      }
    }
  }

  private renderDebugPanel(): void {
    this.debugContainer.classList.toggle('hidden', !this.showDebug);
    if (!this.showDebug) return;
    this.lastDebugRenderTs = performance.now();

    this.debugContainer.innerHTML = '';

    const title = document.createElement('h3');
    title.textContent = FA.debugTitle;

    const grid = document.createElement('div');
    grid.className = 'debug-grid';

    for (const [label, value] of this.getDebugMetricValues()) {
      const box = document.createElement('div');
      box.className = 'debug-metric';
      const s = document.createElement('strong');
      s.textContent = label;
      const v = document.createElement('span');
      v.className = 'ltr-inline';
      v.setAttribute('data-debug-metric', label);
      v.textContent = value;
      box.append(s, v);
      grid.append(box);
    }

    const beliefSection = document.createElement('div');
    beliefSection.className = 'debug-belief-wrap';
    beliefSection.setAttribute('data-testid', 'debug-belief-panel');

    const chartHeader = document.createElement('div');
    chartHeader.className = 'debug-belief-header';
    chartHeader.innerHTML =
      '<strong>Live HMM Belief Trace</strong><span class="ltr-inline">Emerald: P(LEFT) | Sky: P(RIGHT) | Amber: Contact</span>';

    const chartCanvas = document.createElement('canvas');
    chartCanvas.className = 'debug-belief-canvas';
    chartCanvas.dir = 'ltr';
    chartCanvas.width = 560;
    chartCanvas.height = 120;
    chartCanvas.setAttribute('data-testid', 'debug-belief-chart');
    this.beliefGraphCanvas = chartCanvas;
    renderBeliefGraphCanvas(chartCanvas, this.beliefFilter.getSnapshot().trace);

    const tlList = document.createElement('ul');
    tlList.className = 'debug-timeline-list ltr-inline';
    tlList.setAttribute('data-testid', 'debug-timeline-list');

    beliefSection.append(chartHeader, chartCanvas, tlList);

    const btnRow = document.createElement('div');
    btnRow.className = 'btn-row';

    const recBtn = document.createElement('button');
    recBtn.type = 'button';
    recBtn.className = 'btn btn-secondary btn-compact';
    recBtn.textContent = this.recorder.isRecording ? FA.debugStopRec : FA.debugStartRec;
    recBtn.addEventListener('click', () => {
      if (this.recorder.isRecording) {
        const json = this.recorder.exportJson(true);
        const blob = new Blob([json], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = `gol-ya-pooch-session-${Date.now()}.json`;
        a.click();
        URL.revokeObjectURL(url);
      } else {
        this.recorder.start({
          source: 'camera',
          ...(this.golColorModel ? { golColorModel: this.golColorModel } : {}),
        });
      }
      this.renderDebugPanel();
    });

    const sampleBtn = document.createElement('button');
    sampleBtn.type = 'button';
    sampleBtn.className = 'btn btn-secondary btn-compact';
    sampleBtn.textContent = FA.debugReplaySample;
    sampleBtn.setAttribute('data-testid', 'debug-replay-sample-btn');
    sampleBtn.addEventListener('click', () =>
      this.startSessionReplay(createBuiltInSampleSession())
    );

    const transferBtn = document.createElement('button');
    transferBtn.type = 'button';
    transferBtn.className = 'btn btn-secondary btn-compact';
    transferBtn.textContent = FA.debugReplayTransfer;
    transferBtn.setAttribute('data-testid', 'debug-replay-transfer-btn');
    transferBtn.addEventListener('click', () =>
      this.startSessionReplay(createPhase2TransferSampleSession())
    );

    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'application/json,.json';
    fileInput.className = 'hidden';
    fileInput.addEventListener('change', () => {
      const file = fileInput.files?.[0];
      if (!file) return;
      void file.text().then((text) => {
        const session = parseRecordedSession(text);
        this.startSessionReplay(session);
      });
    });

    const loadBtn = document.createElement('button');
    loadBtn.type = 'button';
    loadBtn.className = 'btn btn-secondary btn-compact';
    loadBtn.textContent = FA.debugLoadJson;
    loadBtn.addEventListener('click', () => fileInput.click());

    const benchOnnxBtn = document.createElement('button');
    benchOnnxBtn.type = 'button';
    benchOnnxBtn.className = 'btn btn-secondary btn-compact';
    benchOnnxBtn.textContent = 'تست استنتاج ONNX';
    benchOnnxBtn.setAttribute('data-testid', 'debug-run-onnx-btn');
    benchOnnxBtn.addEventListener('click', () => {
      void this.runDebugOnnxBenchmark();
    });

    btnRow.append(recBtn, sampleBtn, transferBtn, loadBtn, benchOnnxBtn, fileInput);
    this.debugContainer.append(title, grid, beliefSection, btnRow);
    this.updateDebugMetricsInPlace();

    if (this.replayStatusText) {
      const statusP = document.createElement('p');
      statusP.className = 'check-hint';
      statusP.setAttribute('data-testid', 'replay-status');
      statusP.textContent = this.replayStatusText;
      this.debugContainer.append(statusP);
    }
  }
}
