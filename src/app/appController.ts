import { CameraManager, type CameraDeviceSummary } from '../camera/cameraManager';
import {
  createInitialGameContext,
  transitionGameFsm,
  type GameContext,
  type GameEvent,
} from '../game/fsm';
import { renderVisionOverlay } from '../render/canvasOverlay';
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
import { createBuiltInSampleSession } from '../vision/sampleSession';
import type {
  GolColorModel,
  PinchConfirmationState,
  PinchObservation,
  QualityReport,
  RecordedSession,
  RevealVerificationResult,
  VisionFrameHands,
} from '../vision/types';
import { FA, formatSecondsFa, sideLabelFa, toPersianDigits } from './i18n';

const CALIBRATION_CIRCLE = { x: 0.5, y: 0.5, radiusNorm: 0.085 };

export class GolYaPoochApp {
  private readonly root: HTMLElement;
  private ctx: GameContext = createInitialGameContext();
  private readonly camera = new CameraManager();
  private readonly tracker = new MediaPipeHandTracker();
  private readonly stabilityTracker = new CalibrationStabilityTracker(1000, 0.7);
  private readonly pinchTracker = new PinchConfirmationTracker(15);
  private readonly recorder = new SessionRecorder();

  private videoEl!: HTMLVideoElement;
  private overlayCanvas!: HTMLCanvasElement;
  private overlayCtx!: CanvasRenderingContext2D;
  private sampleCanvas!: HTMLCanvasElement;
  private sampleCtx!: CanvasRenderingContext2D;

  private stageContainer!: HTMLElement;
  private stageBannerEl!: HTMLElement;
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
  private lastPinchObs: PinchObservation[] = [];
  private lastPinchState: PinchConfirmationState | null = null;
  private lastRevealResult: RevealVerificationResult | null = null;
  private stabilityPassed = false;
  private stabilityProgress = 0;

  private fps = 0;
  private inferenceMs = 0;
  private lastFrameTs = 0;
  private rafId: number | null = null;
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
  }

  private buildShell(): void {
    this.root.innerHTML = '';

    const shell = document.createElement('div');
    shell.className = 'app-shell';

    // Top bar
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

    // Main grid (camera stage + active screen panel)
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

    this.panelContainer = document.createElement('section');
    this.panelContainer.setAttribute('data-testid', 'screen-panel');

    mainGrid.append(this.stageContainer, this.panelContainer);

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
  }

  private dispatch(event: GameEvent): void {
    const prevState = this.ctx.state;
    this.ctx = transitionGameFsm(this.ctx, event);

    if (prevState !== this.ctx.state) {
      if (this.ctx.state === 'SHOW_GOL') {
        this.pinchTracker.reset();
        this.lastPinchState = null;
        this.lastPinchObs = [];
        this.lastRevealResult = null;
      } else if (this.ctx.state === 'REVEAL_VERIFY') {
        this.lastRevealResult = null;
      } else if (this.ctx.state === 'CALIBRATION') {
        this.stabilityTracker.reset();
        this.stabilityPassed = false;
        this.stabilityProgress = 0;
      }
    }

    this.renderUi();
  }

  private shouldShowCameraStage(): boolean {
    return (
      this.ctx.state !== 'SPLASH' &&
      this.ctx.state !== 'PERMISSION' &&
      this.ctx.state !== 'CAMERA_ERROR'
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

      // Performance budget monitor (< 24 FPS warning + graceful resolution degrade)
      if (this.fps < 22 && this.camera.isStreaming && !this.lowFpsDegraded) {
        this.lowFpsCounter += 1;
        if (this.lowFpsCounter > 45) {
          this.lowFpsDegraded = true;
          void this.camera.start(this.videoEl, { preferLowRes: true }).catch(() => {});
        }
      } else {
        this.lowFpsCounter = 0;
      }

      if (
        this.ctx.state === 'PAUSED' ||
        this.ctx.state === 'CAMERA_ERROR' ||
        !this.camera.isStreaming ||
        this.videoEl.readyState < 2
      ) {
        return;
      }

      this.processCameraFrame(now, dtMs);
    };

    this.rafId = requestAnimationFrame(step);
  }

  private processCameraFrame(now: number, dtMs: number): void {
    const t0 = performance.now();
    const vw = this.videoEl.videoWidth || 640;
    const vh = this.videoEl.videoHeight || 480;
    if (this.overlayCanvas.width !== vw || this.overlayCanvas.height !== vh) {
      this.overlayCanvas.width = vw;
      this.overlayCanvas.height = vh;
    }

    // Draw into 320x240 sampling canvas for fast typed-array ImageData checks
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
      } else if (this.ctx.state === 'HIDING' && this.golColorModel) {
        // Scan both hands for brief gol sightings (recorded numerically for debug/Phase 2)
        scanHandRoisForGol(this.lastHands.left, frameImageData.data, sw, sh, this.golColorModel);
        scanHandRoisForGol(this.lastHands.right, frameImageData.data, sw, sh, this.golColorModel);
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
      this.dispatch({
        type: 'TICK',
        dtMs,
        maxLostDurationMs: this.lastHands?.maxLostDurationMs ?? 0,
      });
    } else {
      // Refresh calibration status & debug view smoothly without spamming FSM
      this.updateLiveOverlay();
      if (this.ctx.state === 'CALIBRATION') {
        this.renderCalibrationChecklistOnly();
      }
      if (this.showDebug) {
        this.renderDebugPanel();
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

    if (session.golColorModel) {
      this.golColorModel = session.golColorModel;
    }

    const summary = replayRecordedSession(session, { requiredPinchFrames: 15 });
    this.replayStatusText = `بازپخش ${toPersianDigits(summary.totalFrames)} فریم آغاز شد…`;

    // Drive the FSM cleanly into SHOW_GOL so replay exercises the live state machine
    this.ctx = {
      ...createInitialGameContext(),
      state: 'SHOW_GOL',
      calibrated: true,
      phaseRemainingMs: 12_000,
    };
    this.renderUi();

    let idx = 0;
    this.replayTimerId = window.setInterval(() => {
      const frame = summary.frames[idx];
      if (!frame) {
        this.stopReplay();
        this.replayStatusText = `بازپخش کامل شد: تأیید گل=${summary.pinchConfirmed ? 'بله' : 'خیر'} (${summary.confirmedSide ?? '-'}) | نتیجه بازگشایی=${summary.finalRevealOutcome ?? '-'}`;
        this.renderUi();
        return;
      }

      this.lastHands = frame.hands;
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
      } else if (this.ctx.state === 'HIDING' && idx >= summary.frames.length - 2) {
        this.dispatch({ type: 'FINISH_HIDING' });
        this.dispatch({ type: 'PROCEED_TO_REVEAL' });
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
        });
      }

      if (this.overlayCanvas.width !== 640) {
        this.overlayCanvas.width = 640;
        this.overlayCanvas.height = 480;
      }
      this.updateLiveOverlay();
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
      fps: this.fps,
      inferenceMs: this.inferenceMs,
    });

    // Update stage banner
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

  private renderCalibrationChecklistOnly(): void {
    const list = this.panelContainer.querySelector('[data-cal-checklist]');
    if (!list) return;
    // Re-render panel cleanly if calibration checklist is active
    this.renderPanel();
  }

  private renderUi(): void {
    const showStage = this.shouldShowCameraStage();
    this.stageContainer.classList.toggle('hidden', !showStage);
    const mainGrid = this.panelContainer.parentElement;
    mainGrid?.classList.toggle('with-camera', showStage);

    this.videoEl.classList.toggle('mirrored', this.mirrored);
    document.body.classList.toggle('reduced-motion', this.reducedMotion);

    this.topbarStatusEl.textContent =
      this.ctx.state === 'MENU' ||
      this.ctx.state === 'SHOW_GOL' ||
      this.ctx.state === 'HIDING' ||
      this.ctx.state === 'COMPUTER_GUESS' ||
      this.ctx.state === 'REVEAL_VERIFY'
        ? `دور ${toPersianDigits(this.ctx.roundNumber)} | شما ${toPersianDigits(this.ctx.humanScore)} - ${toPersianDigits(this.ctx.computerScore)} رایانه`
        : 'نسخه نمایشی فاز ۱';

    this.renderPanel();
    this.renderDebugPanel();
    if (showStage) {
      this.updateLiveOverlay();
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

        const replayBtn = document.createElement('button');
        replayBtn.type = 'button';
        replayBtn.className = 'btn btn-secondary';
        replayBtn.textContent = FA.debugReplaySample;
        replayBtn.setAttribute('data-testid', 'replay-sample-btn');
        replayBtn.addEventListener('click', () =>
          this.startSessionReplay(createBuiltInSampleSession())
        );

        card.append(badge, h2, p, startBtn, replayBtn);
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

        const replayBtn = document.createElement('button');
        replayBtn.type = 'button';
        replayBtn.className = 'btn btn-secondary';
        replayBtn.textContent = FA.debugReplaySample;
        replayBtn.addEventListener('click', () =>
          this.startSessionReplay(createBuiltInSampleSession())
        );

        btnRow.append(grantBtn, replayBtn);
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

        const addCheck = (label: string, ok: boolean, hint?: string | null) => {
          const li = document.createElement('li');
          li.className = `check-item ${ok ? 'ok' : 'fail'}`;
          const left = document.createElement('div');
          const title = document.createElement('div');
          title.textContent = label;
          left.append(title);
          if (hint) {
            const p = document.createElement('p');
            p.className = 'check-hint';
            p.textContent = hint;
            left.append(p);
          }
          const badge = document.createElement('span');
          badge.className = `check-status ${ok ? 'ok' : 'pending'}`;
          badge.textContent = ok ? 'تأیید ✓' : 'در انتظار…';
          li.append(left, badge);
          list.append(li);
        };

        addCheck(FA.calCheckLighting, Boolean(q?.lightingOk), q?.lightingHintFa);
        addCheck(
          FA.calCheckHands,
          this.stabilityPassed,
          this.stabilityPassed
            ? 'هر دو دست با اطمینان کافی تثبیت شدند.'
            : `پیشرفت پایداری: ${toPersianDigits(Math.round(this.stabilityProgress * 100))}٪`
        );
        addCheck(FA.calCheckDistance, Boolean(q?.distanceOk), q?.distanceHintFa);
        addCheck(
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
        const h2 = document.createElement('h2');
        h2.textContent = FA.menuTitle;

        const note = document.createElement('p');
        note.textContent = FA.menuPhaseNote;

        const btnRow = document.createElement('div');
        btnRow.className = 'btn-row';

        const playBtn = document.createElement('button');
        playBtn.type = 'button';
        playBtn.className = 'btn btn-primary';
        playBtn.textContent = FA.menuPlayBtn;
        playBtn.setAttribute('data-testid', 'menu-play-btn');
        playBtn.addEventListener('click', () => this.dispatch({ type: 'START_GAME' }));

        const calBtn = document.createElement('button');
        calBtn.type = 'button';
        calBtn.className = 'btn btn-secondary';
        calBtn.textContent = FA.menuRecalibrateBtn;
        calBtn.addEventListener('click', () => this.dispatch({ type: 'OPEN_CALIBRATION' }));

        btnRow.append(playBtn, calBtn);

        const settingsHeading = document.createElement('h3');
        settingsHeading.style.marginTop = '18px';
        settingsHeading.textContent = FA.menuSettingsHeading;

        const settingsBox = document.createElement('div');
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
          })
        );

        card.append(h2, note, btnRow, settingsHeading, settingsBox);
        break;
      }

      case 'SHOW_GOL':
      case 'HIDING':
      case 'COMPUTER_GUESS':
      case 'REVEAL_VERIFY': {
        const hud = this.makeHudRow();
        card.append(hud);

        if (this.ctx.state === 'SHOW_GOL') {
          const h2 = document.createElement('h2');
          h2.textContent = FA.showGolPrompt;
          const p = document.createElement('p');
          const framesDone = this.lastPinchState?.consecutiveFrames ?? 0;
          const framesReq = this.lastPinchState?.requiredFrames ?? 15;
          p.textContent = `تثبیت پینچ: ${toPersianDigits(framesDone)} از ${toPersianDigits(framesReq)} فریم پیاپی`;
          card.append(h2, p);
        } else if (this.ctx.state === 'HIDING') {
          const h2 = document.createElement('h2');
          h2.textContent = FA.hidingIndicator;
          const p = document.createElement('p');
          p.textContent = FA.hidingInstruction;
          const doneBtn = document.createElement('button');
          doneBtn.type = 'button';
          doneBtn.className = 'btn btn-primary';
          doneBtn.textContent = FA.finishHidingBtn;
          doneBtn.addEventListener('click', () => this.dispatch({ type: 'FINISH_HIDING' }));
          card.append(h2, p, doneBtn);
        } else if (this.ctx.state === 'COMPUTER_GUESS') {
          const h2 = document.createElement('h2');
          h2.textContent = FA.guessPhaseTitle;

          const guessBox = document.createElement('div');
          guessBox.className = 'guess-card';
          const lbl = document.createElement('div');
          lbl.textContent = 'حدس رایانه:';
          const handEl = document.createElement('div');
          handEl.className = 'guess-hand-big';
          handEl.textContent = this.ctx.computerGuess ? sideLabelFa(this.ctx.computerGuess) : '-';
          const note = document.createElement('div');
          note.className = 'check-hint';
          note.textContent = FA.guessPlaceholderNote;
          guessBox.append(lbl, handEl, note);

          const nextBtn = document.createElement('button');
          nextBtn.type = 'button';
          nextBtn.className = 'btn btn-primary';
          nextBtn.textContent = FA.proceedToRevealBtn;
          nextBtn.addEventListener('click', () => this.dispatch({ type: 'PROCEED_TO_REVEAL' }));

          card.append(h2, guessBox, nextBtn);
        } else {
          const h2 = document.createElement('h2');
          h2.textContent = FA.revealPrompt;
          const p = document.createElement('p');
          p.textContent = FA.revealWaitingHands;
          card.append(h2, p);
        }

        const exitRow = document.createElement('div');
        exitRow.className = 'btn-row';
        const exitBtn = document.createElement('button');
        exitBtn.type = 'button';
        exitBtn.className = 'btn btn-secondary';
        exitBtn.textContent = FA.exitToMenuBtn;
        exitBtn.addEventListener('click', () => {
          this.stopReplay();
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

        const detail = document.createElement('p');
        detail.textContent = `گل در ${this.ctx.revealedHand ? sideLabelFa(this.ctx.revealedHand) : '-'} بود | حدس رایانه: ${this.ctx.computerGuess ? sideLabelFa(this.ctx.computerGuess) : '-'}`;

        const btnRow = document.createElement('div');
        btnRow.className = 'btn-row';

        const nextBtn = document.createElement('button');
        nextBtn.type = 'button';
        nextBtn.className = 'btn btn-primary';
        nextBtn.textContent = FA.nextRoundBtn;
        nextBtn.addEventListener('click', () => this.dispatch({ type: 'NEXT_ROUND' }));

        const resetBtn = document.createElement('button');
        resetBtn.type = 'button';
        resetBtn.className = 'btn btn-secondary';
        resetBtn.textContent = FA.resetMatchBtn;
        resetBtn.addEventListener('click', () => this.dispatch({ type: 'RESET_MATCH' }));

        const exitBtn = document.createElement('button');
        exitBtn.type = 'button';
        exitBtn.className = 'btn btn-secondary';
        exitBtn.textContent = FA.exitToMenuBtn;
        exitBtn.addEventListener('click', () => this.dispatch({ type: 'EXIT_TO_MENU' }));

        btnRow.append(nextBtn, resetBtn, exitBtn);
        card.append(h2, detail, btnRow);
        break;
      }

      case 'ROUND_VOID': {
        card.append(this.makeHudRow());
        const h2 = document.createElement('h2');
        h2.textContent = 'دور باطل شد';
        const box = document.createElement('div');
        box.className = 'void-box';
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

  private makeHudRow(): HTMLElement {
    const row = document.createElement('div');
    row.className = 'hud-row';

    const makeStat = (label: string, val: string) => {
      const box = document.createElement('div');
      box.className = 'hud-stat';
      const l = document.createElement('div');
      l.className = 'hud-stat-label';
      l.textContent = label;
      const v = document.createElement('div');
      v.className = 'hud-stat-value';
      v.textContent = val;
      box.append(l, v);
      return box;
    };

    row.append(
      makeStat('امتیاز شما', toPersianDigits(this.ctx.humanScore)),
      makeStat('زمان مرحله', formatSecondsFa(this.ctx.phaseRemainingMs)),
      makeStat('امتیاز رایانه', toPersianDigits(this.ctx.computerScore))
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

  private renderDebugPanel(): void {
    this.debugContainer.classList.toggle('hidden', !this.showDebug);
    if (!this.showDebug) return;

    this.debugContainer.innerHTML = '';

    const title = document.createElement('h3');
    title.textContent = FA.debugTitle;

    const grid = document.createElement('div');
    grid.className = 'debug-grid';

    const addMetric = (label: string, value: string) => {
      const box = document.createElement('div');
      box.className = 'debug-metric';
      const s = document.createElement('strong');
      s.textContent = label;
      const v = document.createElement('span');
      v.className = 'ltr-inline';
      v.textContent = value;
      box.append(s, v);
      grid.append(box);
    };

    addMetric('FSM State', this.ctx.state);
    addMetric('FPS / Infer', `${this.fps.toFixed(1)} fps / ${this.inferenceMs.toFixed(1)} ms`);
    addMetric(
      'Left Hand',
      this.lastHands?.left.detected
        ? `OK (lost:${this.lastHands.left.lostFrames}, conf:${this.lastHands.left.confidence.toFixed(2)})`
        : `LOST (${this.lastHands?.left.lostFrames ?? 0}, occ:${Boolean(this.lastHands?.left.occluded)})`
    );
    addMetric(
      'Right Hand',
      this.lastHands?.right.detected
        ? `OK (lost:${this.lastHands.right.lostFrames}, conf:${this.lastHands.right.confidence.toFixed(2)})`
        : `LOST (${this.lastHands?.right.lostFrames ?? 0}, occ:${Boolean(this.lastHands?.right.occluded)})`
    );
    addMetric(
      'Lighting / Blur',
      this.lastQuality
        ? `L=${this.lastQuality.meanLuminance.toFixed(0)} S=${this.lastQuality.contrastStd.toFixed(0)} B=${this.lastQuality.blurScore.toFixed(0)}`
        : 'N/A'
    );
    addMetric(
      'Recorder',
      `${this.recorder.isRecording ? 'REC' : 'IDLE'} (${this.recorder.frameCount} frames)`
    );

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

    btnRow.append(recBtn, sampleBtn, loadBtn, fileInput);
    this.debugContainer.append(title, grid, btnRow);

    if (this.replayStatusText) {
      const statusP = document.createElement('p');
      statusP.className = 'check-hint';
      statusP.setAttribute('data-testid', 'replay-status');
      statusP.textContent = this.replayStatusText;
      this.debugContainer.append(statusP);
    }
  }
}
