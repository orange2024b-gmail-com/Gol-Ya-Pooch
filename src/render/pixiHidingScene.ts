import { Application, Container, Graphics, Text, TextStyle } from 'pixi.js';
import type { HidingChoreography } from '../ai/hider';

export interface PixiSceneState {
  phase: 'COMPUTER_HIDES' | 'HUMAN_GUESS' | 'REVEALED';
  handsCount: number;
  selectedHandIndex: number | null;
  openedEmptyHands: readonly number[];
  revealedGolHandIndex: number | null;
  allowPoochCalls: boolean;
  reducedMotion: boolean;
}

interface HandSlotTarget {
  x: number;
  y: number;
  scale: number;
}

interface HandNode {
  index: number;
  container: Container;
  bgGlow: Graphics;
  closedArt: Graphics;
  openArt: Graphics;
  golArt: Graphics;
  labelText: Text;
  statusText: Text;
  baseX: number;
  baseY: number;
  isOpen: boolean;
  hasGol: boolean;
}

function easeInOutCubic(t: number): number {
  return t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;
}

function easeOutBack(t: number): number {
  const c1 = 1.70158;
  const c3 = c1 + 1;
  return 1 + c3 * Math.pow(t - 1, 3) + c1 * Math.pow(t - 1, 2);
}

const PERSIAN_DIGITS = ['۰', '۱', '۲', '۳', '۴', '۵', '۶', '۷', '۸', '۹'] as const;
function toFaNum(n: number): string {
  return String(n).replace(/\d/g, (d) => PERSIAN_DIGITS[Number(d)] ?? d);
}

function handLabelFa(index: number, handsCount: number): string {
  if (handsCount === 2) {
    return index === 0 ? 'دست چپ' : 'دست راست';
  }
  return `دست ${toFaNum(index + 1)}`;
}

export class PixiHidingScene {
  private readonly hostEl: HTMLElement;
  private app: Application | null = null;
  private stageRoot = new Container();
  private backdrop = new Graphics();
  private handsLayer = new Container();
  private handNodes: HandNode[] = [];
  private onSelectHand: ((handIndex: number) => void) | null = null;

  private width = 560;
  private height = 360;
  private destroyed = false;
  private initPromise: Promise<void> | null = null;

  private currentState: PixiSceneState = {
    phase: 'COMPUTER_HIDES',
    handsCount: 2,
    selectedHandIndex: null,
    openedEmptyHands: [],
    revealedGolHandIndex: null,
    allowPoochCalls: false,
    reducedMotion: false,
  };

  private choreography: HidingChoreography | null = null;
  private choreoStepIndex = 0;
  private choreoStepElapsedMs = 0;
  private choreoActive = false;
  private choreoOnDone: (() => void) | null = null;
  private idleClockMs = 0;
  private resizeObserver: ResizeObserver | null = null;

  constructor(hostEl: HTMLElement) {
    this.hostEl = hostEl;
  }

  setSelectHandler(handler: (handIndex: number) => void): void {
    this.onSelectHand = handler;
  }

  async init(): Promise<void> {
    if (this.app) return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      const rect = this.hostEl.getBoundingClientRect();
      this.width = Math.max(300, Math.round(rect.width || 560));
      this.height = Math.max(260, Math.round(rect.height || 360));

      const app = new Application();
      try {
        await app.init({
          width: this.width,
          height: this.height,
          background: 0x0b1619,
          antialias: true,
          resolution: Math.min(2, window.devicePixelRatio || 1),
          autoDensity: true,
        });
      } catch {
        // Fallback if WebGL is unavailable in a constrained environment
        await app.init({
          width: this.width,
          height: this.height,
          background: 0x0b1619,
          antialias: true,
          preference: 'webgl',
        });
      }

      if (this.destroyed) {
        app.destroy(true, { children: true });
        return;
      }

      this.app = app;
      this.hostEl.innerHTML = '';
      const canvas = app.canvas;
      canvas.style.width = '100%';
      canvas.style.height = '100%';
      canvas.style.display = 'block';
      canvas.setAttribute('data-testid', 'pixi-hiding-canvas');
      this.hostEl.appendChild(canvas);

      this.stageRoot.addChild(this.backdrop, this.handsLayer);
      app.stage.addChild(this.stageRoot);

      this.drawBackdrop();
      this.rebuildHands(this.currentState.handsCount);
      this.applyStateVisuals();

      app.ticker.add((ticker) => {
        this.tick(ticker.deltaMS);
      });

      if (typeof ResizeObserver !== 'undefined') {
        this.resizeObserver = new ResizeObserver((entries) => {
          const entry = entries[0];
          if (!entry || !this.app) return;
          const nw = Math.max(280, Math.round(entry.contentRect.width));
          const nh = Math.max(250, Math.round(entry.contentRect.height));
          if (Math.abs(nw - this.width) > 2 || Math.abs(nh - this.height) > 2) {
            this.width = nw;
            this.height = nh;
            this.app.renderer.resize(nw, nh);
            this.drawBackdrop();
            this.layoutHands();
          }
        });
        this.resizeObserver.observe(this.hostEl);
      }
    })();

    return this.initPromise;
  }

  updateState(next: PixiSceneState): void {
    const countChanged = next.handsCount !== this.currentState.handsCount;
    this.currentState = next;
    if (!this.app) return;
    if (countChanged) {
      this.rebuildHands(next.handsCount);
    }
    this.applyStateVisuals();
  }

  startChoreography(choreo: HidingChoreography, onDone?: () => void): void {
    this.choreography = choreo;
    this.choreoStepIndex = 0;
    this.choreoStepElapsedMs = 0;
    this.choreoActive = true;
    this.choreoOnDone = onDone ?? null;
    if (this.app && choreo.handsCount !== this.handNodes.length) {
      this.rebuildHands(choreo.handsCount);
    }
    this.layoutHands();
    this.applyStateVisuals();
  }

  stopChoreography(): void {
    this.choreoActive = false;
    this.choreography = null;
    this.choreoOnDone = null;
    this.layoutHands();
  }

  destroy(): void {
    this.destroyed = true;
    this.resizeObserver?.disconnect();
    this.resizeObserver = null;
    if (this.app) {
      this.app.destroy(true, { children: true });
      this.app = null;
    }
    this.handNodes = [];
  }

  private drawBackdrop(): void {
    const g = this.backdrop;
    const w = this.width;
    const h = this.height;
    g.clear();

    // Rich dark emerald/obsidian tabletop background with subtle gold geometric border
    g.roundRect(0, 0, w, h, 16);
    g.fill({ color: 0x0b1619 });

    g.circle(w * 0.5, h * 0.5, Math.min(w, h) * 0.44);
    g.fill({ color: 0x11252a, alpha: 0.72 });
    g.stroke({ color: 0xf5b942, width: 1.5, alpha: 0.28 });

    g.roundRect(10, 10, w - 20, h - 20, 14);
    g.stroke({ color: 0xf5b942, width: 1, alpha: 0.2 });
  }

  private computeSlotTargets(count: number): HandSlotTarget[] {
    const w = this.width;
    const h = this.height;
    const targets: HandSlotTarget[] = [];

    if (count === 2) {
      const scale = Math.min(1.08, Math.max(0.75, Math.min(w / 460, h / 320)));
      targets.push({ x: w * 0.31, y: h * 0.47, scale });
      targets.push({ x: w * 0.69, y: h * 0.47, scale });
      return targets;
    }

    if (count === 4) {
      const compact = w < 520;
      const scale = compact
        ? Math.min(0.86, Math.max(0.64, w / 460))
        : Math.min(0.95, Math.max(0.72, w / 680));
      if (compact) {
        const xs = [w * 0.3, w * 0.7];
        const ys = [h * 0.31, h * 0.68];
        for (const y of ys) {
          for (const x of xs) targets.push({ x, y, scale });
        }
      } else {
        for (let i = 0; i < 4; i++) {
          const frac = (i + 0.5) / 4;
          const arcY = h * 0.48 + (i === 0 || i === 3 ? 16 : -10);
          targets.push({ x: w * (0.08 + frac * 0.84), y: arcY, scale });
        }
      }
      return targets;
    }

    // 6 hands: clean 3x2 layout on both portrait & desktop
    const scale = Math.min(0.82, Math.max(0.56, Math.min(w / 560, h / 380)));
    const xs = [w * 0.2, w * 0.5, w * 0.8];
    const ys = [h * 0.31, h * 0.69];
    for (const y of ys) {
      for (const x of xs) {
        targets.push({ x, y, scale });
      }
    }
    return targets;
  }

  private drawClosedFistGraphics(g: Graphics): void {
    g.clear();
    // Drop shadow
    g.ellipse(0, 42, 40, 12);
    g.fill({ color: 0x000000, alpha: 0.38 });

    // Sleeve / Persian cuff
    g.roundRect(-24, 24, 48, 28, 10);
    g.fill({ color: 0x115e59 });
    g.stroke({ color: 0xf5b942, width: 2.2 });

    // Decorative cuff band
    g.roundRect(-22, 24, 44, 7, 4);
    g.fill({ color: 0xf5b942, alpha: 0.88 });

    // Fist base palm
    g.roundRect(-34, -28, 68, 56, 22);
    g.fill({ color: 0xe6b48c });
    g.stroke({ color: 0x9b6542, width: 2.2 });

    // Curled fingers (4 stylized knuckles)
    const fingerXs = [-25, -9, 7, 23];
    for (let i = 0; i < fingerXs.length; i++) {
      const fx = fingerXs[i]!;
      const fh = i === 1 || i === 2 ? 28 : 24;
      g.roundRect(fx - 7.5, -34, 15, fh, 7.5);
      g.fill({ color: 0xf0c5a1 });
      g.stroke({ color: 0x9b6542, width: 1.8 });
    }

    // Folded thumb across fist front
    g.roundRect(-30, -4, 42, 16, 8);
    g.fill({ color: 0xdfb088 });
    g.stroke({ color: 0x9b6542, width: 1.8 });

    // Subtle palm highlight
    g.ellipse(8, 14, 14, 7);
    g.fill({ color: 0xf9dec7, alpha: 0.35 });
  }

  private drawOpenPalmGraphics(g: Graphics): void {
    g.clear();
    g.ellipse(0, 46, 44, 12);
    g.fill({ color: 0x000000, alpha: 0.35 });

    // Sleeve / Persian cuff
    g.roundRect(-22, 26, 44, 28, 10);
    g.fill({ color: 0x115e59 });
    g.stroke({ color: 0xf5b942, width: 2.2 });

    // Splayed 5 fingers
    const fingers: Array<{ x: number; y: number; w: number; h: number; rot: number }> = [
      { x: -34, y: -6, w: 13, h: 30, rot: -0.58 }, // thumb
      { x: -22, y: -38, w: 13, h: 36, rot: -0.14 },
      { x: -6, y: -44, w: 13.5, h: 41, rot: 0 },
      { x: 10, y: -40, w: 13, h: 37, rot: 0.14 },
      { x: 25, y: -30, w: 12, h: 30, rot: 0.3 },
    ];
    for (const f of fingers) {
      const cx = f.x;
      const cy = f.y;
      g.ellipse(cx, cy, f.w * 0.55, f.h * 0.55);
      g.fill({ color: 0xf0c5a1 });
      g.stroke({ color: 0x9b6542, width: 1.8 });
    }

    // Open palm center
    g.ellipse(0, 4, 32, 29);
    g.fill({ color: 0xe6b48c });
    g.stroke({ color: 0x9b6542, width: 2.2 });

    // Subtle palm crease lines
    g.moveTo(-18, -2);
    g.quadraticCurveTo(0, -10, 18, 0);
    g.stroke({ color: 0xb77d56, width: 1.6, alpha: 0.65 });
  }

  private drawGolGraphics(g: Graphics): void {
    g.clear();
    // Outer emerald/gold halo
    g.circle(0, 2, 20);
    g.fill({ color: 0xf5b942, alpha: 0.28 });
    g.circle(0, 2, 13);
    g.fill({ color: 0xf5b942 });
    g.stroke({ color: 0xfff3c4, width: 2 });
    // Specular highlight
    g.circle(-4, -2, 4.5);
    g.fill({ color: 0xffffff, alpha: 0.85 });
  }

  private rebuildHands(count: number): void {
    this.handsLayer.removeChildren();
    this.handNodes = [];

    const labelStyle = new TextStyle({
      fontFamily: 'Vazirmatn, system-ui, sans-serif',
      fontSize: count <= 2 ? 15 : 13,
      fontWeight: '700',
      fill: 0xf4f7f5,
    });
    const statusStyle = new TextStyle({
      fontFamily: 'Vazirmatn, system-ui, sans-serif',
      fontSize: 12,
      fontWeight: '800',
      fill: 0xffcf6b,
    });

    const targets = this.computeSlotTargets(count);

    for (let i = 0; i < count; i++) {
      const container = new Container();
      container.eventMode = 'static';
      container.cursor = 'pointer';

      const bgGlow = new Graphics();
      const closedArt = new Graphics();
      const openArt = new Graphics();
      const golArt = new Graphics();

      this.drawClosedFistGraphics(closedArt);
      this.drawOpenPalmGraphics(openArt);
      this.drawGolGraphics(golArt);

      openArt.visible = false;
      golArt.visible = false;

      const labelText = new Text({
        text: handLabelFa(i, count),
        style: labelStyle,
      });
      labelText.anchor.set(0.5, 0);
      labelText.position.set(0, 70);

      const statusText = new Text({
        text: '',
        style: statusStyle,
      });
      statusText.anchor.set(0.5, 1);
      statusText.position.set(0, -72);

      container.addChild(bgGlow, closedArt, openArt, golArt, labelText, statusText);

      // Generous invisible hit circle for mobile touch ergonomics
      bgGlow.circle(0, 2, 62);
      bgGlow.fill({ color: 0xffffff, alpha: 0.001 });

      const handIndex = i;
      container.on('pointertap', () => {
        if (this.currentState.phase === 'HUMAN_GUESS') {
          this.onSelectHand?.(handIndex);
        }
      });

      const tgt = targets[i]!;
      container.position.set(tgt.x, tgt.y);
      container.scale.set(tgt.scale);

      this.handsLayer.addChild(container);
      this.handNodes.push({
        index: i,
        container,
        bgGlow,
        closedArt,
        openArt,
        golArt,
        labelText,
        statusText,
        baseX: tgt.x,
        baseY: tgt.y,
        isOpen: false,
        hasGol: false,
      });
    }
  }

  private layoutHands(): void {
    const targets = this.computeSlotTargets(this.handNodes.length);
    for (let i = 0; i < this.handNodes.length; i++) {
      const node = this.handNodes[i]!;
      const tgt = targets[i];
      if (!tgt) continue;
      node.baseX = tgt.x;
      node.baseY = tgt.y;
      if (!this.choreoActive) {
        node.container.position.set(tgt.x, tgt.y);
      }
      node.container.scale.set(tgt.scale);
    }
  }

  private applyStateVisuals(): void {
    const st = this.currentState;
    for (const node of this.handNodes) {
      const isRevealedPhase = st.phase === 'REVEALED';
      const isOpenedEmpty = st.openedEmptyHands.includes(node.index);
      const hasGol = st.revealedGolHandIndex === node.index;
      const shouldOpen = isRevealedPhase || isOpenedEmpty;
      const isSelected = st.selectedHandIndex === node.index;

      node.isOpen = shouldOpen;
      node.hasGol = isRevealedPhase && hasGol;

      node.closedArt.visible = !shouldOpen;
      node.openArt.visible = shouldOpen;
      node.golArt.visible = isRevealedPhase && hasGol;

      node.bgGlow.clear();
      node.bgGlow.circle(0, 4, 60);
      if (isRevealedPhase && hasGol) {
        node.bgGlow.fill({ color: 0x10b981, alpha: 0.22 });
        node.bgGlow.stroke({ color: 0x34d399, width: 3 });
      } else if (isOpenedEmpty) {
        node.bgGlow.fill({ color: 0x38bdf8, alpha: 0.1 });
        node.bgGlow.stroke({ color: 0x38bdf8, width: 2, alpha: 0.65 });
      } else if (isSelected && st.phase === 'HUMAN_GUESS') {
        node.bgGlow.fill({ color: 0xf5b942, alpha: 0.2 });
        node.bgGlow.stroke({ color: 0xffcf6b, width: 3.2 });
      } else {
        node.bgGlow.fill({ color: 0xffffff, alpha: 0.001 });
        if (st.phase === 'HUMAN_GUESS') {
          node.bgGlow.stroke({ color: 0xf5b942, width: 1.5, alpha: 0.38 });
        }
      }

      if (st.phase === 'COMPUTER_HIDES') {
        node.labelText.text = '؟';
        node.statusText.text = '';
      } else {
        node.labelText.text = handLabelFa(node.index, st.handsCount);
        if (isRevealedPhase && hasGol) {
          node.statusText.text = 'گل ✓';
          node.statusText.style.fill = 0x6ee7b7;
        } else if (isOpenedEmpty || (isRevealedPhase && !hasGol)) {
          node.statusText.text = 'پوچ';
          node.statusText.style.fill = 0x7dd3fc;
        } else if (isSelected) {
          node.statusText.text = 'انتخاب شما';
          node.statusText.style.fill = 0xffcf6b;
        } else {
          node.statusText.text = '';
        }
      }

      node.container.eventMode =
        st.phase === 'HUMAN_GUESS' && !isOpenedEmpty ? 'static' : 'none';
      node.container.cursor =
        st.phase === 'HUMAN_GUESS' && !isOpenedEmpty ? 'pointer' : 'default';
    }
  }

  private tick(deltaMs: number): void {
    this.idleClockMs += deltaMs;

    if (this.choreoActive && this.choreography) {
      const steps = this.choreography.steps;
      const step = steps[this.choreoStepIndex];
      if (!step) {
        this.choreoActive = false;
        this.layoutHands();
        const cb = this.choreoOnDone;
        this.choreoOnDone = null;
        cb?.();
        return;
      }

      this.choreoStepElapsedMs += deltaMs;
      const rawT = Math.min(1, this.choreoStepElapsedMs / Math.max(1, step.durationMs));
      const nodeA = this.handNodes[step.slotA];
      const nodeB = this.handNodes[step.slotB];

      // Reset all nodes to base position before applying active step offsets
      for (const n of this.handNodes) {
        n.container.position.set(n.baseX, n.baseY);
        n.container.rotation = 0;
      }

      if (step.type === 'SWAP' && nodeA && nodeB) {
        // Out-and-back sleek sleight-of-hand arc swap passing near center
        const t = easeInOutCubic(rawT);
        const phase = t < 0.5 ? t * 2 : (1 - t) * 2;
        const arcHeight = 46;
        const ax = nodeA.baseX + (nodeB.baseX - nodeA.baseX) * phase;
        const ay =
          nodeA.baseY + (nodeB.baseY - nodeA.baseY) * phase - Math.sin(phase * Math.PI) * arcHeight;
        const bx = nodeB.baseX + (nodeA.baseX - nodeB.baseX) * phase;
        const by =
          nodeB.baseY + (nodeA.baseY - nodeB.baseY) * phase + Math.sin(phase * Math.PI) * arcHeight;
        nodeA.container.position.set(ax, ay);
        nodeB.container.position.set(bx, by);
      } else if (step.type === 'FEINT_SWAP' && nodeA && nodeB) {
        // Halfway fake transfer that snaps back with snappy easing
        const feintDist = 0.42;
        const p = rawT < 0.55 ? easeInOutCubic(rawT / 0.55) : 1 - easeOutBack((rawT - 0.55) / 0.45);
        const frac = feintDist * Math.max(0, p);
        nodeA.container.position.set(
          nodeA.baseX + (nodeB.baseX - nodeA.baseX) * frac,
          nodeA.baseY + (nodeB.baseY - nodeA.baseY) * frac - frac * 30
        );
        nodeB.container.position.set(
          nodeB.baseX + (nodeA.baseX - nodeB.baseX) * frac,
          nodeB.baseY + (nodeA.baseY - nodeB.baseY) * frac + frac * 30
        );
        nodeA.container.rotation = Math.sin(rawT * Math.PI * 2) * 0.08;
      } else if (step.type === 'MISDIRECTION_CROSS' && nodeA && nodeB) {
        // Quick cross + decoy pulse on a third hand
        const p = Math.sin(rawT * Math.PI);
        nodeA.container.position.set(
          nodeA.baseX + (nodeB.baseX - nodeA.baseX) * p * 0.65,
          nodeA.baseY - p * 38
        );
        nodeB.container.position.set(
          nodeB.baseX + (nodeA.baseX - nodeB.baseX) * p * 0.65,
          nodeB.baseY + p * 38
        );
        const decoyIdx = (step.slotA + 1) % this.handNodes.length;
        const decoy = this.handNodes[decoyIdx];
        if (decoy && decoy !== nodeA && decoy !== nodeB) {
          decoy.container.position.set(decoy.baseX, decoy.baseY - Math.sin(rawT * Math.PI * 2) * 14);
        }
      }

      if (rawT >= 1) {
        this.choreoStepIndex += 1;
        this.choreoStepElapsedMs = 0;
      }
      return;
    }

    // Gentle idle breathing motion during HUMAN_GUESS (disabled when reducedMotion is true)
    if (this.currentState.phase === 'HUMAN_GUESS' && !this.currentState.reducedMotion) {
      for (let i = 0; i < this.handNodes.length; i++) {
        const n = this.handNodes[i]!;
        if (this.currentState.openedEmptyHands.includes(n.index)) continue;
        const bob = Math.sin(this.idleClockMs * 0.003 + i * 1.1) * 3.5;
        n.container.position.set(n.baseX, n.baseY + bob);
      }
    }
  }
}
