import { getCupSpriteUrl } from './assets';
import type { NapkinAudioController } from './audio';
import { makeRng } from './rng';
import type { CompanionClipId, CupPhase, DrinkId, Point2D, ToolId, VolumeKey } from './types';

interface DrinkSpec {
  readonly name: string;
  readonly sub: string;
  readonly wash: readonly [number, number, number];
  readonly ring: readonly [number, number, number];
  readonly speck: readonly [number, number, number];
  readonly speckDensity: number;
}

const VOLUME_STOPS = [
  { key: '100', val: 1.0 },
  { key: '65', val: 0.65 },
  { key: '30', val: 0.3 },
  { key: '05', val: 0.05 },
] as const satisfies readonly { key: VolumeKey; val: number }[];

const DRINKS = {
  coffee_latte: {
    name: 'Coffee Latte',
    sub: 'Silky microfoam',
    wash: [158, 114, 68],
    ring: [112, 72, 36],
    speck: [96, 60, 28],
    speckDensity: 0.1,
  },
  black_coffee: {
    name: 'Black Coffee',
    sub: 'Dark roast & crema',
    wash: [110, 60, 22],
    ring: [62, 30, 10],
    speck: [48, 22, 8],
    speckDensity: 0.16,
  },
  matcha_pure: {
    name: 'Ceremonial Matcha',
    sub: 'Whisked Uji froth',
    wash: [96, 134, 44],
    ring: [54, 92, 24],
    speck: [38, 68, 16],
    speckDensity: 0.68,
  },
  matcha_latte: {
    name: 'Matcha Latte',
    sub: 'Creamy jade swirl',
    wash: [130, 158, 88],
    ring: [84, 114, 48],
    speck: [66, 94, 34],
    speckDensity: 0.38,
  },
} as const satisfies Record<DrinkId, DrinkSpec>;

const DRINK_ORDER: readonly DrinkId[] = ['coffee_latte', 'black_coffee', 'matcha_pure', 'matcha_latte'];

interface BlobSpec {
  x: number;
  y: number;
  r: number;
  irr: number;
  seed: number;
}

interface SatelliteSpec {
  x: number;
  y: number;
  r: number;
  seed: number;
}

interface RingSpec {
  radius: number;
  rot: number;
  heavyAngle: number;
  gapAngle: number;
  intensity: number;
}

interface StainItem {
  kind: 'drip' | 'ring';
  x: number;
  y: number;
  drinkId: DrinkId;
  seed: number;
  tStart: number;
  dryDur: number;
  blobs: BlobSpec[];
  satellites: SatelliteSpec[];
  ringSpec: RingSpec | null;
}

interface PointerDragState {
  pointerId: number;
  dragPx: number;
  downX: number;
  downY: number;
  lastX: number;
  lastY: number;
  grabOffX: number;
  grabOffY: number;
  originBowlX: number;
  originBowlY: number;
  dragging: boolean;
  movedDist: number;
  holdTimer: ReturnType<typeof setTimeout>;
}

export interface ViewTransform {
  x: number;
  y: number;
  scale: number;
}

export interface NapkinCupDeps {
  canvas: HTMLCanvasElement;
  cupCanvas: HTMLCanvasElement;
  stageEl: HTMLElement;
  audio: NapkinAudioController;
  getTool(): ToolId;
  getItemCount(): number;
  isCanvasBusy(): boolean;
  isTyping(): boolean;
  isPointInsideTypingTarget(x: number, y: number): boolean;
  commitTextBox(): void;
  restoreFocus(selector: string): void;
  render(): void;
  scheduleRender(): void;
  getViewTransform?(): ViewTransform;
  isMultiTouch?(): boolean;
}

export interface NapkinCupController {
  init(): void;
  orderDrink(id: DrinkId): void;
  markDrawingInteraction(): void;
  hitsCup(px: number, py: number): boolean;
  renderStainsTo(ctx: CanvasRenderingContext2D, now?: number): void;
  renderCupLayer(now?: number): void;
  triggerDrink(opts?: { manual?: boolean; forceDrip?: boolean }): boolean;
  triggerRefill(): boolean;
  onActivity(kind?: string): void;
  onUndo(): void;
  hasDrink(): boolean;
  liftForCrumple(): Promise<boolean | 'cancelled'>;
  resetSheetForCrumple(): void;
  returnAfterCrumple(): void;
  getState(): {
    hasDrink: boolean;
    isDrinkDone: boolean;
    drinkId: DrinkId;
    volume: number;
    targetVolume: number;
    phase: CupPhase;
    rotDeg: number;
    cupNX: number;
    cupNY: number;
    mugMoveCount: number;
    hasInteractedWithDrawing: boolean;
    stainCount: number;
    loadedCups: number;
  };
}

export function createNapkinCup(deps: NapkinCupDeps): NapkinCupController {
  const cupImages = new Map<string, HTMLImageElement>();
  let loadedCups = 0;

  const state = {
    hasDrink: false,
    isDrinkDone: false,
    drinkId: 'coffee_latte' as DrinkId,
    pendingOrderDrinkId: null as DrinkId | null,
    volume: 1.0,
    targetVolume: 1.0,
    cupNX: 0.88,
    cupNY: 0.22,
    rotDeg: 22,
    angVel: 0,
    lift: 0,
    targetLift: 0,
    phase: 'idle' as CupPhase,
    phaseStart: 0,
    drinkDurMs: 2100,
    nextVolAfterSip: 0.65,
    pendingDrip: false,
    dripFired: false,
    setdownPlayed: false,
    refillFrom: 0.05,
    refillProgress: null as number | null,
    stains: [] as StainItem[],
  };

  let crumpleLiftResolvers: Array<(result: boolean | 'cancelled') => void> = [];

  const HOLD_MS = 185;
  const DRAG_PX = 6;
  let pointerState: PointerDragState | null = null;
  let hoveringCup = false;
  let animRaf: number | null = null;

  let hasInteractedWithDrawing = false;
  let mugMoveCount = 0;
  let totalActivity = 0;
  let recSinceVocal = 0;
  let recSinceDrink = 0;
  let lastVocalAt = 0;
  let lastDrinkAt = 0;
  let hasEurekaOnSheet = false;
  let undoTimes: number[] = [];
  let idleTimer: ReturnType<typeof setTimeout> | null = null;
  let idleFiredThisStretch = false;
  let pendingTimers: ReturnType<typeof setTimeout>[] = [];

  const MIN_VOCAL_COOLDOWN_MS = 5200;
  const MIN_DRINK_COOLDOWN_MS = 12500;

  const ENTER_MS = 480;
  const REPLACE_OUT_MS = 350;
  const EXIT_MS = 420;
  const RETURN_MS = 440;
  const REFILL_MS = 2250;
  const CRUMPLE_LIFT_MS = 380;
  const CRUMPLE_RETURN_MS = 440;

  function scheduleTimer(fn: () => void, ms: number): ReturnType<typeof setTimeout> {
    const t = setTimeout(() => {
      pendingTimers = pendingTimers.filter(x => x !== t);
      fn();
    }, ms);
    pendingTimers.push(t);
    return t;
  }

  function clearPendingTimers(): void {
    pendingTimers.forEach(t => clearTimeout(t));
    pendingTimers = [];
  }

  function preloadCups(): void {
    for (const dId of DRINK_ORDER) {
      for (const s of VOLUME_STOPS) {
        const key = `${dId}_${s.key}`;
        const img = new Image();
        img.onload = () => {
          loadedCups++;
          deps.scheduleRender();
        };
        img.src = getCupSpriteUrl(dId, s.key);
        cupImages.set(key, img);
      }
    }
  }

  function getStageSize(): { w: number; h: number } {
    const r = deps.canvas.getBoundingClientRect();
    return { w: r.width || 1000, h: r.height || 660 };
  }

  function getCupSizePx(w: number, h: number): number {
    const m = Math.min(w, h);
    if (m >= 545) return Math.min(500, m * 0.66);
    const t = Math.max(0, Math.min(1, (m - 320) / 225));
    const ratio = 0.48 + 0.18 * t;
    return Math.max(155, Math.min(360, m * ratio));
  }

  function getCupCenter(w: number, h: number): { x: number; y: number; sz: number } {
    const sz = getCupSizePx(w, h);
    const minX = -sz * 0.04;
    const maxX = w + sz * 0.04;
    const minY = -sz * 0.04;
    const maxY = h + sz * 0.04;
    return {
      x: Math.max(minX, Math.min(maxX, state.cupNX * w)),
      y: Math.max(minY, Math.min(maxY, state.cupNY * h)),
      sz,
    };
  }

  function getBowlCenter(cx: number, cy: number, sz: number, rotDeg: number): Point2D {
    const rad = (rotDeg * Math.PI) / 180;
    const off = -sz * 0.088;
    return { x: cx + Math.cos(rad) * off, y: cy + Math.sin(rad) * off };
  }

  function hitsCup(px: number, py: number, isTouch = false): boolean {
    if (!state.hasDrink || state.phase !== 'idle') return false;
    const { w, h } = getStageSize();
    const { x, y, sz } = getCupCenter(w, h);
    const bowl = getBowlCenter(x, y, sz, state.rotDeg);
    const bowlR = sz * 0.335;
    if (Math.hypot(px - bowl.x, py - bowl.y) <= bowlR * (isTouch ? 1.08 : 1.03)) return true;
    const rad = -(state.rotDeg * Math.PI) / 180;
    const dx = px - x;
    const dy = py - y;
    const lx = dx * Math.cos(rad) - dy * Math.sin(rad);
    const ly = dx * Math.sin(rad) + dy * Math.cos(rad);
    return lx >= sz * 0.2 && lx <= sz * 0.45 && Math.abs(ly) <= sz * (isTouch ? 0.12 : 0.09);
  }

  function syncMenuBtn(): void {
    const btn = document.getElementById('drink-menu-btn');
    const sub = document.getElementById('drink-menu-sub');
    const needsDrink = !state.hasDrink || state.isDrinkDone;
    if (btn) {
      btn.classList.toggle('menu-prominent', needsDrink);
      btn.title = !state.hasDrink
        ? 'Open Café Menu to order a drink'
        : state.isDrinkDone
          ? 'Drink finished — open Menu to order another!'
          : 'Open Café Menu to replace drink';
    }
    if (sub) {
      sub.textContent = !state.hasDrink
        ? 'Get caffeinated for better ideas'
        : state.isDrinkDone
          ? 'Cup finished — order a fresh drink'
          : 'Ordering replaces your current drink';
    }
    document.querySelectorAll<HTMLButtonElement>('button.drink-opt').forEach(b => {
      b.classList.toggle('active', state.hasDrink && b.dataset.drink === state.drinkId);
    });
  }

  function buildBlobContour(cx: number, cy: number, baseR: number, seed: number, irr = 0.28, segs = 34): Point2D[] {
    const rnd = makeRng(seed);
    const p1 = rnd() * 6.28, p2 = rnd() * 6.28, p3 = rnd() * 6.28, p4 = rnd() * 6.28;
    const a1 = (0.4 + rnd() * 0.6) * irr * 0.45;
    const a2 = (0.4 + rnd() * 0.6) * irr * 0.35;
    const a3 = (0.3 + rnd() * 0.7) * irr * 0.22;
    const a4 = (0.3 + rnd() * 0.7) * irr * 0.14;
    const ax = 0.88 + rnd() * 0.26;
    const ay = 0.88 + rnd() * 0.26;
    const rot = rnd() * 6.28;
    const c = Math.cos(rot);
    const s = Math.sin(rot);
    const pts: Point2D[] = [];
    for (let i = 0; i < segs; i++) {
      const th = (i / segs) * 6.2831853;
      const w =
        1 +
        a1 * Math.sin(th * 2 + p1) +
        a2 * Math.cos(th * 3 + p2) +
        a3 * Math.sin(th * 5 + p3) +
        a4 * Math.cos(th * 9 + p4);
      const lx = Math.cos(th) * baseR * w * ax;
      const ly = Math.sin(th) * baseR * w * ay;
      pts.push({ x: cx + lx * c - ly * s, y: cy + lx * s + ly * c });
    }
    return pts;
  }

  function traceClosed(ctx: CanvasRenderingContext2D, pts: readonly Point2D[]): void {
    const n = pts.length;
    if (n === 0) return;
    ctx.beginPath();
    ctx.moveTo((pts[n - 1]!.x + pts[0]!.x) * 0.5, (pts[n - 1]!.y + pts[0]!.y) * 0.5);
    for (let i = 0; i < n; i++) {
      const p = pts[i]!;
      const q = pts[(i + 1) % n]!;
      ctx.quadraticCurveTo(p.x, p.y, (p.x + q.x) * 0.5, (p.y + q.y) * 0.5);
    }
    ctx.closePath();
  }

  function createStain(
    kind: 'drip' | 'ring',
    x: number,
    y: number,
    drinkId: DrinkId,
    scale: number,
    bowlRadius: number,
  ): StainItem {
    const seed = (Math.floor(Math.random() * 0xffffffff) ^ (Date.now() & 0xffffffff)) >>> 0;
    const rnd = makeRng(seed);
    const now = performance.now();
    const st: StainItem = {
      kind,
      x,
      y,
      drinkId,
      seed,
      tStart: now,
      dryDur: kind === 'ring' ? 760 : 660,
      blobs: [],
      satellites: [],
      ringSpec: null,
    };
    if (kind === 'drip') {
      const r = (7 + rnd() * 7) * scale;
      st.blobs.push({ x, y, r, irr: 0.28, seed: seed ^ 0x111 });
      const n = 2 + Math.floor(rnd() * 3);
      for (let i = 0; i < n; i++) {
        const a = rnd() * 6.28;
        const d = r * (1.25 + rnd() * 1.45);
        st.satellites.push({
          x: x + Math.cos(a) * d,
          y: y + Math.sin(a) * d,
          r: (1.2 + rnd() * 2.2) * scale,
          seed: seed ^ (i * 919 + 55),
        });
      }
    } else {
      st.ringSpec = {
        radius: bowlRadius * 0.88,
        rot: rnd() * 6.28,
        heavyAngle: rnd() * 6.28,
        gapAngle: rnd() * 6.28,
        intensity: (0.48 + rnd() * 0.28) * Math.min(1.2, scale),
      };
      const ha = st.ringSpec.heavyAngle;
      if (rnd() < 0.55) {
        st.satellites.push({
          x: x + Math.cos(ha) * (st.ringSpec.radius + 4),
          y: y + Math.sin(ha) * (st.ringSpec.radius + 4),
          r: (1.8 + rnd() * 2.2) * scale,
          seed: seed ^ 0x9191,
        });
      }
    }
    return st;
  }

  function randomNapkinDripPos(w: number, h: number): Point2D {
    const padX = Math.max(48, Math.min(96, w * 0.1));
    const padY = Math.max(48, Math.min(84, h * 0.1));
    return {
      x: padX + Math.random() * Math.max(20, w - padX * 2),
      y: padY + Math.random() * Math.max(20, h - padY * 2),
    };
  }

  function addStain(kind: 'drip' | 'trail' | 'ring', x: number, y: number, scale = 1.0): void {
    const { w, h } = getStageSize();
    const bowlR = getCupSizePx(w, h) * 0.335;
    if (kind === 'trail') {
      const ang = Math.random() * 6.2831853;
      const count = 3 + (Math.random() < 0.6 ? 1 : 0);
      for (let i = 0; i < count; i++) {
        const d = i * (20 + Math.random() * 14);
        const p = Math.sin(i * 1.15) * 9.0;
        const sx = Math.max(28, Math.min(w - 28, x + Math.cos(ang) * d - Math.sin(ang) * p));
        const sy = Math.max(28, Math.min(h - 28, y + Math.sin(ang) * d + Math.cos(ang) * p));
        const st = createStain('drip', sx, sy, state.drinkId, scale * (0.98 - i * 0.16), bowlR);
        st.tStart = performance.now() + i * 90;
        state.stains.push(st);
      }
    } else {
      state.stains.push(createStain(kind, x, y, state.drinkId, scale, bowlR));
    }
    if (state.stains.length > 28) state.stains = state.stains.slice(-28);
    kickAnim();
  }

  function renderOneStain(ctx: CanvasRenderingContext2D, st: StainItem, now: number): void {
    if (now < st.tStart) return;
    const d = DRINKS[st.drinkId];
    const rawT = Math.max(0, Math.min(1, (now - st.tStart) / st.dryDur));
    const spread = 0.42 + 0.58 * (1 - Math.pow(1 - rawT, 3));
    const ringDry = Math.pow(rawT, 0.7);
    const [wr, wg, wb] = d.wash;
    const [rr, rg, rb] = d.ring;
    const [sr, sg, sb] = d.speck;

    ctx.save();
    ctx.globalCompositeOperation = 'multiply';

    if (st.kind === 'ring' && st.ringSpec) {
      const rs = st.ringSpec;
      const rad = rs.radius * (0.97 + 0.03 * spread);
      const segs = 104;
      const rnd = makeRng(st.seed ^ 0x777);
      const outerBleed: Point2D[] = [];
      const innerBleed: Point2D[] = [];
      const outerBody: Point2D[] = [];
      const innerBody: Point2D[] = [];
      for (let i = 0; i < segs; i++) {
        const a = (i / segs) * 6.2831853;
        const heavy = 0.5 + 0.5 * Math.cos(a - rs.heavyAngle);
        const drySide = Math.max(0, Math.cos(a - rs.gapAngle));
        const dryMask = Math.max(0.04, 1 - Math.pow(drySide, 2.4) * 0.94);
        const wob =
          Math.sin(a * 3 + rs.rot) * 1.1 +
          Math.cos(a * 5 - rs.rot * 1.7) * 0.6 +
          Math.sin(a * 8 + rs.rot * 0.6) * 0.25;
        const rMid = rad + wob;
        const halfW = (1.0 + 3.1 * Math.pow(heavy, 1.5)) * dryMask * (0.65 + 0.35 * spread);
        const bleedW = halfW * 1.9 + 1.3;
        const ca = Math.cos(a);
        const sa = Math.sin(a);
        outerBody.push({ x: st.x + ca * (rMid + halfW), y: st.y + sa * (rMid + halfW) });
        innerBody.push({ x: st.x + ca * Math.max(2, rMid - halfW), y: st.y + sa * Math.max(2, rMid - halfW) });
        outerBleed.push({ x: st.x + ca * (rMid + bleedW), y: st.y + sa * (rMid + bleedW) });
        innerBleed.push({ x: st.x + ca * Math.max(2, rMid - bleedW), y: st.y + sa * Math.max(2, rMid - bleedW) });
      }
      const fillRibbon = (op: readonly Point2D[], ip: readonly Point2D[], fillStyle: CanvasGradient) => {
        const n = op.length;
        ctx.beginPath();
        ctx.moveTo((op[n - 1]!.x + op[0]!.x) * 0.5, (op[n - 1]!.y + op[0]!.y) * 0.5);
        for (let i = 0; i < n; i++) {
          const p = op[i]!;
          const q = op[(i + 1) % n]!;
          ctx.quadraticCurveTo(p.x, p.y, (p.x + q.x) * 0.5, (p.y + q.y) * 0.5);
        }
        ctx.closePath();
        ctx.moveTo((ip[0]!.x + ip[n - 1]!.x) * 0.5, (ip[0]!.y + ip[n - 1]!.y) * 0.5);
        for (let i = n - 1; i >= 0; i--) {
          const p = ip[i]!;
          const q = ip[(i - 1 + n) % n]!;
          ctx.quadraticCurveTo(p.x, p.y, (p.x + q.x) * 0.5, (p.y + q.y) * 0.5);
        }
        ctx.closePath();
        ctx.fillStyle = fillStyle;
        ctx.fill('evenodd');
      };

      const hx = st.x + Math.cos(rs.heavyAngle) * rad;
      const hy = st.y + Math.sin(rs.heavyAngle) * rad;
      const ox = st.x - Math.cos(rs.heavyAngle) * rad;
      const oy = st.y - Math.sin(rs.heavyAngle) * rad;

      const bleedGrad = ctx.createLinearGradient(hx, hy, ox, oy);
      bleedGrad.addColorStop(0, `rgba(${wr},${wg},${wb},${(0.14 * rs.intensity).toFixed(3)})`);
      bleedGrad.addColorStop(0.55, `rgba(${wr},${wg},${wb},${(0.07 * rs.intensity).toFixed(3)})`);
      bleedGrad.addColorStop(1, `rgba(${wr},${wg},${wb},${(0.025 * rs.intensity).toFixed(3)})`);
      fillRibbon(outerBleed, innerBleed, bleedGrad);

      const bodyGrad = ctx.createLinearGradient(hx, hy, ox, oy);
      bodyGrad.addColorStop(0, `rgba(${wr},${wg},${wb},${(0.32 * rs.intensity).toFixed(3)})`);
      bodyGrad.addColorStop(0.55, `rgba(${wr},${wg},${wb},${(0.18 * rs.intensity).toFixed(3)})`);
      bodyGrad.addColorStop(1, `rgba(${wr},${wg},${wb},${(0.06 * rs.intensity).toFixed(3)})`);
      fillRibbon(outerBody, innerBody, bodyGrad);

      const rimGrad = ctx.createLinearGradient(hx, hy, ox, oy);
      rimGrad.addColorStop(0, `rgba(${rr},${rg},${rb},${((0.18 + 0.28 * ringDry) * rs.intensity).toFixed(3)})`);
      rimGrad.addColorStop(0.55, `rgba(${rr},${rg},${rb},${((0.09 + 0.14 * ringDry) * rs.intensity).toFixed(3)})`);
      rimGrad.addColorStop(1, `rgba(${rr},${rg},${rb},${((0.03 + 0.04 * ringDry) * rs.intensity).toFixed(3)})`);
      ctx.strokeStyle = rimGrad;
      ctx.lineWidth = 1.1;
      traceClosed(ctx, outerBody);
      ctx.stroke();
      ctx.lineWidth = 0.85;
      traceClosed(ctx, innerBody);
      ctx.stroke();

      const pool = ctx.createRadialGradient(hx, hy, 1, hx, hy, 12 * spread);
      pool.addColorStop(0, `rgba(${rr},${rg},${rb},${(0.2 * rs.intensity).toFixed(3)})`);
      pool.addColorStop(1, `rgba(${wr},${wg},${wb},0)`);
      ctx.fillStyle = pool;
      ctx.beginPath();
      ctx.arc(hx, hy, 12 * spread, 0, 6.2831853);
      ctx.fill();

      const speckN = Math.round(36 * d.speckDensity);
      ctx.fillStyle = `rgba(${sr},${sg},${sb},${(0.3 * ringDry).toFixed(3)})`;
      for (let i = 0; i < speckN; i++) {
        const a = rnd() * 6.2831853;
        const rj = rad + (rnd() - 0.5) * 4.5;
        ctx.fillRect(st.x + Math.cos(a) * rj, st.y + Math.sin(a) * rj, 1.1, 1.1);
      }
    }

    for (const b of st.blobs) {
      const curR = Math.max(2, b.r * spread);
      const pts = buildBlobContour(b.x, b.y, curR, b.seed, b.irr, 32);
      traceClosed(ctx, buildBlobContour(b.x, b.y, curR * 1.12, b.seed ^ 0x55, b.irr * 1.1, 32));
      ctx.fillStyle = `rgba(${wr},${wg},${wb},${(0.075 * spread).toFixed(3)})`;
      ctx.fill();

      traceClosed(ctx, pts);
      const grad = ctx.createRadialGradient(b.x, b.y, curR * 0.12, b.x, b.y, curR * 1.15);
      grad.addColorStop(0, `rgba(${wr},${wg},${wb},${(0.25 - 0.05 * ringDry).toFixed(3)})`);
      grad.addColorStop(0.72, `rgba(${wr},${wg},${wb},${(0.29 + 0.04 * ringDry).toFixed(3)})`);
      grad.addColorStop(1, `rgba(${rr},${rg},${rb},${(0.22 + 0.15 * ringDry).toFixed(3)})`);
      ctx.fillStyle = grad;
      ctx.fill();

      traceClosed(ctx, pts);
      ctx.strokeStyle = `rgba(${rr},${rg},${rb},${(0.12 + 0.32 * ringDry).toFixed(3)})`;
      ctx.lineWidth = Math.max(1.0, Math.min(2.4, curR * 0.085));
      ctx.stroke();

      const speckN = Math.round(b.r * 1.2 * d.speckDensity);
      if (speckN > 0) {
        const sRnd = makeRng(b.seed ^ 0x4242);
        ctx.fillStyle = `rgba(${sr},${sg},${sb},${(0.32 * ringDry).toFixed(3)})`;
        for (let i = 0; i < speckN; i++) {
          const a = sRnd() * 6.2831853;
          const rr2 = curR * (0.55 + Math.pow(sRnd(), 0.45) * 0.43);
          ctx.beginPath();
          ctx.arc(b.x + Math.cos(a) * rr2, b.y + Math.sin(a) * rr2, 0.55 + sRnd() * 0.75, 0, 6.2831853);
          ctx.fill();
        }
      }
    }

    for (const s of st.satellites) {
      const curR = Math.max(1, s.r * spread);
      traceClosed(ctx, buildBlobContour(s.x, s.y, curR, s.seed, 0.22, 16));
      ctx.fillStyle = `rgba(${wr},${wg},${wb},0.34)`;
      ctx.fill();
      ctx.strokeStyle = `rgba(${rr},${rg},${rb},${(0.18 + 0.24 * ringDry).toFixed(3)})`;
      ctx.lineWidth = 0.8;
      ctx.stroke();
    }
    ctx.restore();

    if (rawT < 0.6 && st.blobs.length) {
      const b0 = st.blobs[0]!;
      const curR = b0.r * spread;
      ctx.save();
      ctx.fillStyle = `rgba(255, 252, 244, ${((1 - rawT / 0.6) * 0.26).toFixed(3)})`;
      ctx.beginPath();
      ctx.ellipse(b0.x - curR * 0.24, b0.y - curR * 0.26, curR * 0.28, curR * 0.14, -0.5, 0, 6.2831853);
      ctx.fill();
      ctx.restore();
    }
  }

  function renderStainsTo(ctx: CanvasRenderingContext2D, now = performance.now()): void {
    for (const st of state.stains) renderOneStain(ctx, st, now);
  }

  function drawCupSprite(
    ctx: CanvasRenderingContext2D,
    cx: number,
    cy: number,
    sz: number,
    drinkId: DrinkId,
    volume: number,
    rotDeg: number,
    lift: number,
    opacity: number,
    directRefillT: number | null = null,
  ): void {
    if (opacity <= 0.005) return;
    const rad = (rotDeg * Math.PI) / 180;
    const scale = 1 + lift * 0.16;
    const bowlOffX = -sz * 0.088;
    const bowlR = sz * 0.335;

    ctx.save();
    ctx.translate(cx, cy);
    ctx.globalCompositeOperation = 'source-over';
    const shFade = Math.max(0, 1 - lift * 0.42);
    ctx.globalAlpha = opacity * shFade;
    const shDist = sz * (0.048 + lift * 0.14);
    const shDx = shDist * 0.9;
    const shDy = shDist * 0.95;
    const bx = Math.cos(rad) * bowlOffX;
    const by = Math.sin(rad) * bowlOffX;

    const penR = bowlR * (1.22 + lift * 0.3);
    const penA0 = Math.max(0.06, 0.28 - lift * 0.08);
    const penA1 = Math.max(0.03, 0.12 - lift * 0.03);
    const pen = ctx.createRadialGradient(bx + shDx, by + shDy, bowlR * 0.55, bx + shDx, by + shDy, penR);
    pen.addColorStop(0, `rgba(52, 36, 20, ${penA0.toFixed(3)})`);
    pen.addColorStop(0.7, `rgba(66, 46, 26, ${penA1.toFixed(3)})`);
    pen.addColorStop(1, 'rgba(66, 46, 26, 0)');
    ctx.fillStyle = pen;
    ctx.beginPath();
    ctx.arc(bx + shDx, by + shDy, penR, 0, 6.2831853);
    ctx.fill();

    if (lift < 0.85) {
      const occA = Math.max(0, (1 - lift) * 0.28);
      const occ = ctx.createRadialGradient(
        bx + shDx * 0.25,
        by + shDy * 0.25,
        bowlR * 0.72,
        bx + shDx * 0.25,
        by + shDy * 0.25,
        bowlR * 1.04,
      );
      occ.addColorStop(0, `rgba(38, 24, 12, ${occA.toFixed(3)})`);
      occ.addColorStop(0.85, `rgba(48, 32, 18, ${(occA * 0.55).toFixed(3)})`);
      occ.addColorStop(1, 'rgba(48, 32, 18, 0)');
      ctx.fillStyle = occ;
      ctx.beginPath();
      ctx.arc(bx + shDx * 0.25, by + shDy * 0.25, bowlR * 1.04, 0, 6.2831853);
      ctx.fill();
    }

    ctx.save();
    ctx.translate(shDx * 0.85, shDy * 0.9);
    ctx.rotate(rad);
    ctx.fillStyle = `rgba(52, 36, 20, ${Math.max(0.04, 0.18 - lift * 0.05).toFixed(3)})`;
    ctx.beginPath();
    ctx.roundRect(sz * 0.21, -sz * 0.055, sz * 0.22, sz * 0.11, sz * 0.045);
    ctx.fill();
    ctx.restore();
    ctx.restore();

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(rad);
    ctx.scale(scale, scale);
    const half = sz * 0.5;

    if (directRefillT !== null) {
      const imgEmpty = cupImages.get(`${drinkId}_05`);
      const imgFull = cupImages.get(`${drinkId}_100`);
      const fillT = Math.max(0, Math.min(1, directRefillT));
      if (imgEmpty && imgEmpty.complete && imgEmpty.naturalWidth) {
        ctx.globalAlpha = opacity;
        ctx.drawImage(imgEmpty, -half, -half, sz, sz);
      }
      if (fillT > 0.004 && imgFull && imgFull.complete && imgFull.naturalWidth) {
        ctx.globalAlpha = opacity * fillT;
        ctx.drawImage(imgFull, -half, -half, sz, sz);
      }
    } else {
      const v = Math.max(0.05, Math.min(1.0, volume));
      let hi: { key: VolumeKey; val: number } = VOLUME_STOPS[0];
      let lo: { key: VolumeKey; val: number } = VOLUME_STOPS[VOLUME_STOPS.length - 1]!;
      for (let i = 0; i < VOLUME_STOPS.length - 1; i++) {
        if (v <= VOLUME_STOPS[i]!.val && v >= VOLUME_STOPS[i + 1]!.val) {
          hi = VOLUME_STOPS[i]!;
          lo = VOLUME_STOPS[i + 1]!;
          break;
        }
      }
      const t = Math.max(0, Math.min(1, (hi.val - v) / Math.max(0.001, hi.val - lo.val)));
      const imgHi = cupImages.get(`${drinkId}_${hi.key}`);
      const imgLo = cupImages.get(`${drinkId}_${lo.key}`);
      if (imgHi && imgHi.complete && imgHi.naturalWidth) {
        ctx.globalAlpha = opacity;
        ctx.drawImage(imgHi, -half, -half, sz, sz);
      }
      if (t > 0.005 && imgLo && imgLo.complete && imgLo.naturalWidth) {
        ctx.globalAlpha = opacity * t;
        ctx.drawImage(imgLo, -half, -half, sz, sz);
      }
    }
    ctx.restore();
  }

  function exitOffset(cx: number, cy: number, w: number, h: number, sz: number): { dx: number; dy: number } {
    const dirX = cx >= w * 0.5 ? 1 : -1;
    const dx = dirX > 0 ? w - cx + sz * 0.72 : -(cx + sz * 0.72);
    const dy = Math.min(220, (h - cy) * 0.48 + sz * 0.42);
    return { dx, dy };
  }

  function renderCupTo(ctx: CanvasRenderingContext2D, w: number, h: number, now = performance.now()): void {
    if (!state.hasDrink) return;
    const { x, y, sz } = getCupCenter(w, h);
    let drawX = x;
    let drawY = y;
    let drawOpacity = 1;
    let drawLift = state.lift;

    if (state.phase === 'entering') {
      const t = Math.min(1, (now - state.phaseStart) / ENTER_MS);
      const inv = 1 - (1 - Math.pow(1 - t, 3));
      const { dx, dy } = exitOffset(x, y, w, h, sz);
      drawX = x + inv * dx * 0.65;
      drawY = y + inv * dy * 0.55;
      drawLift = inv * 1.65;
      drawOpacity = Math.min(1, Math.pow(t, 0.65));
    } else if (state.phase === 'replacing_out') {
      const t = Math.min(1, (now - state.phaseStart) / REPLACE_OUT_MS);
      const ease = t * t * (3 - 2 * t);
      const { dx, dy } = exitOffset(x, y, w, h, sz);
      drawX = x + ease * dx * 0.65;
      drawY = y + ease * dy * 0.55;
      drawLift = ease * 1.65;
      drawOpacity = Math.max(0, 1 - Math.pow(t, 1.4));
    } else if (state.phase === 'exiting') {
      const t = Math.min(1, (now - state.phaseStart) / EXIT_MS);
      const ease = t * t * (3 - 2 * t);
      const { dx, dy } = exitOffset(x, y, w, h, sz);
      drawX = x + ease * dx * 0.68;
      drawY = y + ease * dy * 0.65;
      drawLift = ease * 2.05;
      drawOpacity = Math.max(0, 1 - Math.pow(t, 1.65));
    } else if (state.phase === 'drinking') {
      drawOpacity = 0;
    } else if (state.phase === 'returning') {
      const t = Math.min(1, (now - state.phaseStart) / RETURN_MS);
      const ease = 1 - Math.pow(1 - t, 3);
      const inv = 1 - ease;
      const { dx, dy } = exitOffset(x, y, w, h, sz);
      drawX = x + inv * dx * 0.68;
      drawY = y + inv * dy * 0.65;
      drawLift = inv * 2.05;
      drawOpacity = Math.min(1, Math.pow(t, 0.65));
    } else if (state.phase === 'crumple_lift') {
      const t = Math.min(1, (now - state.phaseStart) / CRUMPLE_LIFT_MS);
      const ease = t * t * (3 - 2 * t);
      const { dx, dy } = exitOffset(x, y, w, h, sz);
      drawX = x + ease * dx * 0.72;
      drawY = y + ease * dy * 0.65;
      drawLift = Math.max(state.lift, ease * 1.95);
      drawOpacity = Math.max(0, 1 - Math.pow(t, 1.35));
    } else if (state.phase === 'crumple_wait') {
      drawOpacity = 0;
    } else if (state.phase === 'crumple_return') {
      const t = Math.min(1, (now - state.phaseStart) / CRUMPLE_RETURN_MS);
      const ease = 1 - Math.pow(1 - t, 3);
      const inv = 1 - ease;
      const { dx, dy } = exitOffset(x, y, w, h, sz);
      drawX = x + inv * dx * 0.72;
      drawY = y + inv * dy * 0.65;
      drawLift = inv * 1.9;
      drawOpacity = Math.min(1, Math.pow(t, 0.65));
    }

    const directRefillT = state.phase === 'refilling' ? state.refillProgress : null;
    drawCupSprite(ctx, drawX, drawY, sz, state.drinkId, state.volume, state.rotDeg, drawLift, drawOpacity, directRefillT);
  }

  function renderCupLayer(now = performance.now()): void {
    const sr = deps.stageEl.getBoundingClientRect();
    const nr = deps.canvas.getBoundingClientRect();
    if (!sr.width || !sr.height) return;
    const dpr = window.devicePixelRatio || 1;
    const cw = Math.round(sr.width * dpr);
    const ch = Math.round(sr.height * dpr);
    if (deps.cupCanvas.width !== cw || deps.cupCanvas.height !== ch) {
      deps.cupCanvas.width = cw;
      deps.cupCanvas.height = ch;
    }
    const cctx = deps.cupCanvas.getContext('2d');
    if (!cctx) return;
    cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    cctx.clearRect(0, 0, sr.width, sr.height);
    if (!state.hasDrink) return;

    const vt = deps.getViewTransform ? deps.getViewTransform() : { x: 0, y: 0, scale: 1 };
    cctx.save();
    cctx.translate(nr.left - sr.left + vt.x, nr.top - sr.top + vt.y);
    cctx.scale(vt.scale, vt.scale);
    renderCupTo(cctx, nr.width, nr.height, now);
    cctx.restore();
  }

  function needsAnimation(now: number): boolean {
    if (state.phase !== 'idle' && state.phase !== 'crumple_wait') return true;
    if (state.phase === 'idle') {
      if (Math.abs(state.targetLift - state.lift) > 0.004) return true;
      if (Math.abs(state.targetVolume - state.volume) > 0.002) return true;
      if (Math.abs(state.angVel) > 0.03) return true;
    }
    for (const st of state.stains) {
      if (now < st.tStart + st.dryDur + 40) return true;
    }
    return false;
  }

  function stepAnim(now: number): void {
    animRaf = null;

    if (state.phase === 'idle') {
      state.lift += (state.targetLift - state.lift) * 0.22;
      state.volume += (state.targetVolume - state.volume) * 0.16;
      if (Math.abs(state.targetVolume - state.volume) < 0.001) state.volume = state.targetVolume;
      if (!pointerState?.dragging) {
        state.rotDeg += state.angVel;
        state.angVel *= 0.84;
        if (Math.abs(state.angVel) < 0.025) state.angVel = 0;
      }
    } else if (state.phase === 'entering') {
      const t = Math.min(1, (now - state.phaseStart) / ENTER_MS);
      if (!state.setdownPlayed && t >= 0.84) {
        state.setdownPlayed = true;
        void deps.audio.playClip('cup_setdown', { channel: 'foley', volumeMul: 0.92 });
      }
      if (t >= 1) {
        state.phase = 'idle';
        state.lift = 0;
        state.targetLift = 0;
      }
    } else if (state.phase === 'replacing_out') {
      if (now - state.phaseStart >= REPLACE_OUT_MS) {
        state.drinkId = state.pendingOrderDrinkId ?? state.drinkId;
        state.pendingOrderDrinkId = null;
        state.volume = 1.0;
        state.targetVolume = 1.0;
        state.isDrinkDone = false;
        state.phase = 'entering';
        state.phaseStart = now;
        state.setdownPlayed = false;
        syncMenuBtn();
      }
    } else if (state.phase === 'exiting') {
      if (now - state.phaseStart >= EXIT_MS) {
        state.phase = 'drinking';
        state.phaseStart = now;
        state.volume = state.nextVolAfterSip;
        state.targetVolume = state.nextVolAfterSip;
        if (state.targetVolume <= 0.07) {
          state.isDrinkDone = true;
          syncMenuBtn();
        }
        const r = Math.random();
        const sipClip: CompanionClipId = r < 0.45 ? 'sip_coffee' : r < 0.75 ? 'sip_ahhh' : 'swallow_liquid';
        const baseDur = sipClip === 'sip_ahhh' ? 2550 : sipClip === 'sip_coffee' ? 2100 : 1850;
        state.drinkDurMs = baseDur + Math.round(Math.random() * 260);
        void deps.audio.playClip(sipClip, { channel: 'vocal', interrupt: true });
      }
    } else if (state.phase === 'drinking') {
      const t = Math.min(1, (now - state.phaseStart) / state.drinkDurMs);
      if (state.pendingDrip && !state.dripFired && t >= 0.56) {
        state.dripFired = true;
        const { w, h } = getStageSize();
        const pos = randomNapkinDripPos(w, h);
        addStain(Math.random() < 0.34 ? 'drip' : 'trail', pos.x, pos.y, 1.18);
        scheduleTimer(() => {
          void deps.audio.playClip('oops_drip', { channel: 'vocal', interrupt: true });
          lastVocalAt = performance.now();
        }, 220);
      }
      if (t >= 1) {
        state.phase = 'returning';
        state.phaseStart = now;
        state.setdownPlayed = false;
      }
    } else if (state.phase === 'returning') {
      const t = Math.min(1, (now - state.phaseStart) / RETURN_MS);
      if (!state.setdownPlayed && t >= 0.84) {
        state.setdownPlayed = true;
        void deps.audio.playClip('cup_setdown', { channel: 'foley', volumeMul: 0.92 });
      }
      if (t >= 1) {
        state.phase = 'idle';
        state.lift = 0;
        state.targetLift = 0;
      }
    } else if (state.phase === 'refilling') {
      const t = Math.min(1, (now - state.phaseStart) / REFILL_MS);
      const ease = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;
      state.refillProgress = ease;
      state.volume = 0.05 + 0.95 * ease;
      if (t >= 1) {
        state.refillProgress = null;
        state.volume = 1.0;
        state.targetVolume = 1.0;
        state.isDrinkDone = false;
        state.phase = 'idle';
        syncMenuBtn();
      }
    } else if (state.phase === 'crumple_lift') {
      if (now - state.phaseStart >= CRUMPLE_LIFT_MS) {
        state.phase = 'crumple_wait';
        const resolvers = crumpleLiftResolvers.splice(0);
        resolvers.forEach(fn => fn(true));
      }
    } else if (state.phase === 'crumple_return') {
      const t = Math.min(1, (now - state.phaseStart) / CRUMPLE_RETURN_MS);
      if (!state.setdownPlayed && t >= 0.84) {
        state.setdownPlayed = true;
        if (state.hasDrink) void deps.audio.playClip('cup_setdown', { channel: 'foley', volumeMul: 0.92 });
      }
      if (t >= 1) {
        state.phase = 'idle';
        state.lift = 0;
        state.targetLift = 0;
      }
    }

    deps.render();
    if (needsAnimation(now)) {
      animRaf = requestAnimationFrame(stepAnim);
    }
  }

  function kickAnim(): void {
    if (!animRaf) animRaf = requestAnimationFrame(stepAnim);
  }

  function orderDrink(id: DrinkId): void {
    clearPendingTimers();
    if (pointerState) finishPointer(true);
    state.refillProgress = null;
    if (crumpleLiftResolvers.length) {
      crumpleLiftResolvers.splice(0).forEach(fn => fn('cancelled'));
    }

    if (!state.hasDrink) {
      state.drinkId = id;
      state.hasDrink = true;
      state.isDrinkDone = false;
      state.volume = 1.0;
      state.targetVolume = 1.0;
      state.cupNX = 0.88;
      state.cupNY = 0.22;
      state.rotDeg = 22;
      state.phase = 'entering';
      state.phaseStart = performance.now();
      state.setdownPlayed = false;
      mugMoveCount = 0;
      syncMenuBtn();
      kickAnim();
    } else {
      state.pendingOrderDrinkId = id;
      state.phase = 'replacing_out';
      state.phaseStart = performance.now();
      mugMoveCount = 0;
      syncMenuBtn();
      kickAnim();
    }
    deps.restoreFocus('#drink-menu-btn');
  }

  function triggerRefill(): boolean {
    if (!state.hasDrink || state.phase !== 'idle') return false;
    state.phase = 'refilling';
    state.phaseStart = performance.now();
    state.refillFrom = 0.05;
    state.refillProgress = 0;
    state.targetVolume = 1.0;
    state.isDrinkDone = false;
    lastDrinkAt = performance.now();
    recSinceDrink = 0;
    syncMenuBtn();
    void deps.audio.playClip('refill_pour', { channel: 'foley' });
    kickAnim();
    return true;
  }

  function triggerDrink(opts: { manual?: boolean; forceDrip?: boolean } = {}): boolean {
    if (!state.hasDrink || state.phase !== 'idle') return false;
    if (state.targetVolume <= 0.07) {
      return triggerRefill();
    }
    const idx = VOLUME_STOPS.findIndex(s => s.val <= state.targetVolume + 0.02);
    const nextIdx = idx === -1 ? 1 : Math.min(VOLUME_STOPS.length - 1, idx + 1);
    state.nextVolAfterSip = VOLUME_STOPS[nextIdx]!.val;

    const dripProb = !hasInteractedWithDrawing
      ? 0
      : opts.forceDrip
        ? 1.0
        : state.targetVolume >= 0.9
          ? 0.72
          : 0.58;
    state.pendingDrip = Math.random() < dripProb;
    state.dripFired = false;
    state.phase = 'exiting';
    state.phaseStart = performance.now();
    lastDrinkAt = performance.now();
    recSinceDrink = 0;
    kickAnim();
    return true;
  }

  function setCupCursor(cur: string): void {
    deps.stageEl.style.cursor = cur || '';
    const t = deps.getTool();
    deps.canvas.style.cursor = cur || (t === 'move' ? 'default' : t === 'text' ? 'text' : 'crosshair');
  }

  function onPointerDown(e: PointerEvent, p: Point2D, captureEl: HTMLElement): boolean {
    const isTouch = e.pointerType === 'touch';
    if (pointerState) return true;
    if (!hitsCup(p.x, p.y, isTouch)) return false;
    e.preventDefault();
    try {
      captureEl.setPointerCapture(e.pointerId);
    } catch {
      // Ignore capture errors
    }
    const { w, h } = getStageSize();
    const { x, y, sz } = getCupCenter(w, h);
    const bowl = getBowlCenter(x, y, sz, state.rotDeg);

    pointerState = {
      pointerId: e.pointerId,
      dragPx: isTouch ? 12 : DRAG_PX,
      downX: p.x,
      downY: p.y,
      lastX: p.x,
      lastY: p.y,
      grabOffX: p.x - x,
      grabOffY: p.y - y,
      originBowlX: bowl.x,
      originBowlY: bowl.y,
      dragging: false,
      movedDist: 0,
      holdTimer: setTimeout(() => {
        if (!pointerState || pointerState.dragging) return;
        pointerState.dragging = true;
        state.targetLift = 1;
        setCupCursor('grabbing');
        kickAnim();
      }, HOLD_MS),
    };
    resetIdleTimer();
    return true;
  }

  function onPointerMove(e: PointerEvent, p: Point2D): boolean {
    if (pointerState) {
      if (e.pointerId !== pointerState.pointerId) return true;
      const dx = p.x - pointerState.lastX;
      const dy = p.y - pointerState.lastY;
      pointerState.lastX = p.x;
      pointerState.lastY = p.y;
      pointerState.movedDist += Math.hypot(dx, dy);

      if (!pointerState.dragging && pointerState.movedDist > pointerState.dragPx) {
        clearTimeout(pointerState.holdTimer);
        pointerState.dragging = true;
        state.targetLift = 1;
        setCupCursor('grabbing');
      }

      if (pointerState.dragging) {
        const { w, h } = getStageSize();
        const { x, y, sz } = getCupCenter(w, h);
        const newX = p.x - pointerState.grabOffX;
        const newY = p.y - pointerState.grabOffY;
        const minX = -sz * 0.04;
        const maxX = w + sz * 0.04;
        const minY = -sz * 0.04;
        const maxY = h + sz * 0.04;
        state.cupNX = Math.max(minX, Math.min(maxX, newX)) / Math.max(1, w);
        state.cupNY = Math.max(minY, Math.min(maxY, newY)) / Math.max(1, h);

        const rx = (p.x - x) / Math.max(1, sz * 0.35);
        const ry = (p.y - y) / Math.max(1, sz * 0.35);
        const torque = (rx * dy - ry * dx) * 0.18;
        const swivel = dx * 0.08 - dy * 0.03;
        const impulse = Math.max(-5.0, Math.min(5.0, torque + swivel));
        state.angVel = state.angVel * 0.72 + impulse * 0.48;
        state.rotDeg += state.angVel;
        kickAnim();
      }
      return true;
    }

    if (e.pointerType !== 'touch' && !deps.isCanvasBusy()) {
      const over = hitsCup(p.x, p.y);
      if (over !== hoveringCup) {
        hoveringCup = over;
        setCupCursor(over ? 'pointer' : '');
      }
      if (over) return true;
    }
    return false;
  }

  function finishPointer(cancelled = false, pointerId?: number): boolean {
    if (!pointerState) return false;
    if (pointerId !== undefined && pointerId !== pointerState.pointerId) return true;
    const ps = pointerState;
    pointerState = null;
    clearTimeout(ps.holdTimer);

    if (ps.dragging) {
      state.targetLift = 0;
      setCupCursor(hoveringCup ? 'pointer' : '');
      const { w, h } = getStageSize();
      const { x, y, sz } = getCupCenter(w, h);
      const bowl = getBowlCenter(x, y, sz, state.rotDeg);
      const moved = Math.hypot(bowl.x - ps.originBowlX, bowl.y - ps.originBowlY);
      if (!cancelled) {
        void deps.audio.playClip('cup_setdown', { channel: 'foley', volumeMul: 0.92 });
        if (moved > 18) {
          mugMoveCount++;
          const shouldRing = mugMoveCount >= 2 || (hasInteractedWithDrawing && Math.random() < 0.8);
          if (shouldRing) {
            addStain('ring', ps.originBowlX, ps.originBowlY, 0.98);
            const splashProb =
              mugMoveCount >= 3 ? 0.82 : mugMoveCount >= 2 ? 0.68 : hasInteractedWithDrawing ? 0.45 : 0;
            if (Math.random() < splashProb) {
              const pos = randomNapkinDripPos(w, h);
              const kind = Math.random() < 0.65 ? 'trail' : 'drip';
              addStain(kind, pos.x, pos.y, 1.08);
              scheduleTimer(() => {
                void deps.audio.playClip('oops_drip', { channel: 'vocal', interrupt: true });
                lastVocalAt = performance.now();
              }, 220);
            }
          }
        }
      }
      kickAnim();
    } else if (!cancelled && ps.movedDist <= ps.dragPx) {
      triggerDrink({ manual: true });
    }
    return true;
  }

  function resetIdleTimer(): void {
    if (idleTimer) {
      clearTimeout(idleTimer);
      idleTimer = null;
    }
    idleFiredThisStretch = false;
    const waitMs = 7500 + Math.random() * 5000;
    idleTimer = setTimeout(onIdleTimeout, waitMs);
  }

  function onIdleTimeout(): void {
    idleTimer = null;
    if (
      idleFiredThisStretch ||
      deps.getItemCount() < 1 ||
      deps.isTyping() ||
      deps.isCanvasBusy() ||
      state.phase !== 'idle'
    ) {
      return;
    }
    const now = performance.now();
    if (now - lastVocalAt < MIN_VOCAL_COOLDOWN_MS || deps.audio.isVocalBusy()) return;
    idleFiredThisStretch = true;

    const r = Math.random();
    if (r < 0.42) {
      const pick = Math.random();
      const clip: CompanionClipId = pick < 0.48 ? 'hmm_pause' : pick < 0.8 ? 'mm_ponder' : 'sigh_block';
      void deps.audio.playClip(clip, { channel: 'vocal' });
      lastVocalAt = now;
    } else if (r < 0.68 && state.hasDrink && !state.isDrinkDone && now - lastDrinkAt >= MIN_DRINK_COOLDOWN_MS) {
      triggerDrink({ manual: false });
    }
  }

  function markDrawingInteraction(): void {
    hasInteractedWithDrawing = true;
  }

  function onActivity(kind = 'sketch'): void {
    hasInteractedWithDrawing = true;
    totalActivity++;
    recSinceVocal++;
    recSinceDrink++;
    resetIdleTimer();

    if (totalActivity < 2) return;
    const now = performance.now();

    if (
      state.hasDrink &&
      !state.isDrinkDone &&
      state.phase === 'idle' &&
      recSinceDrink >= 3 &&
      now - lastDrinkAt >= MIN_DRINK_COOLDOWN_MS
    ) {
      const pDrink = Math.min(0.58, 0.22 + (recSinceDrink - 3) * 0.1);
      if (Math.random() < pDrink) {
        recSinceDrink = 0;
        const delay = 650 + Math.random() * 600;
        scheduleTimer(() => {
          if (!deps.isCanvasBusy() && !deps.isTyping() && state.hasDrink && !state.isDrinkDone && state.phase === 'idle') {
            triggerDrink({ manual: false });
          }
        }, delay);
        return;
      }
    }

    if (recSinceVocal >= 2 && now - lastVocalAt >= MIN_VOCAL_COOLDOWN_MS && !deps.audio.isVocalBusy()) {
      const pVocal = Math.min(0.72, 0.34 + (recSinceVocal - 2) * 0.12);
      if (Math.random() < pVocal) {
        recSinceVocal = 0;
        lastVocalAt = now;
        let clip: CompanionClipId;
        if (
          !hasEurekaOnSheet &&
          deps.getItemCount() >= 4 &&
          (kind === 'arrow' || kind === 'shape') &&
          Math.random() < 0.28
        ) {
          clip = 'eureka_win';
          hasEurekaOnSheet = true;
        } else {
          const r = Math.random();
          clip = r < 0.32 ? 'ooo_intrigued' : r < 0.64 ? 'aha_spark' : r < 0.86 ? 'mm_ponder' : 'hmm_pause';
        }
        const delay = 280 + Math.random() * 340;
        scheduleTimer(() => {
          if (!deps.audio.isVocalBusy()) void deps.audio.playClip(clip, { channel: 'vocal' });
        }, delay);
      }
    }
  }

  function onUndo(): void {
    const now = performance.now();
    undoTimes = undoTimes.filter(t => now - t < 6000);
    undoTimes.push(now);
    resetIdleTimer();

    if (undoTimes.length >= 2 && now - lastVocalAt >= 7500 && !deps.audio.isVocalBusy()) {
      if (Math.random() < 0.58) {
        lastVocalAt = now;
        const clip: CompanionClipId = Math.random() < 0.65 ? 'sigh_block' : 'hmm_pause';
        scheduleTimer(() => {
          void deps.audio.playClip(clip, { channel: 'vocal' });
        }, 240 + Math.random() * 180);
      }
    }
  }

  function liftForCrumple(): Promise<boolean | 'cancelled'> {
    clearPendingTimers();
    if (pointerState) finishPointer(true);
    state.refillProgress = null;
    if (!state.hasDrink) return Promise.resolve(false);
    if (state.phase === 'drinking' || state.phase === 'crumple_wait') {
      state.phase = 'crumple_wait';
      renderCupLayer();
      return Promise.resolve(true);
    }
    if (state.phase === 'crumple_lift') {
      return new Promise(resolve => crumpleLiftResolvers.push(resolve));
    }
    state.phase = 'crumple_lift';
    state.phaseStart = performance.now();
    state.setdownPlayed = false;
    kickAnim();
    return new Promise(resolve => crumpleLiftResolvers.push(resolve));
  }

  function resetSheetForCrumple(): void {
    state.stains = [];
    hasInteractedWithDrawing = false;
    mugMoveCount = 0;
    hasEurekaOnSheet = false;
    recSinceVocal = 0;
    undoTimes = [];
  }

  function returnAfterCrumple(): void {
    if (!state.hasDrink || state.phase !== 'crumple_wait') return;
    state.phase = 'crumple_return';
    state.phaseStart = performance.now();
    state.setdownPlayed = false;
    kickAnim();
  }

  function init(): void {
    preloadCups();
    const grid = document.getElementById('drink-menu-grid');
    if (grid) {
      grid.innerHTML = DRINK_ORDER.map(id => {
        const d = DRINKS[id];
        return `
          <button class="drink-opt" data-drink="${id}" commandfor="drink-menu" command="hide-popover">
            <img src="${getCupSpriteUrl(id, '100')}" alt="${d.name}" />
            <span class="do-name">${d.name}</span>
            <span class="do-sub">${d.sub}</span>
          </button>
        `;
      }).join('');
      grid.addEventListener('click', e => {
        const btn = (e.target as HTMLElement).closest<HTMLButtonElement>('button.drink-opt');
        const id = btn?.dataset.drink as DrinkId | undefined;
        if (id && id in DRINKS) orderDrink(id);
      });
    }
    syncMenuBtn();
    resetIdleTimer();

    const toNapkin = (e: PointerEvent): Point2D => {
      const r = deps.canvas.getBoundingClientRect();
      const vt = deps.getViewTransform ? deps.getViewTransform() : { x: 0, y: 0, scale: 1 };
      return {
        x: (e.clientX - r.left - vt.x) / vt.scale,
        y: (e.clientY - r.top - vt.y) / vt.scale,
      };
    };

    const activeTouches = new Set<number>();

    deps.stageEl.addEventListener(
      'pointerdown',
      e => {
        if (e.pointerType === 'touch') {
          activeTouches.add(e.pointerId);
          if (activeTouches.size >= 2 || deps.isMultiTouch?.()) {
            if (pointerState) finishPointer(true);
            return;
          }
        }
        if (e.pointerType === 'mouse' && e.button !== 0) return;
        const p = toNapkin(e);
        if (deps.isTyping() && deps.isPointInsideTypingTarget(p.x, p.y)) return;
        if (onPointerDown(e, p, deps.stageEl)) {
          if (deps.isTyping()) deps.commitTextBox();
          e.stopPropagation();
        }
      },
      true,
    );
    deps.stageEl.addEventListener(
      'pointermove',
      e => {
        if (e.pointerType === 'touch' && (activeTouches.size >= 2 || deps.isMultiTouch?.())) return;
        if (onPointerMove(e, toNapkin(e))) e.stopPropagation();
      },
      true,
    );
    deps.stageEl.addEventListener(
      'pointerup',
      e => {
        if (e.pointerType === 'touch') activeTouches.delete(e.pointerId);
        if (finishPointer(false, e.pointerId)) e.stopPropagation();
      },
      true,
    );
    deps.stageEl.addEventListener(
      'pointercancel',
      e => {
        if (e.pointerType === 'touch') activeTouches.delete(e.pointerId);
        finishPointer(true, e.pointerId);
      },
      true,
    );
    deps.stageEl.addEventListener('lostpointercapture', e => finishPointer(true, e.pointerId), false);
    deps.stageEl.addEventListener('pointerleave', () => {
      if (!pointerState && hoveringCup) {
        hoveringCup = false;
        setCupCursor('');
      }
    });
  }

  return {
    init,
    orderDrink,
    markDrawingInteraction,
    hitsCup,
    renderStainsTo,
    renderCupLayer,
    triggerDrink,
    triggerRefill,
    onActivity,
    onUndo,
    hasDrink: () => state.hasDrink,
    liftForCrumple,
    resetSheetForCrumple,
    returnAfterCrumple,
    getState: () => ({
      hasDrink: state.hasDrink,
      isDrinkDone: state.isDrinkDone,
      drinkId: state.drinkId,
      volume: Math.round(state.volume * 100),
      targetVolume: Math.round(state.targetVolume * 100),
      phase: state.phase,
      rotDeg: Math.round(state.rotDeg * 10) / 10,
      cupNX: Math.round(state.cupNX * 1000) / 1000,
      cupNY: Math.round(state.cupNY * 1000) / 1000,
      mugMoveCount,
      hasInteractedWithDrawing,
      stainCount: state.stains.length,
      loadedCups,
    }),
  };
}
