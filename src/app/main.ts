import './styles.css';

import { cachedFetchJSON, MODEL_URLS, PEN_IMAGES, PEN_TYPES } from './assets';
import { createNapkinAudio } from './audio';
import { createCrumpleRunner } from './crumple';
import { createNapkinCup } from './cup';
import { createHandwritingEngine, layoutWordsToStrokes } from './handwriting';
import {
  buildArrowItem,
  buildShapeItem,
  buildShapeStrokes,
  buildStrokeOutline,
  drawPStroke,
  drawRawPreview,
  recognizeMultiStroke,
  setVAEModel,
  strokeAlphaForPen,
} from './ink-vae';
import { buildNapkinSheetCanvas, drawNapkinBackground, getNapkinSpec, nextNapkin } from './napkin-paper';
import { pickStarterDiagram, STARTER_DIAGRAMS } from './starters';
import {
  DEFAULTS,
  STYLE_KEYS,
  WRITING_STYLES,
  type DragToolId,
  type InkStroke,
  type NapkinItem,
  type PenId,
  type Point2D,
  type ShapeItem,
  type ShapeType,
  type StrokePoint,
  type StyleKey,
  type TextItem,
  type TextOwner,
  type ToolId,
  type VAEModel,
  type WordCacheEntry,
} from './types';

// Conditional polyfills per Chrome Modern Web Guidance (only loaded when native support is missing)
if (!Object.hasOwn(HTMLElement.prototype, 'popover')) {
  const popoverPolyfillUrl = 'https://unpkg.com/@oddbird/popover-polyfill@latest/dist/popover.min.js';
  void import(/* @vite-ignore */ popoverPolyfillUrl);
}
if (!('commandForElement' in HTMLButtonElement.prototype)) {
  const invokersPolyfillUrl = 'https://esm.run/invokers-polyfill';
  void import(/* @vite-ignore */ invokersPolyfillUrl).catch(() => {
    // Offline fallback when CDN polyfill cannot be reached
    document.addEventListener('click', e => {
      const btn = (e.target as HTMLElement | null)?.closest<HTMLButtonElement>('button[commandfor][command]');
      if (!btn) return;
      const targetId = btn.getAttribute('commandfor');
      const cmd = btn.getAttribute('command');
      const el = targetId ? document.getElementById(targetId) : null;
      if (!el || typeof el.togglePopover !== 'function') return;
      if (cmd === 'toggle-popover') el.togglePopover();
      else if (cmd === 'show-popover') el.showPopover();
      else if (cmd === 'hide-popover') el.hidePopover();
    });
  });
}

const toolbarEl = document.querySelector('header.toolbar') as HTMLElement;
const canvas = document.getElementById('c') as HTMLCanvasElement;
const cupCanvas = document.getElementById('cup-layer') as HTMLCanvasElement;
const stageEl = document.getElementById('stage') as HTMLElement;
const wrapEl = document.getElementById('wrap') as HTMLElement;
const inlineInput = document.getElementById('inline-text') as HTMLInputElement;
const selActionsEl = document.getElementById('sel-actions') as HTMLElement | null;
const selEditBtn = document.getElementById('sel-edit-btn') as HTMLButtonElement | null;
const selDeleteBtn = document.getElementById('sel-delete-btn') as HTMLButtonElement | null;
const resetViewBtn = document.getElementById('reset-view-btn') as HTMLButtonElement | null;
const crumpleOverlay = document.getElementById('crumple-overlay') as HTMLCanvasElement;

let currentColor = '#1b3d8f';
let currentStyleKey: StyleKey = 'upright';
let currentPenId: PenId = 'ballpoint';

let tool: ToolId = 'sketch';
let items: NapkinItem[] = [];
let activeRaw: Point2D[] | null = null;
let pendingStrokes: Point2D[][] = [];
let pendingTimer: ReturnType<typeof setTimeout> | null = null;

interface DragShapeState {
  tool: DragToolId;
  x: number;
  y: number;
  cur: Point2D;
  color: string;
  penId: PenId;
  seed: number;
  cacheKey: string;
  previewStrokes: InkStroke[] | null;
}
let dragShape: DragShapeState | null = null;

interface ResizeSnapshot {
  x: number;
  y: number;
  w: number;
  h: number;
  x1?: number;
  y1?: number;
  x2?: number;
  y2?: number;
  anchorX?: number;
  anchorY?: number;
  strokes: StrokePoint[][];
  textStrokes: StrokePoint[][] | null;
  rawList: Point2D[][] | null;
}

type HandleId = 'nw' | 'ne' | 'sw' | 'se';
let selectedItem: NapkinItem | null = null;
let hoverItem: NapkinItem | null = null;
let dragItem: NapkinItem | null = null;
let dragOff: [number, number] = [0, 0];
let dragStartPos: [number, number] = [0, 0];
let resizeState: { item: NapkinItem; handle: HandleId; snap: ResizeSnapshot } | null = null;
let lastPointerType = 'mouse';
let activePointerId: number | null = null;
let pendingTextTap: { pointerId: number; x: number; y: number } | null = null;
let lastMoveTap: { time: number; x: number; y: number; item: NapkinItem } | null = null;

// Mobile two-finger pan & pinch-to-zoom camera (identity on desktop: scale=1, x=0, y=0)
let viewX = 0;
let viewY = 0;
let viewScale = 1;
const touchPoints = new Map<number, Point2D>();
let pinchState: {
  startMid: Point2D;
  startDist: number;
  startViewX: number;
  startViewY: number;
  startScale: number;
} | null = null;

interface TypingSnapshot {
  text: string;
  strokes: InkStroke[];
  cache: WordCacheEntry[];
  cacheStyle: StyleKey | undefined;
}

interface TypingTarget {
  x: number;
  y: number;
  shapeRef: ShapeItem | null;
  textItem: TextItem | null;
  wasNew: boolean;
  snapshot: TypingSnapshot | null;
  openedAt: number;
  lastLen: number;
}

let typingTarget: TypingTarget | null = null;
let caretVisible = true;
let caretTimer: ReturnType<typeof setInterval> | null = null;
let renderQueued = false;
let refocusingInlineInput = false;

const stylePreviewBufs: Partial<Record<StyleKey, number[]>> = {};

function scheduleRender(): void {
  if (renderQueued) return;
  renderQueued = true;
  requestAnimationFrame(() => {
    renderQueued = false;
    render();
  });
}

const audio = createNapkinAudio();
const handwriting = createHandwritingEngine(scheduleRender);

const cup = createNapkinCup({
  canvas,
  cupCanvas,
  stageEl,
  audio,
  getTool: () => tool,
  getItemCount: () => items.length,
  isCanvasBusy: () => !!(activeRaw || dragShape || dragItem || resizeState || pinchState),
  isTyping: () => !!typingTarget,
  isPointInsideTypingTarget,
  commitTextBox,
  restoreFocus,
  render: () => render(),
  scheduleRender,
  getViewTransform: () => ({ x: viewX, y: viewY, scale: viewScale }),
  isMultiTouch: () => touchPoints.size >= 2 || !!pinchState,
});

function restoreFocus(sel: string): void {
  requestAnimationFrame(() => {
    document.querySelector<HTMLElement>(sel)?.focus();
  });
}

function setColor(hex: string, btnEl?: HTMLElement | null): void {
  currentColor = hex;
  document.querySelectorAll('button.swatch').forEach(b => b.classList.remove('active'));
  if (btnEl) btnEl.classList.add('active');
  paintStylePreviews();
}

function setPen(penId: PenId): void {
  const pen = PEN_TYPES.find(p => p.id === penId) ?? PEN_TYPES[0];
  currentPenId = pen.id;
  audio.setPreset(pen.id);
  const img = document.getElementById('pen-btn-img') as HTMLImageElement | null;
  const lbl = document.getElementById('pen-btn-lbl');
  if (img) img.src = pen.img;
  if (lbl) lbl.textContent = pen.short;
  document.querySelectorAll<HTMLButtonElement>('button.pen-opt').forEach(b => {
    b.classList.toggle('active', b.dataset.pen === pen.id);
  });
  paintStylePreviews();
}

function setWritingStyle(key: StyleKey): void {
  currentStyleKey = key;
  document.querySelectorAll<HTMLButtonElement>('button.style-opt').forEach(b => {
    b.classList.toggle('active', b.dataset.style === key);
  });
}

function paintStylePreviews(): void {
  for (const key of STYLE_KEYS) {
    const cv = document.querySelector<HTMLCanvasElement>(`canvas.style-prev[data-style-prev="${key}"]`);
    if (!cv) continue;
    const dpr = 2;
    const w = 136;
    const h = 34;
    if (cv.width !== w * dpr || cv.height !== h * dpr) {
      cv.width = w * dpr;
      cv.height = h * dpr;
    }
    const ctx = cv.getContext('2d');
    if (!ctx) continue;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#ece1ce';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = 'rgba(110, 86, 56, 0.16)';
    ctx.lineWidth = 0.75;
    ctx.beginPath();
    ctx.moveTo(8, h - 7.5);
    ctx.lineTo(w - 8, h - 7.5);
    ctx.stroke();

    const buf = stylePreviewBufs[key];
    if (buf && buf.length >= 6) {
      const { strokes } = layoutWordsToStrokes([buf], w * 0.5, h * 0.48, 20, w - 18, currentColor, 1.45, 'center', currentPenId);
      for (const s of strokes) drawPStroke(ctx, s.pts, s.baseW, s.color, s.alpha, currentPenId);
    } else {
      ctx.fillStyle = 'rgba(110, 100, 87, 0.45)';
      ctx.font = 'italic 11px Georgia, serif';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText('Napkin', w * 0.5, h * 0.5);
    }
  }
}

async function generateStylePreviews(): Promise<void> {
  await Promise.all(
    STYLE_KEYS.map(key => {
      const dummy: TextOwner = {};
      return handwriting
        .synthesizeCursiveStreaming('Napkin', 68, 17, 20, 118, currentColor, null, dummy, key, 'center', currentPenId)
        .then(() => {
          const buf = dummy.wordCache?.[0]?.buf;
          if (buf) {
            stylePreviewBufs[key] = buf;
            paintStylePreviews();
          }
        });
    }),
  );
}

function initPenAndStyleMenu(): void {
  const penBtnImg = document.getElementById('pen-btn-img') as HTMLImageElement | null;
  if (penBtnImg) penBtnImg.src = PEN_IMAGES[currentPenId];

  const penList = document.getElementById('pen-list');
  if (penList) {
    penList.innerHTML = PEN_TYPES.map(
      p => `
      <button class="pen-opt${p.id === currentPenId ? ' active' : ''}" data-pen="${p.id}">
        <img src="${p.img}" alt="${p.name}" />
        <span><span class="po-name">${p.name}</span><span class="po-hint">${p.hint}</span></span>
      </button>
    `,
    ).join('');
    penList.addEventListener('click', e => {
      const btn = (e.target as HTMLElement).closest<HTMLButtonElement>('button.pen-opt');
      const id = btn?.dataset.pen as PenId | undefined;
      if (id) setPen(id);
    });
  }

  const styleGrid = document.getElementById('style-grid');
  if (styleGrid) {
    styleGrid.innerHTML = STYLE_KEYS.map(
      k => `
      <button class="style-opt${k === currentStyleKey ? ' active' : ''}" data-style="${k}">
        <canvas class="style-prev" data-style-prev="${k}" width="272" height="68" aria-hidden="true"></canvas>
        <span class="so-lbl">${WRITING_STYLES[k].label}</span>
      </button>
    `,
    ).join('');
    styleGrid.addEventListener('click', e => {
      const btn = (e.target as HTMLElement).closest<HTMLButtonElement>('button.style-opt');
      const key = btn?.dataset.style as StyleKey | undefined;
      if (key && key in WRITING_STYLES) setWritingStyle(key);
    });
    paintStylePreviews();
  }
}

function flushPendingNow(): void {
  if (pendingTimer) {
    clearTimeout(pendingTimer);
    pendingTimer = null;
  }
  if (!pendingStrokes.length) return;
  const batch = pendingStrokes;
  pendingStrokes = [];
  const created = recognizeMultiStroke(batch, currentColor, currentPenId, currentStyleKey);
  items.push(created);
  cup.onActivity(created.kind || 'sketch');
  scheduleRender();
}

function setTool(t: ToolId): void {
  flushPendingNow();
  tool = t;
  hoverItem = null;
  if (t !== 'move') selectedItem = null;
  document.querySelectorAll('button.tool-btn[id^="t-"]').forEach(b => b.classList.remove('active'));
  document.getElementById('t-' + t)?.classList.add('active');
  canvas.style.cursor = t === 'move' ? 'default' : t === 'text' ? 'text' : 'crosshair';
  scheduleRender();
}

function getResizeHandles(it: NapkinItem): Array<{ id: HandleId; x: number; y: number; cursor: string }> {
  const pad = 5;
  return [
    { id: 'nw', x: it.x - pad, y: it.y - pad, cursor: 'nwse-resize' },
    { id: 'ne', x: it.x + it.w + pad, y: it.y - pad, cursor: 'nesw-resize' },
    { id: 'sw', x: it.x - pad, y: it.y + it.h + pad, cursor: 'nesw-resize' },
    { id: 'se', x: it.x + it.w + pad, y: it.y + it.h + pad, cursor: 'nwse-resize' },
  ];
}

function screenToWorld(sx: number, sy: number): Point2D {
  return { x: (sx - viewX) / viewScale, y: (sy - viewY) / viewScale };
}

function worldToScreen(wx: number, wy: number): Point2D {
  return { x: wx * viewScale + viewX, y: wy * viewScale + viewY };
}

function clampView(nx: number, ny: number, ns: number, w: number, h: number): void {
  const s = Math.max(0.55, Math.min(2.5, ns));
  const scaledW = w * s;
  const scaledH = h * s;
  const minX = Math.min(w * 0.65 - scaledW, w - scaledW - 24);
  const maxX = Math.max(w * 0.35, 24);
  const minY = Math.min(h * 0.65 - scaledH, h - scaledH - 24);
  const maxY = Math.max(h * 0.35, 24);
  viewScale = s;
  viewX = Math.max(minX, Math.min(maxX, nx));
  viewY = Math.max(minY, Math.min(maxY, ny));
  updateResetViewBtn();
}

function updateResetViewBtn(): void {
  if (!resetViewBtn) return;
  const isDefault = Math.abs(viewScale - 1) <= 0.02 && Math.abs(viewX) <= 4 && Math.abs(viewY) <= 4;
  resetViewBtn.hidden = isDefault;
  if (!isDefault) {
    resetViewBtn.textContent = Math.abs(viewScale - 1) > 0.02 ? `${Math.round(viewScale * 100)}%` : 'Reset';
  }
}

function resetView(): void {
  viewX = 0;
  viewY = 0;
  viewScale = 1;
  updateResetViewBtn();
  scheduleRender();
}

function hitResizeHandle(it: NapkinItem | null, px: number, py: number, isTouch = false) {
  if (!it) return null;
  const tol = (isTouch ? 22 : 7) / viewScale;
  for (const h of getResizeHandles(it)) {
    if (Math.abs(px - h.x) <= tol && Math.abs(py - h.y) <= tol) return h;
  }
  return null;
}

function syncToolbarBottom(): void {
  if (!toolbarEl) return;
  const bottom = Math.round(toolbarEl.getBoundingClientRect().bottom + 8);
  document.documentElement.style.setProperty('--toolbar-bottom', `${bottom}px`);
}

function setupFlexiblePopoverFallbacks(): void {
  if (typeof CSS !== 'undefined' && CSS.supports('anchor-name', '--a') && CSS.supports('position-area', 'block-end')) {
    return;
  }
  const pairs: Array<[string, string]> = [
    ['pen-menu', 'pen-btn'],
    ['drink-menu', 'drink-menu-btn'],
    ['save-menu', 'save-btn'],
  ];
  const positionPopover = (pop: HTMLElement, btn: HTMLElement) => {
    const ar = btn.getBoundingClientRect();
    const vw = window.innerWidth;
    const vh = window.innerHeight;
    const pad = 8;
    const pw = Math.min(pop.offsetWidth || 300, vw - pad * 2);
    const ph = pop.offsetHeight || 240;

    // Flexible inline direction: open left (right-aligned) -> flip right (left-aligned) -> clamp inside viewport
    let left = ar.right - pw;
    if (left < pad) {
      const rightOpen = ar.left;
      left = rightOpen + pw <= vw - pad ? rightOpen : Math.max(pad, Math.min(vw - pw - pad, (vw - pw) * 0.5));
    }

    // Flexible block direction: open below -> flip above if clipped below
    let top = ar.bottom + 6;
    if (top + ph > vh - pad && ar.top - ph - 6 >= pad) {
      top = ar.top - ph - 6;
    } else {
      top = Math.max(pad, Math.min(vh - Math.min(ph, vh - pad * 2) - pad, top));
    }

    pop.style.left = `${Math.round(left)}px`;
    pop.style.right = 'auto';
    pop.style.top = `${Math.round(top)}px`;
  };

  for (const [popId, btnId] of pairs) {
    const pop = document.getElementById(popId);
    const btn = document.getElementById(btnId);
    if (!pop || !btn) continue;
    pop.addEventListener('toggle', e => {
      if ((e as ToggleEvent).newState === 'open') {
        requestAnimationFrame(() => positionPopover(pop, btn));
      }
    });
  }
}

function updateSelectionActions(): void {
  if (!selActionsEl) return;
  const show =
    lastPointerType === 'touch' &&
    tool === 'move' &&
    !!selectedItem &&
    items.includes(selectedItem) &&
    !dragItem &&
    !resizeState &&
    !pinchState &&
    !typingTarget;
  if (!show || !selectedItem) {
    selActionsEl.hidden = true;
    return;
  }
  const canEdit = selectedItem.kind === 'shape' || selectedItem.kind === 'text';
  if (selEditBtn) selEditBtn.hidden = !canEdit;
  selActionsEl.hidden = false;

  const r = canvas.getBoundingClientRect();
  const barW = selActionsEl.offsetWidth || (canEdit ? 124 : 74);
  const barH = selActionsEl.offsetHeight || 42;
  const topLeft = worldToScreen(selectedItem.x, selectedItem.y);
  const botRight = worldToScreen(selectedItem.x + selectedItem.w, selectedItem.y + selectedItem.h);
  const cx = (topLeft.x + botRight.x) * 0.5;
  const left = Math.max(8, Math.min(r.width - barW - 8, cx - barW * 0.5));
  const above = topLeft.y - barH - 14;
  const top = above >= 8 ? above : Math.min(r.height - barH - 8, botRight.y + 14);
  selActionsEl.style.left = `${Math.round(left)}px`;
  selActionsEl.style.top = `${Math.round(Math.max(8, top))}px`;
}

function snapshotItemGeometry(it: NapkinItem): ResizeSnapshot {
  return {
    x: it.x,
    y: it.y,
    w: it.w,
    h: it.h,
    ...(it.kind === 'arrow' ? { x1: it.x1, y1: it.y1, x2: it.x2, y2: it.y2 } : {}),
    ...(it.kind === 'text' ? { anchorX: it.anchorX, anchorY: it.anchorY } : {}),
    strokes: it.strokes.map(s => s.pts.map(p => [p[0], p[1], p[2]] as StrokePoint)),
    textStrokes: it.kind === 'shape' ? it.textStrokes.map(s => s.pts.map(p => [p[0], p[1], p[2]] as StrokePoint)) : null,
    rawList: it.kind === 'handwrite' ? it.rawList.map(r => r.map(p => ({ x: p.x, y: p.y }))) : null,
  };
}

function applyLiveResize(it: NapkinItem, snap: ResizeSnapshot, newX: number, newY: number, newW: number, newH: number): void {
  const sx = newW / Math.max(1, snap.w);
  const sy = newH / Math.max(1, snap.h);
  const mapX = (x: number) => newX + (x - snap.x) * sx;
  const mapY = (y: number) => newY + (y - snap.y) * sy;

  it.x = newX;
  it.y = newY;
  it.w = newW;
  it.h = newH;
  it.strokes.forEach((s, si) => {
    const orig = snap.strokes[si]!;
    for (let i = 0; i < orig.length; i++) {
      s.pts[i]![0] = mapX(orig[i]![0]);
      s.pts[i]![1] = mapY(orig[i]![1]);
    }
  });
  if (it.kind === 'shape' && snap.textStrokes) {
    it.textStrokes.forEach((s, si) => {
      const orig = snap.textStrokes![si]!;
      for (let i = 0; i < orig.length; i++) {
        s.pts[i]![0] = mapX(orig[i]![0]);
        s.pts[i]![1] = mapY(orig[i]![1]);
      }
    });
  }
  if (it.kind === 'handwrite' && snap.rawList) {
    it.rawList.forEach((r, ri) => {
      const orig = snap.rawList![ri]!;
      for (let i = 0; i < orig.length; i++) {
        r[i]!.x = mapX(orig[i]!.x);
        r[i]!.y = mapY(orig[i]!.y);
      }
    });
  }
  if (it.kind === 'text' && snap.anchorX !== undefined && snap.anchorY !== undefined) {
    it.anchorX = mapX(snap.anchorX);
    it.anchorY = mapY(snap.anchorY);
  }
  if (it.kind === 'arrow' && snap.x1 !== undefined && snap.y1 !== undefined && snap.x2 !== undefined && snap.y2 !== undefined) {
    it.x1 = mapX(snap.x1);
    it.y1 = mapY(snap.y1);
    it.x2 = mapX(snap.x2);
    it.y2 = mapY(snap.y2);
  }
}

function finalizeResize(it: NapkinItem): void {
  if (it.kind === 'shape') {
    it.strokes = buildShapeStrokes(it.shapeType, it.x, it.y, it.w, it.h, it.color, it.penId);
    if (it.label) {
      const maxW = it.shapeType === 'diamond' ? it.w * 0.62 : it.w * 0.84;
      const targetH = Math.min(34, Math.max(16, it.h * 0.48));
      void handwriting.synthesizeCursiveStreaming(
        it.label,
        it.x + it.w / 2,
        it.y + it.h / 2,
        targetH,
        maxW,
        it.color,
        it,
        null,
        it.styleKey || currentStyleKey,
        'center',
        it.penId,
      );
    }
  } else if (it.kind === 'arrow') {
    const fresh = buildArrowItem(it.x1, it.y1, it.x2, it.y2, it.color, it.penId);
    it.strokes = fresh.strokes;
    it.x = fresh.x;
    it.y = fresh.y;
    it.w = fresh.w;
    it.h = fresh.h;
  }
}

function render(opts: { skipCup?: boolean } = {}): void {
  const r = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const cw = Math.round(r.width * dpr);
  const ch = Math.round(r.height * dpr);
  if (canvas.width !== cw || canvas.height !== ch) {
    canvas.width = cw;
    canvas.height = ch;
  }
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const isTransformed = Math.abs(viewScale - 1) > 0.001 || Math.abs(viewX) > 0.5 || Math.abs(viewY) > 0.5;
  if (isTransformed) {
    ctx.fillStyle = '#c8b69b';
    ctx.fillRect(0, 0, r.width, r.height);
  }

  ctx.save();
  if (isTransformed) {
    ctx.translate(viewX, viewY);
    ctx.scale(viewScale, viewScale);
    ctx.save();
    ctx.shadowColor = 'rgba(55, 40, 22, 0.22)';
    ctx.shadowBlur = 18;
    ctx.shadowOffsetY = 6;
    ctx.fillStyle = getNapkinSpec().baseHex;
    ctx.fillRect(0, 0, r.width, r.height);
    ctx.restore();
  }

  drawNapkinBackground(ctx, r.width, r.height);
  cup.renderStainsTo(ctx);

  for (const it of items) {
    for (const s of it.strokes) drawPStroke(ctx, s.pts, s.baseW, s.color, s.alpha, s.penId || it.penId);
    if (it.kind === 'shape') {
      for (const s of it.textStrokes) drawPStroke(ctx, s.pts, s.baseW, s.color, s.alpha, s.penId || it.penId);
    }
  }

  if (typingTarget) {
    const val = inlineInput.value || '';
    const len = val.length;
    const selStart = Math.min(len, inlineInput.selectionStart ?? len);
    const selEnd = Math.min(len, inlineInput.selectionEnd ?? len);
    const trailingSpaces = (val.match(/\s+$/) ?? [''])[0].length;

    let minX = typingTarget.x;
    let maxX = typingTarget.x;
    let cx = typingTarget.x;
    let cy = typingTarget.y;
    let chH = 26;
    let col = currentColor;

    if (typingTarget.shapeRef) {
      const sh = typingTarget.shapeRef;
      col = sh.color;
      cy = sh.y + sh.h / 2;
      chH = Math.min(26, Math.max(16, sh.h * 0.48));

      ctx.save();
      ctx.strokeStyle = 'rgba(35, 32, 28, 0.30)';
      ctx.lineWidth = 1.1;
      ctx.setLineDash([4, 3]);
      ctx.strokeRect(sh.x - 4, sh.y - 4, sh.w + 8, sh.h + 8);
      ctx.restore();

      if (sh.textStrokes.length) {
        minX = 1e9;
        maxX = -1e9;
        for (const s of sh.textStrokes) {
          for (const pt of s.pts) {
            if (pt[0] < minX) minX = pt[0];
            if (pt[0] > maxX) maxX = pt[0];
          }
        }
        const effMaxX = maxX + trailingSpaces * 10;
        cx = selEnd >= len ? effMaxX + 5 : minX + (effMaxX - minX) * (selEnd / Math.max(1, len));
      } else {
        minX = sh.x + sh.w / 2;
        maxX = minX + trailingSpaces * 10;
        cx = maxX;
        if (len === 0) {
          ctx.save();
          ctx.fillStyle = 'rgba(110, 100, 87, 0.42)';
          ctx.font = 'italic 12px Georgia, "Times New Roman", serif';
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText('Type label...', sh.x + sh.w / 2, cy);
          ctx.restore();
        }
      }
    } else if (typingTarget.textItem) {
      const ti = typingTarget.textItem;
      col = ti.color;
      cy = ti.anchorY;
      chH = 28;
      minX = ti.anchorX;
      const hasInk = ti.strokes.length > 0;
      const effMaxX = hasInk ? ti.x + ti.w + trailingSpaces * 12 : ti.anchorX + trailingSpaces * 12;
      maxX = effMaxX;
      cx = selEnd >= len ? (hasInk ? effMaxX + 5 : effMaxX) : minX + (effMaxX - minX) * (selEnd / Math.max(1, len));

      const lineEnd = Math.max(ti.anchorX + 116, effMaxX + 18);
      ctx.save();
      ctx.strokeStyle = 'rgba(92, 74, 52, 0.28)';
      ctx.lineWidth = 1.1;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(ti.anchorX - 3, cy + chH * 0.56);
      ctx.lineTo(lineEnd, cy + chH * 0.56);
      ctx.stroke();
      ctx.restore();

      if (len === 0) {
        ctx.save();
        ctx.fillStyle = 'rgba(110, 100, 87, 0.44)';
        ctx.font = 'italic 14px Georgia, "Times New Roman", serif';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.fillText('Type on napkin...', ti.anchorX + 7, cy);
        ctx.restore();
      }
    }

    if (selStart !== selEnd && len > 0 && maxX > minX) {
      const sx0 = minX + (maxX - minX) * (selStart / len);
      const sx1 = minX + (maxX - minX) * (selEnd / len);
      ctx.save();
      ctx.fillStyle = col;
      ctx.globalAlpha = 0.14;
      ctx.fillRect(sx0 - 2, cy - chH * 0.55, Math.max(6, sx1 - sx0 + 4), chH * 1.1);
      ctx.restore();
    }

    const caretScr = worldToScreen(cx, cy - chH * 0.5);
    inlineInput.style.left = Math.max(4, Math.min(r.width - 8, caretScr.x)) + 'px';
    inlineInput.style.top = Math.max(4, Math.min(r.height - 30, caretScr.y)) + 'px';

    if (caretVisible) {
      ctx.save();
      ctx.strokeStyle = col;
      ctx.globalAlpha = 0.78;
      ctx.lineWidth = 1.7;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(cx, cy - chH * 0.5);
      ctx.lineTo(cx, cy + chH * 0.5);
      ctx.stroke();
      ctx.restore();
    }
  }

  if (tool === 'move' && hoverItem && hoverItem !== selectedItem && !dragItem && !resizeState) {
    ctx.save();
    ctx.strokeStyle = 'rgba(35, 32, 28, 0.25)';
    ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(hoverItem.x - 5, hoverItem.y - 5, hoverItem.w + 10, hoverItem.h + 10);
    ctx.restore();
  }
  if (selectedItem && items.includes(selectedItem)) {
    ctx.save();
    ctx.strokeStyle = 'rgba(35, 32, 28, 0.58)';
    ctx.lineWidth = 1.1;
    ctx.setLineDash([4, 3]);
    ctx.strokeRect(selectedItem.x - 5, selectedItem.y - 5, selectedItem.w + 10, selectedItem.h + 10);
    ctx.setLineDash([]);
    const hs = (lastPointerType === 'touch' ? 10 : 7) / viewScale;
    const hh = hs * 0.5;
    for (const h of getResizeHandles(selectedItem)) {
      ctx.fillStyle = '#f8f3ea';
      ctx.strokeStyle = '#23201c';
      ctx.lineWidth = 1.4 / viewScale;
      ctx.fillRect(h.x - hh, h.y - hh, hs, hs);
      ctx.strokeRect(h.x - hh, h.y - hh, hs, hs);
    }
    ctx.restore();
  }

  for (const ps of pendingStrokes) drawRawPreview(ctx, ps, currentColor, currentPenId);
  if (activeRaw) drawRawPreview(ctx, activeRaw, currentColor, currentPenId);

  if (dragShape && dragShape.cur) {
    const x1 = dragShape.x;
    const y1 = dragShape.y;
    const x2 = dragShape.cur.x;
    const y2 = dragShape.cur.y;
    const dist = Math.hypot(x2 - x1, y2 - y1);
    if (dist >= 4) {
      const key = `${Math.round(x2 * 2)},${Math.round(y2 * 2)}`;
      if (dragShape.cacheKey !== key || !dragShape.previewStrokes) {
        dragShape.cacheKey = key;
        if (dragShape.tool === 'arrow') {
          dragShape.previewStrokes = buildArrowItem(x1, y1, x2, y2, dragShape.color, dragShape.penId, dragShape.seed).strokes;
        } else {
          const x = Math.min(x1, x2);
          const y = Math.min(y1, y2);
          const w = Math.max(4, Math.abs(x2 - x1));
          const h = Math.max(4, Math.abs(y2 - y1));
          dragShape.previewStrokes = buildShapeStrokes(dragShape.tool, x, y, w, h, dragShape.color, dragShape.penId, dragShape.seed);
        }
      }
      for (const s of dragShape.previewStrokes) {
        drawPStroke(ctx, s.pts, s.baseW, s.color, s.alpha, s.penId);
      }
    }
  }

  ctx.restore();
  updateSelectionActions();

  if (!opts.skipCup) {
    cup.renderCupLayer();
  }
}

function stopCaretBlink(): void {
  if (caretTimer) {
    clearInterval(caretTimer);
    caretTimer = null;
  }
}

function openTextBox(
  x: number,
  y: number,
  shapeRef: ShapeItem | null = null,
  existingTextItem: TextItem | null = null,
  opts: { deferFocus?: boolean } = {},
): void {
  flushPendingNow();
  stopCaretBlink();
  selectedItem = null;
  let textItem = existingTextItem;
  let wasNew = false;
  let snapshot: TypingSnapshot | null = null;

  if (shapeRef) {
    snapshot = {
      text: shapeRef.label || '',
      strokes: shapeRef.textStrokes.slice(),
      cache: shapeRef.wordCache ? shapeRef.wordCache.slice() : [],
      cacheStyle: shapeRef.wordCacheStyle,
    };
  } else if (existingTextItem) {
    snapshot = {
      text: existingTextItem.text || '',
      strokes: existingTextItem.strokes.slice(),
      cache: existingTextItem.wordCache ? existingTextItem.wordCache.slice() : [],
      cacheStyle: existingTextItem.wordCacheStyle,
    };
  } else {
    wasNew = true;
    textItem = {
      kind: 'text',
      text: '',
      styleKey: currentStyleKey,
      penId: currentPenId,
      anchorX: x,
      anchorY: y,
      x,
      y: y - 16,
      w: 20,
      h: 32,
      color: currentColor,
      strokes: [],
      wordCache: [],
      wordCacheStyle: currentStyleKey,
    };
    items.push(textItem);
  }

  typingTarget = { x, y, shapeRef, textItem, wasNew, snapshot, openedAt: performance.now(), lastLen: 0 };

  const r = canvas.getBoundingClientRect();
  const anchorX = shapeRef ? shapeRef.x + shapeRef.w / 2 : textItem!.anchorX;
  const anchorY = shapeRef ? shapeRef.y + shapeRef.h / 2 : textItem!.anchorY;
  const anchorScr = worldToScreen(anchorX, anchorY - 14);
  inlineInput.style.left = Math.max(4, Math.min(r.width - 40, anchorScr.x)) + 'px';
  inlineInput.style.top = Math.max(4, Math.min(r.height - 34, anchorScr.y)) + 'px';
  inlineInput.classList.add('active');
  inlineInput.tabIndex = 0;
  inlineInput.removeAttribute('aria-hidden');
  inlineInput.value = shapeRef ? shapeRef.label || '' : textItem!.text || '';
  typingTarget.lastLen = inlineInput.value.length;

  if (!opts.deferFocus) {
    inlineInput.focus({ preventScroll: true });
    try {
      inlineInput.setSelectionRange(inlineInput.value.length, inlineInput.value.length);
    } catch {
      // Ignore selection errors on unsupported inputs
    }
    requestAnimationFrame(() => {
      if (!typingTarget) return;
      if (document.activeElement !== inlineInput) {
        inlineInput.focus({ preventScroll: true });
      }
      try {
        inlineInput.setSelectionRange(inlineInput.value.length, inlineInput.value.length);
      } catch {
        // Ignore selection errors
      }
    });
  }

  caretVisible = true;
  caretTimer = setInterval(() => {
    caretVisible = !caretVisible;
    scheduleRender();
  }, 530);
  scheduleRender();
}

function onLiveTextInput(): void {
  if (!typingTarget) return;
  caretVisible = true;
  const val = inlineInput.value;
  if (val.length > typingTarget.lastLen && val.trim().length > 0) {
    audio.triggerTypingPulse();
  }
  typingTarget.lastLen = val.length;
  if (typingTarget.shapeRef) {
    const it = typingTarget.shapeRef;
    it.label = val;
    it.styleKey = currentStyleKey;
    const maxW = it.shapeType === 'diamond' ? it.w * 0.62 : it.w * 0.84;
    const targetH = Math.min(30, Math.max(18, it.h * 0.48));
    void handwriting.synthesizeCursiveStreaming(
      val,
      it.x + it.w / 2,
      it.y + it.h / 2,
      targetH,
      maxW,
      it.color,
      it,
      null,
      currentStyleKey,
      'center',
      it.penId,
    );
  } else if (typingTarget.textItem) {
    const ti = typingTarget.textItem;
    ti.text = val;
    ti.styleKey = currentStyleKey;
    const r = canvas.getBoundingClientRect();
    const freeMaxW = Math.max(120, Math.min(640, r.width - ti.anchorX - 16));
    void handwriting.synthesizeCursiveStreaming(
      val,
      ti.anchorX,
      ti.anchorY,
      32,
      freeMaxW,
      ti.color,
      null,
      ti,
      currentStyleKey,
      'left',
      ti.penId,
    );
  }
  scheduleRender();
}

function deactivateInlineInput(): void {
  inlineInput.classList.remove('active');
  inlineInput.tabIndex = -1;
  inlineInput.setAttribute('aria-hidden', 'true');
  inlineInput.blur();
}

function commitTextBox(): void {
  if (!typingTarget) return;
  const tgt = typingTarget;
  typingTarget = null;
  stopCaretBlink();
  const val = inlineInput.value.trim();
  deactivateInlineInput();

  if (!val) {
    if (tgt.shapeRef) {
      tgt.shapeRef.label = '';
      tgt.shapeRef.textStrokes = [];
      tgt.shapeRef.wordCache = [];
    } else if (tgt.textItem) {
      const idx = items.indexOf(tgt.textItem);
      if (idx !== -1) items.splice(idx, 1);
    }
  } else if (tgt.shapeRef) {
    tgt.shapeRef.label = val;
    cup.onActivity('text');
  } else if (tgt.textItem) {
    tgt.textItem.text = val;
    cup.onActivity('text');
  }
  scheduleRender();
}

function cancelTextBox(): void {
  if (!typingTarget) return;
  const tgt = typingTarget;
  typingTarget = null;
  stopCaretBlink();
  deactivateInlineInput();

  if (tgt.wasNew && tgt.textItem) {
    tgt.textItem.latestReqId = handwriting.nextReqId();
    const idx = items.indexOf(tgt.textItem);
    if (idx !== -1) items.splice(idx, 1);
  } else if (tgt.snapshot) {
    const owner: TextOwner | null = tgt.shapeRef ?? tgt.textItem;
    if (owner) owner.latestReqId = handwriting.nextReqId();
    if (tgt.shapeRef) {
      tgt.shapeRef.label = tgt.snapshot.text;
      tgt.shapeRef.textStrokes = tgt.snapshot.strokes;
    } else if (tgt.textItem) {
      tgt.textItem.text = tgt.snapshot.text;
      tgt.textItem.strokes = tgt.snapshot.strokes;
    }
    if (owner) {
      owner.wordCache = tgt.snapshot.cache;
      owner.wordCacheStyle = tgt.snapshot.cacheStyle;
    }
  }
  scheduleRender();
}

function isPointInsideTypingTarget(x: number, y: number): boolean {
  if (!typingTarget) return false;
  if (typingTarget.shapeRef) {
    const sh = typingTarget.shapeRef;
    return x >= sh.x - 6 && x <= sh.x + sh.w + 6 && y >= sh.y - 6 && y <= sh.y + sh.h + 6;
  }
  if (typingTarget.textItem) {
    const ti = typingTarget.textItem;
    const w = Math.max(120, ti.w + 24);
    return x >= ti.anchorX - 8 && x <= ti.anchorX + w && y >= ti.anchorY - 22 && y <= ti.anchorY + 22;
  }
  return false;
}

const clearAllWithCrumple = createCrumpleRunner({
  canvas,
  wrapEl,
  overlay: crumpleOverlay,
  audio,
  beforeCrumple() {
    if (typingTarget) commitTextBox();
    if (pendingTimer) {
      clearTimeout(pendingTimer);
      pendingTimer = null;
    }
    pendingStrokes = [];
    activeRaw = null;
    dragShape = null;
    dragItem = null;
    resizeState = null;
    selectedItem = null;
    hoverItem = null;
    audio.stopDraw();
  },
  hasContent: () => items.length > 0,
  renderForSnapshot: () => render({ skipCup: true }),
  clearAndGenerateFreshSheet() {
    items = [];
    viewX = 0;
    viewY = 0;
    viewScale = 1;
    updateResetViewBtn();
    cup.resetSheetForCrumple();
    nextNapkin();
    render();
  },
  liftCupForCrumple: () => cup.liftForCrumple(),
  returnCupAfterCrumple: () => cup.returnAfterCrumple(),
});

function undo(): void {
  if (pendingStrokes.length) {
    pendingStrokes.pop();
    if (!pendingStrokes.length && pendingTimer) {
      clearTimeout(pendingTimer);
      pendingTimer = null;
    }
  } else {
    const removed = items.pop();
    if (removed === selectedItem) selectedItem = null;
  }
  cup.onUndo();
  scheduleRender();
}

function triggerDownload(url: string, filename: string): void {
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  if (url.startsWith('blob:')) setTimeout(() => URL.revokeObjectURL(url), 4000);
}

function getCleanExportBg(w: number, h: number, scale = 2): HTMLCanvasElement {
  const sp = getNapkinSpec();
  const bg = buildNapkinSheetCanvas(sp, w, h, scale);
  const bctx = bg.getContext('2d');
  if (bctx) {
    bctx.setTransform(scale, 0, 0, scale, 0, 0);
    cup.renderStainsTo(bctx, performance.now() + 5000);
  }
  return bg;
}

function exportPNG(): void {
  if (typingTarget) commitTextBox();
  flushPendingNow();
  const r = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width));
  const h = Math.max(1, Math.round(r.height));
  const scale = 2;

  const out = document.createElement('canvas');
  out.width = w * scale;
  out.height = h * scale;
  const octx = out.getContext('2d');
  if (!octx) return;
  octx.drawImage(getCleanExportBg(w, h, scale), 0, 0);

  octx.setTransform(scale, 0, 0, scale, 0, 0);
  for (const it of items) {
    for (const s of it.strokes) drawPStroke(octx, s.pts, s.baseW, s.color, s.alpha, s.penId || it.penId);
    if (it.kind === 'shape') {
      for (const s of it.textStrokes) drawPStroke(octx, s.pts, s.baseW, s.color, s.alpha, s.penId || it.penId);
    }
  }

  if (out.toBlob) {
    out.toBlob(blob => {
      if (blob) triggerDownload(URL.createObjectURL(blob), 'napkin-sketch.png');
    }, 'image/png');
  } else {
    triggerDownload(out.toDataURL('image/png'), 'napkin-sketch.png');
  }
}

function exportSVG(): void {
  if (typingTarget) commitTextBox();
  flushPendingNow();
  const r = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.round(r.width));
  const h = Math.max(1, Math.round(r.height));
  const bgHex = getNapkinSpec().baseHex;

  const bgCanvas = getCleanExportBg(w, h, 2);
  const bgDataUrl = bgCanvas.toDataURL('image/jpeg', 0.93);

  const inkLines: string[] = [];
  const emit = (s: InkStroke, fallbackPen: PenId) => {
    const pid = s.penId || fallbackPen;
    const outline = buildStrokeOutline(s.pts, s.baseW, pid);
    if (!outline) return;
    const op = strokeAlphaForPen(pid, s.alpha).toFixed(2);
    if (outline.dot) {
      inkLines.push(
        `<circle cx="${outline.x.toFixed(1)}" cy="${outline.y.toFixed(1)}" r="${outline.r.toFixed(2)}" fill="${s.color}" fill-opacity="${op}"/>`,
      );
    } else {
      const d = 'M' + outline.poly.map(p => `${p[0].toFixed(1)},${p[1].toFixed(1)}`).join('L') + 'Z';
      inkLines.push(`<path d="${d}" fill="${s.color}" fill-opacity="${op}" fill-rule="nonzero"/>`);
    }
  };
  for (const it of items) {
    it.strokes.forEach(s => emit(s, it.penId));
    if (it.kind === 'shape') it.textStrokes.forEach(s => emit(s, it.penId));
  }

  const svg = [
    `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="${w}" height="${h}" viewBox="0 0 ${w} ${h}">`,
    `<rect width="${w}" height="${h}" fill="${bgHex}"/>`,
    `<image width="${w}" height="${h}" xlink:href="${bgDataUrl}" preserveAspectRatio="none"/>`,
    `<g style="mix-blend-mode:multiply">${inkLines.join('')}</g>`,
    `</svg>`,
  ].join('');

  triggerDownload(URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml;charset=utf-8' })), 'napkin-sketch.svg');
}

async function loadStarterScene(forcedIdx: number | null = null): Promise<string> {
  const r = canvas.getBoundingClientRect();
  const diagram =
    forcedIdx != null && STARTER_DIAGRAMS[forcedIdx] ? STARTER_DIAGRAMS[forcedIdx]! : pickStarterDiagram();

  const marginX = r.width < 520 ? 16 : 36;
  const marginY = r.height < 420 ? 16 : 32;
  const availW = Math.max(140, r.width - marginX * 2);
  const availH = Math.max(140, r.height - marginY * 2);
  const scale = Math.max(0.35, Math.min(1, availW / diagram.w, availH / diagram.h));

  const scaledW = diagram.w * scale;
  const scaledH = diagram.h * scale;
  const ox =
    scale < 1
      ? Math.max(marginX, (r.width - scaledW) * 0.5)
      : Math.max(36, Math.min(r.width - diagram.w - 36, (r.width - diagram.w) * 0.44));
  const oy =
    scale < 1
      ? Math.max(marginY, (r.height - scaledH) * 0.44)
      : Math.max(32, Math.min(r.height - diagram.h - 32, (r.height - diagram.h) * 0.46));

  const sx = (v: number) => ox + (v - ox) * scale;
  const sy = (v: number) => oy + (v - oy) * scale;

  items = [];
  const synthJobs: Array<() => Promise<void>> = [];
  diagram.build(ox, oy, {
    shape(type: ShapeType, x: number, y: number, w: number, h: number, label: string, color = '#1d3557') {
      const rx = sx(x);
      const ry = sy(y);
      const rw = w * scale;
      const rh = h * scale;
      const sh = buildShapeItem(type, rx, ry, rw, rh, label, color, currentPenId, currentStyleKey);
      items.push(sh);
      if (label) {
        const th = Math.max(15, Math.round((type === 'diamond' ? 25 : 27) * scale));
        const maxW = type === 'diamond' ? rw * 0.64 : rw * 0.84;
        synthJobs.push(() =>
          handwriting.synthesizeCursiveStreaming(
            label,
            sh.x + sh.w / 2,
            sh.y + sh.h / 2,
            th,
            maxW,
            sh.color,
            sh,
            null,
            currentStyleKey,
            'center',
            currentPenId,
          ),
        );
      }
      return sh;
    },
    arrow(x1: number, y1: number, x2: number, y2: number, color = '#1d3557') {
      const ar = buildArrowItem(sx(x1), sy(y1), sx(x2), sy(y2), color, currentPenId);
      items.push(ar);
      return ar;
    },
    note(cx: number, cy: number, text: string, targetH = 23, maxW = 200, color = '#1d3557') {
      const rcx = sx(cx);
      const rcy = sy(cy);
      const rth = Math.max(14, Math.round(targetH * scale));
      const rMaxW = maxW * scale;
      const ti: TextItem = {
        kind: 'text',
        text,
        styleKey: currentStyleKey,
        penId: currentPenId,
        anchorX: rcx,
        anchorY: rcy,
        x: rcx - rMaxW * 0.4,
        y: rcy - rth * 0.6,
        w: rMaxW * 0.8,
        h: rth * 1.2,
        color,
        strokes: [],
        wordCache: [],
        wordCacheStyle: currentStyleKey,
      };
      items.push(ti);
      synthJobs.push(() =>
        handwriting.synthesizeCursiveStreaming(
          text,
          rcx,
          rcy,
          rth,
          rMaxW,
          color,
          null,
          ti,
          currentStyleKey,
          'center',
          currentPenId,
        ),
      );
      return ti;
    },
  });

  scheduleRender();
  await Promise.all(synthJobs.map(fn => fn()));
  return diagram.name;
}

function getScreenPos(e: PointerEvent | MouseEvent): Point2D {
  const r = canvas.getBoundingClientRect();
  return { x: e.clientX - r.left, y: e.clientY - r.top };
}

function getPos(e: PointerEvent | MouseEvent): Point2D {
  const s = getScreenPos(e);
  return screenToWorld(s.x, s.y);
}

function hitItem(x: number, y: number, isTouch = false): NapkinItem | null {
  const pad = (isTouch ? 18 : 8) / viewScale;
  for (let i = items.length - 1; i >= 0; i--) {
    const it = items[i]!;
    if (x >= it.x - pad && x <= it.x + it.w + pad && y >= it.y - pad && y <= it.y + it.h + pad) return it;
  }
  return null;
}

function cancelSingleTouchInteraction(): void {
  activePointerId = null;
  pendingTextTap = null;
  activeRaw = null;
  dragShape = null;
  dragItem = null;
  resizeState = null;
  audio.stopDraw();
}

function startTwoFingerPanZoom(): void {
  const pts = Array.from(touchPoints.values());
  if (pts.length < 2) return;
  const a = pts[0]!;
  const b = pts[1]!;
  pinchState = {
    startMid: { x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5 },
    startDist: Math.max(20, Math.hypot(b.x - a.x, b.y - a.y)),
    startViewX: viewX,
    startViewY: viewY,
    startScale: viewScale,
  };
}

function deleteSelectedItem(): void {
  if (!selectedItem) return;
  const idx = items.indexOf(selectedItem);
  if (idx !== -1) items.splice(idx, 1);
  selectedItem = null;
  hoverItem = null;
  scheduleRender();
}

// ---- Wire DOM Listeners ----
inlineInput.addEventListener('input', onLiveTextInput);
inlineInput.addEventListener('compositionupdate', onLiveTextInput);
inlineInput.addEventListener('keydown', e => {
  if (e.key === 'Enter') {
    e.preventDefault();
    commitTextBox();
    return;
  }
  if (e.key === 'Escape') {
    e.preventDefault();
    cancelTextBox();
    return;
  }
  caretVisible = true;
  scheduleRender();
});
inlineInput.addEventListener('keyup', () => {
  if (typingTarget) {
    caretVisible = true;
    scheduleRender();
  }
});
inlineInput.addEventListener('select', () => {
  if (typingTarget) scheduleRender();
});
inlineInput.addEventListener('blur', () => {
  if (refocusingInlineInput) return;
  if (typingTarget && performance.now() - typingTarget.openedAt < 350) {
    inlineInput.focus({ preventScroll: true });
    return;
  }
  commitTextBox();
});

// Focus inline input synchronously inside native `click` on mobile so iOS Safari / Android Chrome summon the soft keyboard
canvas.addEventListener('mousedown', e => {
  if (typingTarget) e.preventDefault();
});
canvas.addEventListener('click', () => {
  if (!typingTarget) return;
  if (document.activeElement === inlineInput && lastPointerType === 'touch') {
    refocusingInlineInput = true;
    inlineInput.blur();
    refocusingInlineInput = false;
  }
  inlineInput.focus({ preventScroll: true });
  try {
    inlineInput.setSelectionRange(inlineInput.value.length, inlineInput.value.length);
  } catch {
    // Ignore selection errors
  }
});

canvas.addEventListener('pointerdown', e => {
  if (e.pointerType === 'mouse' && e.button !== 0) return;
  const isTouch = e.pointerType === 'touch';
  lastPointerType = e.pointerType || 'mouse';

  if (isTouch) {
    touchPoints.set(e.pointerId, getScreenPos(e));
    if (touchPoints.size >= 2) {
      if (typingTarget) commitTextBox();
      cancelSingleTouchInteraction();
      startTwoFingerPanZoom();
      try {
        canvas.setPointerCapture(e.pointerId);
      } catch {
        // Ignore capture errors
      }
      scheduleRender();
      return;
    }
  }

  const p = getPos(e);

  if (typingTarget) {
    if (isPointInsideTypingTarget(p.x, p.y)) {
      if (!isTouch) e.preventDefault();
      inlineInput.focus({ preventScroll: true });
      caretVisible = true;
      scheduleRender();
      return;
    }
    commitTextBox();
    if (tool !== 'text') return;
  }

  if (tool === 'text') {
    if (isTouch) {
      pendingTextTap = { pointerId: e.pointerId, x: p.x, y: p.y };
      return;
    }
    e.preventDefault();
    const hit = hitItem(p.x, p.y, false);
    if (hit && hit.kind === 'shape') openTextBox(hit.x + hit.w / 2, hit.y + hit.h / 2, hit, null);
    else if (hit && hit.kind === 'text') openTextBox(hit.anchorX, hit.anchorY, null, hit);
    else openTextBox(p.x, p.y, null, null);
    return;
  }

  if (activePointerId !== null && activePointerId !== e.pointerId) return;
  activePointerId = e.pointerId;
  try {
    canvas.setPointerCapture(e.pointerId);
  } catch {
    // Ignore capture errors
  }

  if (tool === 'move') {
    if (selectedItem) {
      const h = hitResizeHandle(selectedItem, p.x, p.y, isTouch);
      if (h) {
        resizeState = { item: selectedItem, handle: h.id, snap: snapshotItemGeometry(selectedItem) };
        canvas.style.cursor = h.cursor;
        scheduleRender();
        return;
      }
    }
    const hit = hitItem(p.x, p.y, isTouch);
    selectedItem = hit;
    if (hit) {
      dragItem = hit;
      dragOff = [p.x, p.y];
      dragStartPos = [p.x, p.y];
      canvas.style.cursor = 'grabbing';
    }
    scheduleRender();
    return;
  }

  if (tool === 'box' || tool === 'oval' || tool === 'diamond' || tool === 'arrow') {
    flushPendingNow();
    dragShape = {
      tool,
      x: p.x,
      y: p.y,
      cur: p,
      color: currentColor,
      penId: currentPenId,
      seed: (Math.floor(Math.random() * 0xffffffff) ^ (Date.now() & 0xffffffff)) >>> 0,
      cacheKey: '',
      previewStrokes: null,
    };
    audio.startDraw(p.x, p.y);
    scheduleRender();
    return;
  }

  if (pendingTimer) {
    clearTimeout(pendingTimer);
    pendingTimer = null;
  }
  activeRaw = [p];
  audio.startDraw(p.x, p.y);
  scheduleRender();
});

canvas.addEventListener('pointermove', e => {
  if (e.pointerType === 'touch' && touchPoints.has(e.pointerId)) {
    touchPoints.set(e.pointerId, getScreenPos(e));
    if (pinchState && touchPoints.size >= 2) {
      const pts = Array.from(touchPoints.values());
      const a = pts[0]!;
      const b = pts[1]!;
      const curMid = { x: (a.x + b.x) * 0.5, y: (a.y + b.y) * 0.5 };
      const curDist = Math.max(20, Math.hypot(b.x - a.x, b.y - a.y));
      const newScale = Math.max(0.55, Math.min(2.5, pinchState.startScale * (curDist / pinchState.startDist)));
      const wx = (pinchState.startMid.x - pinchState.startViewX) / pinchState.startScale;
      const wy = (pinchState.startMid.y - pinchState.startViewY) / pinchState.startScale;
      const r = canvas.getBoundingClientRect();
      clampView(curMid.x - wx * newScale, curMid.y - wy * newScale, newScale, r.width, r.height);
      scheduleRender();
      return;
    }
  }
  if (pinchState) return;

  const p = getPos(e);

  if (pendingTextTap && e.pointerId === pendingTextTap.pointerId) {
    if (Math.hypot(p.x - pendingTextTap.x, p.y - pendingTextTap.y) > 14) {
      pendingTextTap = null;
    }
    return;
  }

  if (activePointerId !== null && e.pointerId !== activePointerId) return;

  if (resizeState) {
    const { item, handle, snap } = resizeState;
    const minW = 28;
    const minH = 22;
    let x = snap.x;
    let y = snap.y;
    let w = snap.w;
    let h = snap.h;
    const right = snap.x + snap.w;
    const bottom = snap.y + snap.h;
    if (handle.includes('e')) w = Math.max(minW, p.x - snap.x);
    if (handle.includes('s')) h = Math.max(minH, p.y - snap.y);
    if (handle.includes('w')) {
      x = Math.min(right - minW, p.x);
      w = right - x;
    }
    if (handle.includes('n')) {
      y = Math.min(bottom - minH, p.y);
      h = bottom - y;
    }
    applyLiveResize(item, snap, x, y, w, h);
    scheduleRender();
    return;
  }

  if (dragItem) {
    const dx = p.x - dragOff[0];
    const dy = p.y - dragOff[1];
    dragOff = [p.x, p.y];
    dragItem.x += dx;
    dragItem.y += dy;
    if (dragItem.kind === 'text') {
      dragItem.anchorX += dx;
      dragItem.anchorY += dy;
    }
    if (dragItem.kind === 'arrow') {
      dragItem.x1 += dx;
      dragItem.y1 += dy;
      dragItem.x2 += dx;
      dragItem.y2 += dy;
    }
    const shift = (s: InkStroke) => {
      for (const pt of s.pts) {
        pt[0] += dx;
        pt[1] += dy;
      }
    };
    dragItem.strokes.forEach(shift);
    if (dragItem.kind === 'shape') dragItem.textStrokes.forEach(shift);
    if (dragItem.kind === 'handwrite') {
      for (const r of dragItem.rawList) for (const pt of r) { pt.x += dx; pt.y += dy; }
    }
    scheduleRender();
    return;
  }

  if (tool === 'move') {
    if (e.pointerType === 'touch') return;
    const rh = hitResizeHandle(selectedItem, p.x, p.y, false);
    if (rh) {
      canvas.style.cursor = rh.cursor;
      return;
    }
    const h = hitItem(p.x, p.y, false);
    canvas.style.cursor = h ? 'grab' : 'default';
    if (h !== hoverItem) {
      hoverItem = h;
      scheduleRender();
    }
    return;
  }

  if (dragShape) {
    dragShape.cur = p;
    audio.updateDraw(p.x, p.y);
    scheduleRender();
    return;
  }
  if (activeRaw) {
    activeRaw.push(p);
    audio.updateDraw(p.x, p.y);
    scheduleRender();
  }
});

canvas.addEventListener('pointerup', e => {
  const isTouch = e.pointerType === 'touch';
  if (isTouch) {
    touchPoints.delete(e.pointerId);
    if (pinchState) {
      if (touchPoints.size < 2) {
        pinchState = null;
        activePointerId = null;
      } else {
        startTwoFingerPanZoom();
      }
      scheduleRender();
      return;
    }
  }
  const p = getPos(e);

  if (pendingTextTap && e.pointerId === pendingTextTap.pointerId) {
    const tap = pendingTextTap;
    pendingTextTap = null;
    if (Math.hypot(p.x - tap.x, p.y - tap.y) <= 14 / viewScale) {
      const hit = hitItem(p.x, p.y, true);
      if (hit && hit.kind === 'shape') openTextBox(hit.x + hit.w / 2, hit.y + hit.h / 2, hit, null, { deferFocus: true });
      else if (hit && hit.kind === 'text') openTextBox(hit.anchorX, hit.anchorY, null, hit, { deferFocus: true });
      else openTextBox(p.x, p.y, null, null, { deferFocus: true });
    }
    return;
  }

  if (activePointerId !== null && e.pointerId !== activePointerId) return;
  activePointerId = null;
  if (canvas.hasPointerCapture(e.pointerId)) {
    try {
      canvas.releasePointerCapture(e.pointerId);
    } catch {
      // Ignore release errors
    }
  }
  audio.stopDraw();

  if (resizeState) {
    const it = resizeState.item;
    resizeState = null;
    finalizeResize(it);
    canvas.style.cursor = 'default';
    scheduleRender();
    return;
  }

  if (dragItem) {
    const movedIt = dragItem;
    dragItem = null;
    canvas.style.cursor = 'grab';
    if (isTouch && Math.hypot(p.x - dragStartPos[0], p.y - dragStartPos[1]) < 12) {
      const now = performance.now();
      if (
        lastMoveTap &&
        lastMoveTap.item === movedIt &&
        now - lastMoveTap.time < 360 &&
        Math.hypot(p.x - lastMoveTap.x, p.y - lastMoveTap.y) < 32
      ) {
        lastMoveTap = null;
        if (movedIt.kind === 'shape') openTextBox(movedIt.x + movedIt.w / 2, movedIt.y + movedIt.h / 2, movedIt, null, { deferFocus: true });
        else if (movedIt.kind === 'text') openTextBox(movedIt.anchorX, movedIt.anchorY, null, movedIt, { deferFocus: true });
        return;
      }
      lastMoveTap = { time: now, x: p.x, y: p.y, item: movedIt };
    } else {
      lastMoveTap = null;
    }
    scheduleRender();
    return;
  }

  const tapThresh = isTouch ? 12 : 5;

  if (dragShape && dragShape.cur) {
    const ds = dragShape;
    const x1 = ds.x;
    const y1 = ds.y;
    const x2 = ds.cur.x;
    const y2 = ds.cur.y;
    dragShape = null;
    const dist = Math.hypot(x2 - x1, y2 - y1);
    if (dist < tapThresh) {
      const hit = hitItem(p.x, p.y, isTouch);
      if (hit && hit.kind === 'shape') openTextBox(hit.x + hit.w / 2, hit.y + hit.h / 2, hit, null, { deferFocus: isTouch });
      else if (hit && hit.kind === 'text') openTextBox(hit.anchorX, hit.anchorY, null, hit, { deferFocus: isTouch });
      else openTextBox(p.x, p.y, null, null, { deferFocus: isTouch });
    } else if (ds.tool === 'arrow' && dist > 16) {
      items.push(buildArrowItem(x1, y1, x2, y2, ds.color, ds.penId, ds.seed));
      cup.onActivity('arrow');
    } else if (ds.tool !== 'arrow' && Math.abs(x2 - x1) > 18 && Math.abs(y2 - y1) > 14) {
      items.push(
        buildShapeItem(
          ds.tool,
          Math.min(x1, x2),
          Math.min(y1, y2),
          Math.abs(x2 - x1),
          Math.abs(y2 - y1),
          '',
          ds.color,
          ds.penId,
          currentStyleKey,
          ds.seed,
        ),
      );
      cup.onActivity('shape');
    }
    scheduleRender();
    return;
  }

  if (activeRaw) {
    const pts = activeRaw;
    activeRaw = null;
    let arc = 0;
    for (let i = 1; i < pts.length; i++) {
      arc += Math.hypot(pts[i]!.x - pts[i - 1]!.x, pts[i]!.y - pts[i - 1]!.y);
    }

    const tapThreshWorld = tapThresh / viewScale;
    if (!pendingStrokes.length && arc < tapThreshWorld) {
      const hit = hitItem(p.x, p.y, isTouch);
      if (hit && hit.kind === 'shape') openTextBox(hit.x + hit.w / 2, hit.y + hit.h / 2, hit, null, { deferFocus: isTouch });
      else if (hit && hit.kind === 'text') openTextBox(hit.anchorX, hit.anchorY, null, hit, { deferFocus: isTouch });
      else openTextBox(p.x, p.y, null, null, { deferFocus: isTouch });
      scheduleRender();
      return;
    }

    if (pts.length >= 2 || pendingStrokes.length > 0) {
      const first = pts[0]!;
      pendingStrokes.push(pts.length >= 2 ? pts : [first, { x: first.x + 0.8, y: first.y + 0.8 }]);
      pendingTimer = setTimeout(flushPendingNow, DEFAULTS.debounceMs);
    }
    scheduleRender();
  }
});

canvas.addEventListener('pointercancel', e => {
  if (e.pointerType === 'touch') {
    touchPoints.delete(e.pointerId);
    if (pinchState && touchPoints.size < 2) pinchState = null;
  }
  if (pendingTextTap && e.pointerId === pendingTextTap.pointerId) pendingTextTap = null;
  if (activePointerId !== null && e.pointerId !== activePointerId) return;
  cancelSingleTouchInteraction();
  scheduleRender();
});
canvas.addEventListener('lostpointercapture', e => {
  if (activePointerId === e.pointerId) activePointerId = null;
  audio.stopDraw();
});

canvas.addEventListener('dblclick', e => {
  if (tool !== 'move') return;
  const p = getPos(e);
  const hit = hitItem(p.x, p.y);
  if (hit && hit.kind === 'shape') openTextBox(hit.x + hit.w / 2, hit.y + hit.h / 2, hit, null);
  else if (hit && hit.kind === 'text') openTextBox(hit.anchorX, hit.anchorY, null, hit);
});

// Toolbar button bindings
document.querySelectorAll<HTMLButtonElement>('button.tool-btn[data-tool]').forEach(btn => {
  btn.addEventListener('click', () => {
    const t = btn.dataset.tool as ToolId | undefined;
    if (t) setTool(t);
  });
});

document.querySelectorAll<HTMLButtonElement>('button.swatch[data-color]').forEach(btn => {
  btn.addEventListener('click', () => {
    const hex = btn.dataset.color;
    if (hex) setColor(hex, btn);
  });
});

document.getElementById('sound-toggle')?.addEventListener('click', () => audio.toggleMute());
document.getElementById('undo-btn')?.addEventListener('click', () => undo());
document.getElementById('clear-btn')?.addEventListener('click', () => void clearAllWithCrumple());
document.getElementById('save-btn')?.addEventListener('click', () => syncToolbarBottom());
document.getElementById('pen-btn')?.addEventListener('click', () => syncToolbarBottom());
document.getElementById('drink-menu-btn')?.addEventListener('click', () => syncToolbarBottom());
document.getElementById('export-png-btn')?.addEventListener('click', () => {
  exportPNG();
  restoreFocus('#save-btn');
});
document.getElementById('export-svg-btn')?.addEventListener('click', () => {
  exportSVG();
  restoreFocus('#save-btn');
});
document.getElementById('pen-close-btn')?.addEventListener('click', () => restoreFocus('#pen-btn'));
document.getElementById('drink-close-btn')?.addEventListener('click', () => restoreFocus('#drink-menu-btn'));

selActionsEl?.addEventListener('pointerdown', e => e.stopPropagation());
selEditBtn?.addEventListener('click', () => {
  if (!selectedItem) return;
  const it = selectedItem;
  if (it.kind === 'shape') openTextBox(it.x + it.w / 2, it.y + it.h / 2, it, null);
  else if (it.kind === 'text') openTextBox(it.anchorX, it.anchorY, null, it);
});
selDeleteBtn?.addEventListener('click', () => deleteSelectedItem());

resetViewBtn?.addEventListener('pointerdown', e => e.stopPropagation());
resetViewBtn?.addEventListener('click', () => resetView());

window.addEventListener('keydown', e => {
  if (typingTarget) return;
  const target = e.target as HTMLElement | null;
  if (target && /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName)) return;
  if ((e.key === 'Backspace' || e.key === 'Delete') && selectedItem) {
    e.preventDefault();
    deleteSelectedItem();
    return;
  }
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z') {
    e.preventDefault();
    undo();
    return;
  }
  const k = e.key.toLowerCase();
  if (k === 's') setTool('sketch');
  else if (k === 't') setTool('text');
  else if (k === 'b') setTool('box');
  else if (k === 'o') setTool('oval');
  else if (k === 'd') setTool('diamond');
  else if (k === 'a') setTool('arrow');
  else if (k === 'v' || k === 'm') setTool('move');
});

window.addEventListener('resize', () => {
  syncToolbarBottom();
  scheduleRender();
});
window.visualViewport?.addEventListener('resize', () => {
  syncToolbarBottom();
  scheduleRender();
});
if (typeof ResizeObserver !== 'undefined') {
  const ro = new ResizeObserver(() => {
    syncToolbarBottom();
    scheduleRender();
  });
  ro.observe(wrapEl);
  if (toolbarEl) ro.observe(toolbarEl);
}

// Bootstrap app
syncToolbarBottom();
setupFlexiblePopoverFallbacks();
initPenAndStyleMenu();
audio.initUI();
cup.init();
scheduleRender();

void cachedFetchJSON<VAEModel>(MODEL_URLS.vae).then(model => {
  setVAEModel(model);
  void loadStarterScene();
});
void handwriting.initWasmPool().then(() => generateStylePreviews());
