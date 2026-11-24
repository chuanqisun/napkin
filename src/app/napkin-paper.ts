import { clampByte, makeRng, rgbToHex } from './rng';

type EmbossStyle = 'dimple' | 'linen' | 'diamond' | 'pinpoint';
type BorderStyle = 'none' | 'double' | 'dotted' | 'stitch';

interface NapkinPalette {
  readonly name: string;
  readonly r: number;
  readonly g: number;
  readonly b: number;
  readonly fiberMul: number;
  readonly fleckMul: number;
}

interface DiagCrease {
  corner: number;
  span: number;
  skew: number;
  alpha: number;
  flip: 1 | -1;
}

interface GhostCupRing {
  nx: number;
  ny: number;
  radius: number;
  alpha: number;
  rot: number;
}

export interface NapkinSpec {
  id: number;
  seed: number;
  paletteName: string;
  baseHex: string;
  embossStyle: EmbossStyle;
  embossStep: number;
  embossAlpha: number;
  borderStyle: BorderStyle;
  borderInset: number;
  fiberMul: number;
  fleckMul: number;
  tileOffX: number;
  tileOffY: number;
  foldVPos: number;
  foldVAngle: number;
  foldVWidth: number;
  foldVStrength: number;
  foldVFlip: boolean;
  foldHPos: number;
  foldHAngle: number;
  foldHWidth: number;
  foldHStrength: number;
  foldHFlip: boolean;
  lightX: number;
  lightY: number;
  vigAlpha: number;
  diagCreases: DiagCrease[];
  cupRing: GhostCupRing | null;
}

const NAPKIN_PALETTES = [
  { name: 'Warm Kraft', r: 231, g: 219, b: 198, fiberMul: 1.05, fleckMul: 1.15 },
  { name: 'Cafe Ecru', r: 236, g: 227, b: 210, fiberMul: 0.9, fleckMul: 0.8 },
  { name: 'Toasted Oat', r: 226, g: 211, b: 185, fiberMul: 1.25, fleckMul: 1.35 },
  { name: 'Soft Linen', r: 234, g: 224, b: 205, fiberMul: 0.95, fleckMul: 0.95 },
  { name: 'Diner Ivory', r: 240, g: 232, b: 217, fiberMul: 0.78, fleckMul: 0.65 },
  { name: 'Raw Sand', r: 229, g: 215, b: 191, fiberMul: 1.18, fleckMul: 1.25 },
] as const satisfies readonly NapkinPalette[];

const EMBOSS_STYLES: readonly EmbossStyle[] = ['dimple', 'linen', 'diamond', 'pinpoint'];
const BORDER_STYLES: readonly BorderStyle[] = ['none', 'double', 'dotted', 'stitch'];

let napkinSeq = 0;
let lastPaletteIdx = -1;
let lastEmbossIdx = -1;
let currentSpec: NapkinSpec | null = null;
let bgCache: { id: number; cw: number; ch: number; canvas: HTMLCanvasElement } | null = null;

export function getNapkinSpec(): NapkinSpec {
  return currentSpec ?? nextNapkin();
}

export function createNapkinSpec(): NapkinSpec {
  const seed =
    (Math.floor(Math.random() * 0xffffffff) ^
      (Date.now() & 0xffffffff) ^
      (++napkinSeq * 2654435761)) >>>
    0;
  const rnd = makeRng(seed);

  let pIdx = Math.floor(rnd() * NAPKIN_PALETTES.length);
  if (pIdx === lastPaletteIdx) {
    pIdx = (pIdx + 1 + Math.floor(rnd() * (NAPKIN_PALETTES.length - 1))) % NAPKIN_PALETTES.length;
  }
  lastPaletteIdx = pIdx;
  const pal = NAPKIN_PALETTES[pIdx]!;

  let eIdx = Math.floor(rnd() * EMBOSS_STYLES.length);
  if (eIdx === lastEmbossIdx) {
    eIdx = (eIdx + 1 + Math.floor(rnd() * (EMBOSS_STYLES.length - 1))) % EMBOSS_STYLES.length;
  }
  lastEmbossIdx = eIdx;

  const lumDrift = (rnd() - 0.5) * 7;
  const warmDrift = (rnd() - 0.5) * 4;
  const r = clampByte(pal.r + lumDrift + warmDrift);
  const g = clampByte(pal.g + lumDrift);
  const b = clampByte(pal.b + lumDrift - warmDrift);

  const hasCupRing = rnd() < 0.35;
  const ringCorner = Math.floor(rnd() * 4);
  const ringX = ringCorner % 2 === 0 ? 0.06 + rnd() * 0.14 : 0.8 + rnd() * 0.14;
  const ringY = ringCorner < 2 ? 0.08 + rnd() * 0.16 : 0.76 + rnd() * 0.16;

  const diagCount = 1 + (rnd() < 0.55 ? 1 : 0);
  const diagCreases: DiagCrease[] = [];
  for (let i = 0; i < diagCount; i++) {
    diagCreases.push({
      corner: Math.floor(rnd() * 4),
      span: 0.16 + rnd() * 0.22,
      skew: (rnd() - 0.5) * 0.12,
      alpha: 0.022 + rnd() * 0.026,
      flip: rnd() > 0.5 ? 1 : -1,
    });
  }

  return {
    id: napkinSeq,
    seed,
    paletteName: pal.name,
    baseHex: rgbToHex(r, g, b),
    embossStyle: EMBOSS_STYLES[eIdx]!,
    embossStep: eIdx === 3 ? 6 : rnd() < 0.5 ? 8 : 10,
    embossAlpha: 0.72 + rnd() * 0.52,
    borderStyle: BORDER_STYLES[Math.floor(rnd() * BORDER_STYLES.length)]!,
    borderInset: 15 + Math.floor(rnd() * 13),
    fiberMul: pal.fiberMul * (0.85 + rnd() * 0.35),
    fleckMul: pal.fleckMul * (0.8 + rnd() * 0.45),
    tileOffX: Math.floor(rnd() * 320),
    tileOffY: Math.floor(rnd() * 320),
    foldVPos: 0.465 + rnd() * 0.07,
    foldVAngle: (rnd() - 0.5) * 0.024,
    foldVWidth: 26 + rnd() * 18,
    foldVStrength: 0.032 + rnd() * 0.028,
    foldVFlip: rnd() > 0.5,
    foldHPos: 0.465 + rnd() * 0.07,
    foldHAngle: (rnd() - 0.5) * 0.022,
    foldHWidth: 24 + rnd() * 16,
    foldHStrength: 0.028 + rnd() * 0.026,
    foldHFlip: rnd() > 0.5,
    lightX: 0.42 + rnd() * 0.16,
    lightY: 0.4 + rnd() * 0.18,
    vigAlpha: 0.075 + rnd() * 0.045,
    diagCreases,
    cupRing: hasCupRing
      ? {
          nx: ringX,
          ny: ringY,
          radius: 40 + rnd() * 26,
          alpha: 0.022 + rnd() * 0.022,
          rot: rnd() * Math.PI * 2,
        }
      : null,
  };
}

export function nextNapkin(): NapkinSpec {
  currentSpec = createNapkinSpec();
  bgCache = null;
  const wrapEl = document.getElementById('wrap');
  if (wrapEl) wrapEl.style.background = currentSpec.baseHex;
  return currentSpec;
}

let trailerNextNapkinQueue: Array<{
  paletteIdx: number;
  embossStyle: EmbossStyle;
  borderStyle: BorderStyle;
  cupRing: GhostCupRing | null;
}> = [];

export function queueTrailerNapkin(spec: {
  paletteIdx: number;
  embossStyle: EmbossStyle;
  borderStyle: BorderStyle;
  cupRing: GhostCupRing | null;
}): void {
  trailerNextNapkinQueue.push(spec);
}

export function setTrailerNapkin(opts: {
  paletteIdx: number;
  embossStyle: EmbossStyle;
  borderStyle: BorderStyle;
  cupRing: GhostCupRing | null;
}): NapkinSpec {
  const sp = createNapkinSpec();
  const pal = NAPKIN_PALETTES[opts.paletteIdx % NAPKIN_PALETTES.length]!;
  sp.paletteName = pal.name;
  sp.baseHex = rgbToHex(pal.r, pal.g, pal.b);
  sp.embossStyle = opts.embossStyle;
  sp.borderStyle = opts.borderStyle;
  sp.fiberMul = pal.fiberMul;
  sp.fleckMul = pal.fleckMul;
  sp.cupRing = opts.cupRing;
  currentSpec = sp;
  bgCache = null;
  const wrapEl = document.getElementById('wrap');
  if (wrapEl) wrapEl.style.background = currentSpec.baseHex;
  return sp;
}

export function nextNapkinOrTrailer(): NapkinSpec {
  const queued = trailerNextNapkinQueue.shift();
  if (queued) return setTrailerNapkin(queued);
  return nextNapkin();
}

function createNapkinPattern(ctx: CanvasRenderingContext2D, sp: NapkinSpec): CanvasPattern | null {
  const S = 320;
  const tile = document.createElement('canvas');
  tile.width = S;
  tile.height = S;
  const tctx = tile.getContext('2d');
  if (!tctx) return null;

  tctx.fillStyle = sp.baseHex;
  tctx.fillRect(0, 0, S, S);

  const ea = sp.embossAlpha;
  if (sp.embossStyle === 'dimple') {
    const step = sp.embossStep;
    for (let y = 0; y < S; y += step) {
      const rowShift = ((y / step) % 2) * (step / 2);
      for (let x = 0; x < S; x += step) {
        const px = (x + rowShift) % S;
        tctx.fillStyle = `rgba(112, 86, 56, ${(0.042 * ea).toFixed(4)})`;
        tctx.fillRect(px, y, 2, 2);
        tctx.fillStyle = `rgba(255, 250, 240, ${(0.062 * ea).toFixed(4)})`;
        tctx.fillRect(px + 1, y + 1, 2, 2);
      }
    }
  } else if (sp.embossStyle === 'linen') {
    const step = 6;
    tctx.lineWidth = 0.8;
    for (let i = 0; i < S; i += step) {
      tctx.strokeStyle = `rgba(110, 84, 54, ${(0.028 * ea).toFixed(4)})`;
      tctx.beginPath();
      tctx.moveTo(0, i);
      tctx.lineTo(S, i);
      tctx.stroke();
      tctx.beginPath();
      tctx.moveTo(i, 0);
      tctx.lineTo(i, S);
      tctx.stroke();
      tctx.strokeStyle = `rgba(255, 251, 242, ${(0.04 * ea).toFixed(4)})`;
      tctx.beginPath();
      tctx.moveTo(0, i + 1);
      tctx.lineTo(S, i + 1);
      tctx.stroke();
    }
  } else if (sp.embossStyle === 'diamond') {
    const step = 16;
    tctx.lineWidth = 0.75;
    for (let k = -S; k <= S * 2; k += step) {
      tctx.strokeStyle = `rgba(112, 86, 56, ${(0.03 * ea).toFixed(4)})`;
      tctx.beginPath();
      tctx.moveTo(k, 0);
      tctx.lineTo(k + S, S);
      tctx.stroke();
      tctx.beginPath();
      tctx.moveTo(k + S, 0);
      tctx.lineTo(k, S);
      tctx.stroke();
      tctx.strokeStyle = `rgba(255, 250, 240, ${(0.042 * ea).toFixed(4)})`;
      tctx.beginPath();
      tctx.moveTo(k + 1, 0);
      tctx.lineTo(k + S + 1, S);
      tctx.stroke();
    }
  } else {
    const step = 6;
    for (let y = 0; y < S; y += step) {
      for (let x = 0; x < S; x += step) {
        tctx.fillStyle = `rgba(108, 82, 52, ${(0.038 * ea).toFixed(4)})`;
        tctx.fillRect(x, y, 1.3, 1.3);
        tctx.fillStyle = `rgba(255, 251, 244, ${(0.055 * ea).toFixed(4)})`;
        tctx.fillRect(x + 1, y + 1, 1.3, 1.3);
      }
    }
  }

  const rnd = makeRng(sp.seed ^ 0x9e3779b9);

  const drawWrapped = (x: number, y: number, margin: number, fn: (px: number, py: number) => void) => {
    const oxs = [0];
    if (x < margin) oxs.push(S);
    if (x > S - margin) oxs.push(-S);
    const oys = [0];
    if (y < margin) oys.push(S);
    if (y > S - margin) oys.push(-S);
    for (const ox of oxs) for (const oy of oys) fn(x + ox, y + oy);
  };

  for (let i = 0; i < 150; i++) {
    const x = rnd() * S;
    const y = rnd() * S;
    const r = 7 + rnd() * 22;
    tctx.fillStyle = rnd() > 0.5 ? 'rgba(130, 102, 68, 0.015)' : 'rgba(255, 248, 236, 0.022)';
    drawWrapped(x, y, r + 1, (px, py) => {
      tctx.beginPath();
      tctx.arc(px, py, r, 0, Math.PI * 2);
      tctx.fill();
    });
  }

  const fiberCount = Math.round(480 * sp.fiberMul);
  tctx.lineWidth = 0.7;
  for (let i = 0; i < fiberCount; i++) {
    const x = rnd() * S;
    const y = rnd() * S;
    const len = 2 + rnd() * 8;
    const ang = rnd() * Math.PI * 2;
    const dx = Math.cos(ang) * len;
    const dy = Math.sin(ang) * len;
    tctx.strokeStyle = rnd() > 0.38 ? 'rgba(118, 90, 58, 0.072)' : 'rgba(255, 252, 244, 0.088)';
    drawWrapped(x, y, len + 1, (px, py) => {
      tctx.beginPath();
      tctx.moveTo(px, py);
      tctx.lineTo(px + dx, py + dy);
      tctx.stroke();
    });
  }

  const fleckCount = Math.round(280 * sp.fleckMul);
  for (let i = 0; i < fleckCount; i++) {
    const sz = 0.9 + rnd() * 0.7;
    tctx.fillStyle = rnd() > 0.28 ? 'rgba(102, 74, 44, 0.082)' : 'rgba(72, 52, 32, 0.105)';
    tctx.fillRect(rnd() * S, rnd() * S, sz, sz);
  }

  return ctx.createPattern(tile, 'repeat');
}

function drawCreaseBand(
  ctx: CanvasRenderingContext2D,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  halfW: number,
  strength: number,
  flip: boolean,
): void {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy) || 1;
  const nx = -dy / len;
  const ny = dx / len;
  const mx = (x1 + x2) * 0.5;
  const my = (y1 + y2) * 0.5;
  const g = ctx.createLinearGradient(mx - nx * halfW, my - ny * halfW, mx + nx * halfW, my + ny * halfW);
  const dark = `rgba(88, 65, 38, ${strength.toFixed(4)})`;
  const light = `rgba(255, 250, 240, ${(strength * 1.18).toFixed(4)})`;
  g.addColorStop(0, 'rgba(88, 65, 38, 0)');
  g.addColorStop(0.45, flip ? light : dark);
  g.addColorStop(0.54, flip ? dark : light);
  g.addColorStop(1, 'rgba(88, 65, 38, 0)');

  ctx.save();
  ctx.fillStyle = g;
  ctx.beginPath();
  ctx.moveTo(x1 - nx * halfW, y1 - ny * halfW);
  ctx.lineTo(x2 - nx * halfW, y2 - ny * halfW);
  ctx.lineTo(x2 + nx * halfW, y2 + ny * halfW);
  ctx.lineTo(x1 + nx * halfW, y1 + ny * halfW);
  ctx.closePath();
  ctx.fill();
  ctx.restore();
}

export function buildNapkinSheetCanvas(sp: NapkinSpec, w: number, h: number, dpr: number): HTMLCanvasElement {
  const off = document.createElement('canvas');
  off.width = Math.max(1, Math.round(w * dpr));
  off.height = Math.max(1, Math.round(h * dpr));
  const ctx = off.getContext('2d');
  if (!ctx) return off;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

  const pat = createNapkinPattern(ctx, sp);
  if (pat) {
    ctx.save();
    ctx.translate(-sp.tileOffX, -sp.tileOffY);
    ctx.fillStyle = pat;
    ctx.fillRect(sp.tileOffX, sp.tileOffY, w, h);
    ctx.restore();
  }

  const rnd = makeRng(sp.seed ^ 0x517cc1b7);

  const cloudCount = 5 + Math.floor(rnd() * 4);
  for (let i = 0; i < cloudCount; i++) {
    const cx = rnd() * w;
    const cy = rnd() * h;
    const rad = Math.min(w, h) * (0.18 + rnd() * 0.28);
    const warm = rnd() > 0.45;
    const gr = ctx.createRadialGradient(cx, cy, 0, cx, cy, rad);
    gr.addColorStop(0, warm ? 'rgba(128, 96, 58, 0.036)' : 'rgba(255, 250, 240, 0.048)');
    gr.addColorStop(1, 'rgba(128, 96, 58, 0)');
    ctx.fillStyle = gr;
    ctx.fillRect(0, 0, w, h);
  }

  const uniqueFlecks = Math.round((45 + rnd() * 40) * sp.fleckMul);
  for (let i = 0; i < uniqueFlecks; i++) {
    const x = rnd() * w;
    const y = rnd() * h;
    if (rnd() < 0.45) {
      const len = 4 + rnd() * 11;
      const ang = rnd() * Math.PI * 2;
      ctx.strokeStyle = rnd() > 0.3 ? 'rgba(104, 76, 44, 0.11)' : 'rgba(255, 251, 242, 0.13)';
      ctx.lineWidth = 0.65 + rnd() * 0.45;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.quadraticCurveTo(
        x + Math.cos(ang + 0.4) * len * 0.5,
        y + Math.sin(ang + 0.4) * len * 0.5,
        x + Math.cos(ang) * len,
        y + Math.sin(ang) * len,
      );
      ctx.stroke();
    } else {
      ctx.fillStyle = `rgba(82, 58, 34, ${(0.09 + rnd() * 0.09).toFixed(3)})`;
      ctx.beginPath();
      ctx.ellipse(x, y, 0.7 + rnd() * 1.1, 0.5 + rnd() * 0.8, rnd() * Math.PI, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  if (sp.borderStyle !== 'none' && w > 120 && h > 120) {
    const pad = sp.borderInset;
    const bw = w - pad * 2;
    const bh = h - pad * 2;
    ctx.save();
    if (sp.borderStyle === 'double') {
      const drawRim = (inset: number, dx: number, dy: number, color: string) => {
        ctx.strokeStyle = color;
        ctx.lineWidth = 1.0;
        ctx.strokeRect(inset + dx, inset + dy, w - inset * 2, h - inset * 2);
      };
      drawRim(pad, 0, 0, 'rgba(102, 76, 46, 0.085)');
      drawRim(pad, 1, 1, 'rgba(255, 250, 240, 0.12)');
      drawRim(pad + 5, 0, 0, 'rgba(102, 76, 46, 0.065)');
      drawRim(pad + 5, 1, 1, 'rgba(255, 250, 240, 0.095)');
    } else if (sp.borderStyle === 'dotted') {
      const step = 9;
      for (let x = pad; x <= w - pad; x += step) {
        for (const y of [pad, h - pad]) {
          ctx.fillStyle = 'rgba(102, 76, 46, 0.11)';
          ctx.fillRect(x, y, 2, 2);
          ctx.fillStyle = 'rgba(255, 250, 240, 0.15)';
          ctx.fillRect(x + 1, y + 1, 2, 2);
        }
      }
      for (let y = pad + step; y < h - pad; y += step) {
        for (const x of [pad, w - pad]) {
          ctx.fillStyle = 'rgba(102, 76, 46, 0.11)';
          ctx.fillRect(x, y, 2, 2);
          ctx.fillStyle = 'rgba(255, 250, 240, 0.15)';
          ctx.fillRect(x + 1, y + 1, 2, 2);
        }
      }
    } else if (sp.borderStyle === 'stitch') {
      ctx.setLineDash([4, 5]);
      ctx.lineWidth = 1.1;
      ctx.strokeStyle = 'rgba(102, 76, 46, 0.095)';
      ctx.strokeRect(pad, pad, bw, bh);
      ctx.strokeStyle = 'rgba(255, 250, 240, 0.13)';
      ctx.strokeRect(pad + 1, pad + 1, bw, bh);
    }
    ctx.restore();
  }

  const vx = w * sp.foldVPos;
  const vTilt = Math.tan(sp.foldVAngle) * h * 0.5;
  drawCreaseBand(ctx, vx - vTilt, -10, vx + vTilt, h + 10, sp.foldVWidth * 0.5, sp.foldVStrength, sp.foldVFlip);

  const hy = h * sp.foldHPos;
  const hTilt = Math.tan(sp.foldHAngle) * w * 0.5;
  drawCreaseBand(ctx, -10, hy - hTilt, w + 10, hy + hTilt, sp.foldHWidth * 0.5, sp.foldHStrength, sp.foldHFlip);

  for (const dc of sp.diagCreases) {
    const s = Math.min(w, h) * dc.span;
    const sk = s * dc.skew;
    let x1: number, y1: number, x2: number, y2: number;
    if (dc.corner === 0) {
      x1 = -10; y1 = s + sk; x2 = s - sk; y2 = -10;
    } else if (dc.corner === 1) {
      x1 = w - s + sk; y1 = -10; x2 = w + 10; y2 = s + sk;
    } else if (dc.corner === 2) {
      x1 = -10; y1 = h - s - sk; x2 = s + sk; y2 = h + 10;
    } else {
      x1 = w - s - sk; y1 = h + 10; x2 = w + 10; y2 = h - s + sk;
    }
    drawCreaseBand(ctx, x1, y1, x2, y2, 11, dc.alpha, dc.flip > 0);
  }

  if (sp.cupRing) {
    const cr = sp.cupRing;
    const rx = cr.nx * w;
    const ry = cr.ny * h;
    const rad = cr.radius;
    ctx.save();
    ctx.translate(rx, ry);
    ctx.rotate(cr.rot);
    for (let pass = 0; pass < 3; pass++) {
      ctx.beginPath();
      const segs = 48;
      for (let i = 0; i <= segs; i++) {
        const a = (i / segs) * Math.PI * 2;
        const wob = Math.sin(a * 3 + pass) * 1.4 + Math.cos(a * 5 - pass * 1.7) * 0.9;
        const rNow = rad + (pass - 1) * 1.8 + wob;
        const px = Math.cos(a) * rNow;
        const py = Math.sin(a) * rNow * 0.96;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      }
      ctx.strokeStyle = `rgba(116, 74, 34, ${(cr.alpha * (pass === 1 ? 1.0 : 0.45)).toFixed(4)})`;
      ctx.lineWidth = pass === 1 ? 2.4 : 4.6;
      ctx.stroke();
    }
    ctx.restore();
  }

  const vig = ctx.createRadialGradient(
    w * sp.lightX,
    h * sp.lightY,
    Math.min(w, h) * 0.3,
    w * 0.5,
    h * 0.5,
    Math.max(w, h) * 0.8,
  );
  vig.addColorStop(0, 'rgba(255, 249, 238, 0.045)');
  vig.addColorStop(1, `rgba(108, 80, 48, ${sp.vigAlpha.toFixed(4)})`);
  ctx.fillStyle = vig;
  ctx.fillRect(0, 0, w, h);

  return off;
}

export function drawNapkinBackground(ctx: CanvasRenderingContext2D, w: number, h: number): void {
  const sp = getNapkinSpec();
  const dpr = window.devicePixelRatio || 1;
  const cw = Math.round(w * dpr);
  const ch = Math.round(h * dpr);
  if (!bgCache || bgCache.id !== sp.id || bgCache.cw !== cw || bgCache.ch !== ch) {
    bgCache = {
      id: sp.id,
      cw,
      ch,
      canvas: buildNapkinSheetCanvas(sp, w, h, dpr),
    };
  }
  ctx.drawImage(bgCache.canvas, 0, 0, w, h);
}
