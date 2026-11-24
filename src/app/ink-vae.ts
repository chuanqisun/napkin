import { chaikin, makeRng, randn, type Rng } from './rng';
import {
  DEFAULTS,
  WRITING_STYLES,
  type ArrowItem,
  type FreehandItem,
  type HandwriteItem,
  type InkStroke,
  type NapkinItem,
  type PenId,
  type Point2D,
  type ShapeItem,
  type ShapeType,
  type StrokePoint,
  type StyleKey,
  type VAEModel,
} from './types';

let vaeModel: VAEModel | null = null;

export function setVAEModel(model: VAEModel): void {
  vaeModel = model;
}

function silu(x: number): number {
  return x / (1 + Math.exp(-x));
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
}

export function decodeResidual(
  isCurved: 0 | 1,
  isPres: number,
  chordLen: number,
  tau: number = DEFAULTS.tau,
  rng: Rng = Math.random,
): StrokePoint[] {
  if (!vaeModel) {
    return Array.from({ length: 16 }, (_, i): StrokePoint => [
      0,
      0,
      0.25 + 0.55 * Math.sin((i / 15) * Math.PI),
    ]);
  }
  const nll = Math.max(-1.5, Math.min(1.5, (Math.log(Math.max(chordLen, 25)) - Math.log(140)) / 1.2));
  let h: number[] = [];
  for (let i = 0; i < vaeModel.latent_dim; i++) h.push(randn(rng) * tau);
  h.push(isCurved, isPres, nll);

  for (let li = 0; li < vaeModel.layers.length; li++) {
    const layer = vaeModel.layers[li]!;
    const out = new Array<number>(layer.b.length);
    for (let i = 0; i < layer.b.length; i++) {
      let sum = layer.b[i]!;
      const row = layer.w[i]!;
      for (let j = 0; j < row.length; j++) sum += row[j]! * h[j]!;
      out[i] = li < vaeModel.layers.length - 1 ? silu(sum) : sum;
    }
    h = out;
  }

  const K = vaeModel.k;
  const sc = vaeModel.res_scale;
  const out: StrokePoint[] = [];
  for (let i = 0; i < K; i++) {
    out.push([h[i * 3]! * sc, h[i * 3 + 1]! * sc, sigmoid(h[i * 3 + 2]!)]);
  }
  return out;
}

export function designerLine(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  isPres: number = DEFAULTS.isPres,
  tau: number = DEFAULTS.tau,
  ov = 3,
  rng: Rng = Math.random,
): StrokePoint[] {
  const dx = x2 - x1;
  const dy = y2 - y1;
  const len = Math.hypot(dx, dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const nx = -uy;
  const ny = ux;
  const sx = x1 - ux * ov * rng();
  const sy = y1 - uy * ov * rng();
  const ex = x2 + ux * ov * rng();
  const ey = y2 + uy * ov * rng();
  const res = decodeResidual(0, isPres, len, tau, rng);
  const K = res.length;
  return res.map(([ru, rv, p], i): StrokePoint => {
    const t = i / (K - 1);
    return [
      sx + (ex - sx) * (t + ru) + nx * rv * len,
      sy + (ey - sy) * (t + ru) + ny * rv * len,
      p,
    ];
  });
}

export function designerBezier(
  x0: number,
  y0: number,
  cx: number,
  cy: number,
  x2: number,
  y2: number,
  isPres: number = DEFAULTS.isPres,
  tau: number = DEFAULTS.tau,
  rng: Rng = Math.random,
): StrokePoint[] {
  const chord = Math.hypot(x2 - x0, y2 - y0) || 1;
  const ux = (x2 - x0) / chord;
  const uy = (y2 - y0) / chord;
  const nx = -uy;
  const ny = ux;
  const res = decodeResidual(1, isPres, chord, tau, rng);
  const K = res.length;
  return res.map(([ru, rv, p], i): StrokePoint => {
    const t = i / (K - 1);
    const mt = 1 - t;
    const env = Math.sin(t * Math.PI);
    const bx = mt * mt * x0 + 2 * mt * t * cx + t * t * x2;
    const by = mt * mt * y0 + 2 * mt * t * cy + t * t * y2;
    return [bx + (ux * ru + nx * rv) * chord * env, by + (uy * ru + ny * rv) * chord * env, p];
  });
}

export function buildShapeStrokes(
  shapeType: ShapeType,
  x: number,
  y: number,
  w: number,
  h: number,
  col: string,
  penId: PenId,
  seed: number | null = null,
): InkStroke[] {
  const strokes: InkStroke[] = [];
  const rng = seed != null ? makeRng(seed) : Math.random;
  const ovScale = Math.min(1, Math.min(w, h) / 28);
  const addPass = (isP: number, tau: number, ov: number, alpha: number, baseW: number) => {
    const scaledOv = ov * ovScale;
    if (shapeType === 'box') {
      strokes.push({ pts: designerLine(x, y, x + w, y, isP, tau, scaledOv, rng), alpha, baseW, color: col, penId });
      strokes.push({ pts: designerLine(x + w, y, x + w, y + h, isP, tau, scaledOv, rng), alpha, baseW, color: col, penId });
      strokes.push({ pts: designerLine(x + w, y + h, x, y + h, isP, tau, scaledOv, rng), alpha, baseW, color: col, penId });
      strokes.push({ pts: designerLine(x, y + h, x, y, isP, tau, scaledOv, rng), alpha, baseW, color: col, penId });
    } else if (shapeType === 'diamond') {
      const cx = x + w / 2;
      const cy = y + h / 2;
      strokes.push({ pts: designerLine(cx, y, x + w, cy, isP, tau, scaledOv, rng), alpha, baseW, color: col, penId });
      strokes.push({ pts: designerLine(x + w, cy, cx, y + h, isP, tau, scaledOv, rng), alpha, baseW, color: col, penId });
      strokes.push({ pts: designerLine(cx, y + h, x, cy, isP, tau, scaledOv, rng), alpha, baseW, color: col, penId });
      strokes.push({ pts: designerLine(x, cy, cx, y, isP, tau, scaledOv, rng), alpha, baseW, color: col, penId });
    } else {
      const cx = x + w / 2;
      const cy = y + h / 2;
      const rx = w / 2;
      const ry = h / 2;
      strokes.push({ pts: designerBezier(cx - rx, cy, cx - rx, cy - ry, cx, cy - ry, isP, tau, rng), alpha, baseW, color: col, penId });
      strokes.push({ pts: designerBezier(cx, cy - ry, cx + rx, cy - ry, cx + rx, cy, isP, tau, rng), alpha, baseW, color: col, penId });
      strokes.push({ pts: designerBezier(cx + rx, cy, cx + rx, cy + ry, cx, cy + ry, isP, tau, rng), alpha, baseW, color: col, penId });
      strokes.push({ pts: designerBezier(cx, cy + ry, cx - rx, cy + ry, cx - rx, cy, isP, tau, rng), alpha, baseW, color: col, penId });
    }
  };
  if (DEFAULTS.underSketch) addPass(0.0, DEFAULTS.tau * 1.15, 6.0, 0.2, 1.25);
  addPass(DEFAULTS.isPres, DEFAULTS.tau, 2.5, 0.98, 2.1);
  return strokes;
}

export function buildShapeItem(
  shapeType: ShapeType,
  x: number,
  y: number,
  w: number,
  h: number,
  label: string,
  color: string,
  penId: PenId,
  styleKey: StyleKey,
  seed: number | null = null,
): ShapeItem {
  return {
    kind: 'shape',
    shapeType,
    x,
    y,
    w,
    h,
    label,
    styleKey,
    penId,
    color,
    strokes: buildShapeStrokes(shapeType, x, y, w, h, color, penId, seed),
    textStrokes: [],
  };
}

export function buildArrowItem(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  color: string,
  penId: PenId,
  seed: number | null = null,
): ArrowItem {
  const rng = seed != null ? makeRng(seed) : Math.random;
  const dist = Math.hypot(x2 - x1, y2 - y1);
  const ang = Math.atan2(y2 - y1, x2 - x1);
  const hl = Math.min(12, Math.max(4, dist * 0.38));
  const ovScale = Math.min(1, dist / 36);
  const strokes: InkStroke[] = [
    { pts: designerLine(x1, y1, x2, y2, DEFAULTS.isPres, DEFAULTS.tau, 1.5 * ovScale, rng), alpha: 0.96, baseW: 1.95, color, penId },
    {
      pts: designerLine(
        x2 - hl * Math.cos(ang - 0.45),
        y2 - hl * Math.sin(ang - 0.45),
        x2,
        y2,
        DEFAULTS.isPres,
        DEFAULTS.tau,
        1.0 * ovScale,
        rng,
      ),
      alpha: 0.96,
      baseW: 1.8,
      color,
      penId,
    },
    {
      pts: designerLine(
        x2 - hl * Math.cos(ang + 0.45),
        y2 - hl * Math.sin(ang + 0.45),
        x2,
        y2,
        DEFAULTS.isPres,
        DEFAULTS.tau,
        1.0 * ovScale,
        rng,
      ),
      alpha: 0.96,
      baseW: 1.8,
      color,
      penId,
    },
  ];
  return {
    kind: 'arrow',
    x1,
    y1,
    x2,
    y2,
    penId,
    x: Math.min(x1, x2) - 6,
    y: Math.min(y1, y2) - 6,
    w: Math.max(18, Math.abs(x2 - x1) + 12),
    h: Math.max(18, Math.abs(y2 - y1) + 12),
    color,
    strokes,
  };
}

export function buildHandwriteObject(
  rawList: Point2D[][],
  color: string,
  styleKey: StyleKey,
  penId: PenId,
): HandwriteItem {
  const st = WRITING_STYLES[styleKey];
  const rawCopy = rawList.map(r => r.map(p => ({ x: p.x, y: p.y })));
  let sumY = 0;
  let cnt = 0;
  let minX = 1e9;
  let maxX = -1e9;
  for (const raw of rawCopy) {
    for (const p of raw) {
      sumY += p.y;
      cnt++;
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
    }
  }
  const midY = cnt ? sumY / cnt : 0;
  let slope = 0;
  if (st.hwBaselinePull > 0 && maxX - minX > 30 && cnt > 8) {
    let sxx = 0;
    let sxy = 0;
    const cx = (minX + maxX) * 0.5;
    for (const raw of rawCopy) {
      for (const p of raw) {
        const dx = p.x - cx;
        const dy = p.y - midY;
        sxx += dx * dx;
        sxy += dx * dy;
      }
    }
    if (sxx > 1) slope = Math.max(-0.22, Math.min(0.22, sxy / sxx)) * st.hwBaselinePull;
  }
  const midX = (minX + maxX) * 0.5;

  const strokes: InkStroke[] = rawCopy.map(raw => {
    const first = raw[0]!;
    const simp: Point2D[] = [first];
    for (let i = 1; i < raw.length; i++) {
      const prev = simp[simp.length - 1]!;
      const cur = raw[i]!;
      if (Math.hypot(cur.x - prev.x, cur.y - prev.y) >= st.hwMinDist) simp.push(cur);
    }
    if (simp.length < 2) simp.push({ x: first.x + 1.0, y: first.y + 0.8 });
    const basePts: [number, number][] = simp.map(p => {
      const yCorr = p.y - (p.x - midX) * slope;
      return [p.x + (midY - yCorr) * st.hwSlant, yCorr];
    });
    const smoothed = st.hwChaikin > 0 ? chaikin(basePts, st.hwChaikin) : basePts;
    const pts: StrokePoint[] = smoothed.map(([x, y], i) => {
      const prev = smoothed[Math.max(0, i - 1)]!;
      const sp = Math.hypot(x - prev[0], y - prev[1]);
      const t = i / Math.max(1, smoothed.length - 1);
      const endTaper = Math.min(1, t * 7, (1 - t) * 7);
      const p = Math.max(st.hwMinP, Math.min(0.95, (0.9 - sp * st.hwVelDrop) * (0.42 + 0.58 * endTaper)));
      return [x, y, p];
    });
    return { pts, alpha: 0.96, baseW: st.hwBaseW, color, penId };
  });

  let bMinX = 1e9;
  let bMaxX = -1e9;
  let bMinY = 1e9;
  let bMaxY = -1e9;
  for (const s of strokes) {
    for (const [x, y] of s.pts) {
      if (x < bMinX) bMinX = x;
      if (x > bMaxX) bMaxX = x;
      if (y < bMinY) bMinY = y;
      if (y > bMaxY) bMaxY = y;
    }
  }
  return {
    kind: 'handwrite',
    styleKey,
    penId,
    color,
    rawList: rawCopy,
    strokes,
    x: bMinX,
    y: bMinY,
    w: Math.max(16, bMaxX - bMinX),
    h: Math.max(16, bMaxY - bMinY),
  };
}

export function buildFreehandObject(rawList: Point2D[][], color: string, penId: PenId): FreehandItem {
  const strokes: InkStroke[] = rawList.map(raw => {
    const first = raw[0]!;
    const simp: Point2D[] = [first];
    for (let i = 1; i < raw.length; i++) {
      const prev = simp[simp.length - 1]!;
      const cur = raw[i]!;
      if (Math.hypot(cur.x - prev.x, cur.y - prev.y) >= 2.2) simp.push(cur);
    }
    if (simp.length < 2) simp.push({ x: first.x + 0.9, y: first.y + 0.9 });
    const smoothed = chaikin(
      simp.map(p => [p.x, p.y] as [number, number]),
      1,
    );
    const pts: StrokePoint[] = smoothed.map(([x, y], i) => {
      const prev = smoothed[Math.max(0, i - 1)]!;
      const sp = Math.hypot(x - prev[0], y - prev[1]);
      const t = i / Math.max(1, smoothed.length - 1);
      const endTaper = Math.min(1, t * 7, (1 - t) * 7);
      const p = Math.max(0.22, Math.min(0.94, (0.87 - sp * 0.03) * (0.42 + 0.58 * endTaper)));
      return [x, y, p];
    });
    return { pts, alpha: 0.96, baseW: 1.9, color, penId };
  });

  let minX = 1e9;
  let maxX = -1e9;
  let minY = 1e9;
  let maxY = -1e9;
  for (const s of strokes) {
    for (const [x, y] of s.pts) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
  }
  return {
    kind: 'free',
    penId,
    color,
    strokes,
    x: minX,
    y: minY,
    w: Math.max(16, maxX - minX),
    h: Math.max(16, maxY - minY),
  };
}

function resampleStrokesUniform(rawList: Point2D[][], totalK = 64): Point2D[] {
  const lengths = rawList.map(raw => {
    let L = 0;
    for (let i = 1; i < raw.length; i++) {
      L += Math.hypot(raw[i]!.x - raw[i - 1]!.x, raw[i]!.y - raw[i - 1]!.y);
    }
    return Math.max(L, 1e-3);
  });
  const sumL = lengths.reduce((a, b) => a + b, 0);
  const out: Point2D[] = [];
  rawList.forEach((raw, idx) => {
    const k = Math.max(3, Math.round((totalK * lengths[idx]!) / sumL));
    const cum: number[] = [0];
    for (let i = 1; i < raw.length; i++) {
      cum.push(cum[i - 1]! + Math.hypot(raw[i]!.x - raw[i - 1]!.x, raw[i]!.y - raw[i - 1]!.y));
    }
    const L = cum[cum.length - 1] || 1;
    let j = 0;
    for (let m = 0; m < k; m++) {
      const target = (m / Math.max(1, k - 1)) * L;
      while (j < cum.length - 2 && cum[j + 1]! < target) j++;
      const span = Math.max(1e-6, cum[j + 1]! - cum[j]!);
      const t = Math.max(0, Math.min(1, (target - cum[j]!) / span));
      const a = raw[j]!;
      const b = raw[j + 1] ?? a;
      out.push({ x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t });
    }
  });
  return out;
}

export function recognizeMultiStroke(
  rawList: Point2D[][],
  color: string,
  penId: PenId,
  styleKey: StyleKey,
): NapkinItem {
  const stats = rawList.map(raw => {
    let arc = 0;
    for (let i = 1; i < raw.length; i++) {
      arc += Math.hypot(raw[i]!.x - raw[i - 1]!.x, raw[i]!.y - raw[i - 1]!.y);
    }
    const s = raw[0]!;
    const e = raw[raw.length - 1]!;
    const chord = Math.hypot(e.x - s.x, e.y - s.y);
    return { raw, s, e, arc, chord, straightness: chord / Math.max(arc, 1) };
  });

  const uniPts = resampleStrokesUniform(rawList, 64);
  let minX = 1e9;
  let maxX = -1e9;
  let minY = 1e9;
  let maxY = -1e9;
  for (const p of uniPts) {
    if (p.x < minX) minX = p.x;
    if (p.x > maxX) maxX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.y > maxY) maxY = p.y;
  }
  const w = maxX - minX;
  const h = maxY - minY;
  const diag = Math.hypot(w, h);
  if (diag < 42) return buildFreehandObject(rawList, color, penId);

  // High-confidence arrow detection: requires an actual arrowhead (chevron / barbs / hooked tip)
  // so plain straight lines, underlines, or letters (l, 1, I, /, -, T) stay freehand.
  if (rawList.length === 1) {
    const main = stats[0]!;
    if (main.raw.length >= 6 && main.arc >= 60) {
      // Find the point farthest from start; in a 1-stroke arrow (shaft + reverse barb),
      // the shaft goes straight to the tip and then reverses sharply to form the barb.
      let tipIdx = 0;
      let maxDist = 0;
      for (let i = 1; i < main.raw.length; i++) {
        const d = Math.hypot(main.raw[i]!.x - main.s.x, main.raw[i]!.y - main.s.y);
        if (d > maxDist) {
          maxDist = d;
          tipIdx = i;
        }
      }
      if (tipIdx >= 2 && tipIdx <= main.raw.length - 2 && maxDist >= 52) {
        let shaftArc = 0;
        for (let i = 1; i <= tipIdx; i++) {
          shaftArc += Math.hypot(main.raw[i]!.x - main.raw[i - 1]!.x, main.raw[i]!.y - main.raw[i - 1]!.y);
        }
        let barbArc = 0;
        for (let i = tipIdx + 1; i < main.raw.length; i++) {
          barbArc += Math.hypot(main.raw[i]!.x - main.raw[i - 1]!.x, main.raw[i]!.y - main.raw[i - 1]!.y);
        }
        const tip = main.raw[tipIdx]!;
        const barbEnd = main.e;
        const barbChord = Math.hypot(barbEnd.x - tip.x, barbEnd.y - tip.y);
        const shaftStraight = maxDist / Math.max(shaftArc, 1);
        const ux = (tip.x - main.s.x) / maxDist;
        const uy = (tip.y - main.s.y) / maxDist;
        const bx = barbEnd.x - tip.x;
        const by = barbEnd.y - tip.y;
        const backProj = -(bx * ux + by * uy);
        const latOff = Math.abs(-uy * bx + ux * by);
        if (
          shaftStraight > 0.88 &&
          barbArc >= 10 &&
          barbArc <= shaftArc * 0.45 &&
          barbChord >= 8 &&
          backProj > barbChord * 0.35 &&
          latOff > barbChord * 0.18 &&
          latOff < barbChord * 0.92
        ) {
          return buildArrowItem(main.s.x, main.s.y, tip.x, tip.y, color, penId);
        }
      }
    }
  } else if (rawList.length === 2 || rawList.length === 3) {
    let mainIdx = 0;
    for (let i = 1; i < stats.length; i++) {
      if (stats[i]!.arc > stats[mainIdx]!.arc) mainIdx = i;
    }
    const main = stats[mainIdx]!;
    if (main.straightness > 0.84 && main.chord > 48) {
      const others = stats.filter((_, i) => i !== mainIdx);
      const checkHeadAt = (tail: Point2D, tip: Point2D): boolean => {
        const len = Math.hypot(tip.x - tail.x, tip.y - tail.y) || 1;
        const ux = (tip.x - tail.x) / len;
        const uy = (tip.y - tail.y) / len;
        const nx = -uy;
        const ny = ux;
        const tipTol = Math.max(24, len * 0.26);

        if (others.length === 1) {
          // 2-stroke arrow: shaft + V-shaped chevron whose apex is near tip
          const o = others[0]!;
          if (o.arc < 14 || o.arc > main.arc * 0.65 || o.straightness > 0.82) return false;
          let bestIdx = 0;
          let bestD = 1e9;
          for (let i = 0; i < o.raw.length; i++) {
            const d = Math.hypot(o.raw[i]!.x - tip.x, o.raw[i]!.y - tip.y);
            if (d < bestD) {
              bestD = d;
              bestIdx = i;
            }
          }
          if (bestD > tipTol) return false;
          const apex = o.raw[bestIdx]!;
          const sBack = -((o.s.x - apex.x) * ux + (o.s.y - apex.y) * uy);
          const eBack = -((o.e.x - apex.x) * ux + (o.e.y - apex.y) * uy);
          const sSide = (o.s.x - apex.x) * nx + (o.s.y - apex.y) * ny;
          const eSide = (o.e.x - apex.x) * nx + (o.e.y - apex.y) * ny;
          return sBack > 4 && eBack > 4 && sSide * eSide < -12;
        }

        // 3-stroke arrow: shaft + 2 barbs on opposite sides pointing back from tip
        const sides: number[] = [];
        for (const o of others) {
          if (o.arc < 8 || o.arc > main.arc * 0.5 || o.straightness < 0.72) return false;
          const ds = Math.hypot(o.s.x - tip.x, o.s.y - tip.y);
          const de = Math.hypot(o.e.x - tip.x, o.e.y - tip.y);
          if (Math.min(ds, de) > tipTol) return false;
          const near = ds <= de ? o.s : o.e;
          const far = ds <= de ? o.e : o.s;
          const bx = far.x - near.x;
          const by = far.y - near.y;
          const blen = Math.hypot(bx, by) || 1;
          const back = -(bx * ux + by * uy) / blen;
          const side = (bx * nx + by * ny) / blen;
          if (back < 0.25 || Math.abs(side) < 0.18 || Math.abs(side) > 0.95) return false;
          sides.push(side);
        }
        return sides.length === 2 && sides[0]! * sides[1]! < 0;
      };

      if (checkHeadAt(main.s, main.e)) return buildArrowItem(main.s.x, main.s.y, main.e.x, main.e.y, color, penId);
      if (checkHeadAt(main.e, main.s)) return buildArrowItem(main.e.x, main.e.y, main.s.x, main.s.y, color, penId);
    }
  }

  // High-confidence closed shape detection (box / oval / diamond)
  const aspect = Math.max(w / Math.max(h, 1), h / Math.max(w, 1));
  const totalArc = stats.reduce((a, s) => a + s.arc, 0);
  const boxPerim = 2 * (w + h);
  const ovalPerim = Math.PI * (3 * (w / 2 + h / 2) - Math.sqrt((3 * (w / 2) + h / 2) * (w / 2 + 3 * (h / 2))));
  const minPerim = Math.min(ovalPerim, 2 * Math.hypot(w, h));
  const perimOk = totalArc >= minPerim * 0.78 && totalArc <= boxPerim * 1.38;

  const isSingleClosed = rawList.length === 1 && stats[0]!.chord < diag * 0.28 && perimOk;
  let isMultiClosed = false;
  if (rawList.length >= 2 && rawList.length <= 4 && perimOk) {
    const minStrokeArc = Math.min(w, h) * 0.22;
    const allLongEnough = stats.every(s => s.arc >= minStrokeArc);
    if (allLongEnough) {
      // Every stroke endpoint should connect near another stroke's endpoint (forms a closed loop, not separate letters)
      const connTol = Math.max(20, diag * 0.22);
      isMultiClosed = stats.every((st, idx) => {
        const nearOther = (pt: Point2D) =>
          stats.some(
            (o, j) =>
              j !== idx &&
              (Math.hypot(pt.x - o.s.x, pt.y - o.s.y) <= connTol ||
                Math.hypot(pt.x - o.e.x, pt.y - o.e.y) <= connTol),
          );
        return nearOther(st.s) && nearOther(st.e);
      });
    }
  }

  if ((isSingleClosed || isMultiClosed) && w >= 44 && h >= 34 && diag >= 62 && aspect <= 4.2) {
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const rx = w / 2;
    const ry = h / 2;
    let eErr = 0;
    let bErr = 0;
    let dErr = 0;
    const eErrs: number[] = [];
    const bErrs: number[] = [];
    const dErrs: number[] = [];
    const radii: number[] = [];
    let octantMask = 0;

    for (const p of uniPts) {
      const dx = (p.x - cx) / rx;
      const dy = (p.y - cy) / ry;
      const ang = Math.atan2(dy, dx);
      const oct = Math.floor(((ang + Math.PI) / (Math.PI * 2)) * 8) & 7;
      octantMask |= 1 << oct;

      const nx = Math.abs(dx);
      const ny = Math.abs(dy);
      const r = Math.hypot(nx, ny);
      radii.push(r);
      const invR = 1 / Math.max(r, 1e-6);
      const ca = nx * invR;
      const sa = ny * invR;
      const rBox = 1 / Math.max(Math.max(ca, sa), 1e-6);
      const rDia = 1 / Math.max(ca + sa, 1e-6);
      const ee = Math.abs(r - 1.0);
      const be = Math.abs(r - rBox);
      const de = Math.abs(r - rDia);
      eErr += ee;
      bErr += be;
      dErr += de;
      eErrs.push(ee);
      bErrs.push(be);
      dErrs.push(de);
    }

    let octCount = 0;
    for (let k = 0; k < 8; k++) if (octantMask & (1 << k)) octCount++;

    const n = uniPts.length;
    eErr /= n;
    bErr /= n;
    dErr /= n;

    radii.sort((a, b) => a - b);
    const p10 = radii[Math.floor(n * 0.1)]!;
    const p15 = radii[Math.floor(n * 0.15)]!;
    const p90 = radii[Math.min(n - 1, Math.floor(n * 0.9))]!;

    // Reject open arcs (<7 octants) and shapes/words with interior strokes (low p10)
    if (octCount >= 7 && p10 >= 0.6) {
      const bScore = bErr - (p90 > 1.16 ? 0.035 : -0.025);
      const dScore = dErr - (p15 < 0.77 ? 0.025 : 0.0);
      const eScore = eErr;

      let shapeType: ShapeType = 'oval';
      let winErr = eErr;
      let winArr = eErrs;
      if (dScore < eScore && dScore < bScore) {
        shapeType = 'diamond';
        winErr = dErr;
        winArr = dErrs;
      } else if (bScore < eScore) {
        shapeType = 'box';
        winErr = bErr;
        winArr = bErrs;
      }

      winArr.sort((a, b) => a - b);
      const winP90 = winArr[Math.min(n - 1, Math.floor(n * 0.9))]!;
      if (winErr < 0.165 && winP90 < 0.3) {
        return buildShapeItem(shapeType, minX, minY, w, h, '', color, penId, styleKey);
      }
    }
  }

  return buildFreehandObject(rawList, color, penId);
}

export type StrokeOutline =
  | { dot: true; x: number; y: number; r: number }
  | { dot: false; poly: [number, number][] };

export function buildStrokeOutline(
  pts: readonly StrokePoint[],
  baseW: number,
  penId: PenId = 'ballpoint',
): StrokeOutline | null {
  if (!pts || pts.length === 0) return null;

  const clean: StrokePoint[] = [pts[0]!];
  for (let i = 1; i < pts.length; i++) {
    const prev = clean[clean.length - 1]!;
    const cur = pts[i]!;
    if (Math.hypot(cur[0] - prev[0], cur[1] - prev[1]) >= 0.35) {
      clean.push(cur);
    }
  }
  const n = clean.length;

  const radiusFor = (p: number, ang: number): number => {
    if (penId === 'pencil') return Math.max(0.45, baseW * (0.76 + 0.56 * p) * 0.5);
    if (penId === 'marker') return Math.max(1.3, baseW * (1.85 + 0.62 * p) * 0.5);
    if (penId === 'fountain') {
      const nib = 0.42 + 0.96 * Math.abs(Math.sin(ang - 0.72));
      return Math.max(0.28, baseW * (0.3 + 1.32 * p) * nib * 0.5);
    }
    if (penId === 'mech') return Math.max(0.28, baseW * (0.5 + 0.24 * p) * 0.5);
    return Math.max(0.32, baseW * (0.38 + 0.92 * p) * 0.5);
  };

  if (n < 2) {
    const first = clean[0]!;
    return { dot: true, x: first[0], y: first[1], r: radiusFor(first[2] ?? 0.8, 0) };
  }

  const segT: [number, number][] = [];
  for (let i = 0; i < n - 1; i++) {
    const dx = clean[i + 1]![0] - clean[i]![0];
    const dy = clean[i + 1]![1] - clean[i]![1];
    const d = Math.hypot(dx, dy) || 1;
    segT.push([dx / d, dy / d]);
  }

  const rawR = new Array<number>(n);
  for (let i = 0; i < n; i++) {
    const t: [number, number] =
      i === 0
        ? segT[0]!
        : i === n - 1
          ? segT[n - 2]!
          : [segT[i - 1]![0] + segT[i]![0], segT[i - 1]![1] + segT[i]![1]];
    const ang = Math.atan2(t[1], t[0]);
    rawR[i] = radiusFor(clean[i]![2] ?? 0.75, ang);
  }
  const rad = new Array<number>(n);
  rad[0] = rawR[0]!;
  rad[n - 1] = rawR[n - 1]!;
  for (let i = 1; i < n - 1; i++) {
    rad[i] = rawR[i - 1]! * 0.22 + rawR[i]! * 0.56 + rawR[i + 1]! * 0.22;
  }

  const left: [number, number][] = [];
  const right: [number, number][] = [];
  const pushArc = (
    arr: [number, number][],
    cx: number,
    cy: number,
    r: number,
    aStart: number,
    aEnd: number,
    steps: number,
  ) => {
    let dA = aEnd - aStart;
    while (dA > Math.PI) dA -= Math.PI * 2;
    while (dA < -Math.PI) dA += Math.PI * 2;
    for (let s = 0; s <= steps; s++) {
      const a = aStart + dA * (s / steps);
      arr.push([cx + Math.cos(a) * r, cy + Math.sin(a) * r]);
    }
  };

  {
    const [tx, ty] = segT[0]!;
    const nx = -ty;
    const ny = tx;
    const r = rad[0]!;
    left.push([clean[0]![0] + nx * r, clean[0]![1] + ny * r]);
    right.push([clean[0]![0] - nx * r, clean[0]![1] - ny * r]);
  }

  for (let i = 1; i < n - 1; i++) {
    const [tx0, ty0] = segT[i - 1]!;
    const [tx1, ty1] = segT[i]!;
    const nx0 = -ty0;
    const ny0 = tx0;
    const nx1 = -ty1;
    const ny1 = tx1;
    const x = clean[i]![0];
    const y = clean[i]![1];
    const r = rad[i]!;
    const dot = tx0 * tx1 + ty0 * ty1;

    if (dot > 0.55) {
      let mx = nx0 + nx1;
      let my = ny0 + ny1;
      const ml = Math.hypot(mx, my) || 1;
      mx /= ml;
      my /= ml;
      const miter = Math.min(1.22, 1 / Math.max(0.6, mx * nx0 + my * ny0));
      left.push([x + mx * r * miter, y + my * r * miter]);
      right.push([x - mx * r * miter, y - my * r * miter]);
    } else {
      const cross = tx0 * ty1 - ty0 * tx1;
      if (cross >= 0) {
        pushArc(left, x, y, r, Math.atan2(ny0, nx0), Math.atan2(ny1, nx1), 4);
        right.push([x - nx0 * r, y - ny0 * r]);
        right.push([x - nx1 * r, y - ny1 * r]);
      } else {
        left.push([x + nx0 * r, y + ny0 * r]);
        left.push([x + nx1 * r, y + ny1 * r]);
        pushArc(right, x, y, r, Math.atan2(-ny0, -nx0), Math.atan2(-ny1, -nx1), 4);
      }
    }
  }

  {
    const [tx, ty] = segT[n - 2]!;
    const nx = -ty;
    const ny = tx;
    const r = rad[n - 1]!;
    left.push([clean[n - 1]![0] + nx * r, clean[n - 1]![1] + ny * r]);
    right.push([clean[n - 1]![0] - nx * r, clean[n - 1]![1] - ny * r]);
  }

  const poly: [number, number][] = [];
  for (const pt of left) poly.push(pt);

  const [etx, ety] = segT[n - 2]!;
  const endAng = Math.atan2(ety, etx);
  const er = rad[n - 1]!;
  for (let s = 1; s < 5; s++) {
    const a = endAng + Math.PI * 0.5 - (Math.PI * s) / 5;
    poly.push([clean[n - 1]![0] + Math.cos(a) * er, clean[n - 1]![1] + Math.sin(a) * er]);
  }

  for (let i = right.length - 1; i >= 0; i--) poly.push(right[i]!);

  const [stx, sty] = segT[0]!;
  const startAng = Math.atan2(sty, stx);
  const sr = rad[0]!;
  for (let s = 1; s < 5; s++) {
    const a = startAng - Math.PI * 0.5 - (Math.PI * s) / 5;
    poly.push([clean[0]![0] + Math.cos(a) * sr, clean[0]![1] + Math.sin(a) * sr]);
  }

  return { dot: false, poly };
}

function tracePoly(ctx: CanvasRenderingContext2D, poly: readonly [number, number][], ox = 0, oy = 0): void {
  if (poly.length < 3) return;
  ctx.beginPath();
  ctx.moveTo(poly[0]![0] + ox, poly[0]![1] + oy);
  for (let i = 1; i < poly.length; i++) {
    ctx.lineTo(poly[i]![0] + ox, poly[i]![1] + oy);
  }
  ctx.closePath();
}

export function strokeAlphaForPen(penId: PenId, alphaScale: number): number {
  if (penId === 'pencil') return Math.min(0.68, 0.52 * alphaScale);
  if (penId === 'marker') return Math.min(0.66, 0.58 * alphaScale);
  if (penId === 'fountain') return Math.min(0.96, 0.91 * alphaScale);
  if (penId === 'mech') return Math.min(0.86, 0.8 * alphaScale);
  return Math.min(0.95, 0.88 * alphaScale);
}

export function drawPStroke(
  ctx: CanvasRenderingContext2D,
  pts: readonly StrokePoint[],
  baseW: number,
  color: string,
  alphaScale: number,
  penId: PenId = 'ballpoint',
): void {
  if (!pts || pts.length < 1) return;
  const outline = buildStrokeOutline(pts, baseW, penId);
  if (!outline) return;

  const al = strokeAlphaForPen(penId, alphaScale);
  ctx.save();
  ctx.globalCompositeOperation = 'multiply';
  ctx.fillStyle = color;

  if (outline.dot) {
    ctx.globalAlpha = al;
    ctx.beginPath();
    ctx.arc(outline.x, outline.y, outline.r, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
    return;
  }

  ctx.globalAlpha = al;
  tracePoly(ctx, outline.poly, 0, 0);
  ctx.fill('nonzero');

  if (penId === 'pencil') {
    ctx.globalAlpha = al * 0.28;
    tracePoly(ctx, outline.poly, 0.72, 0.52);
    ctx.fill('nonzero');
  }

  ctx.restore();
}

export function drawRawPreview(
  ctx: CanvasRenderingContext2D,
  rawPts: readonly Point2D[],
  color: string,
  penId: PenId,
): void {
  if (!rawPts || rawPts.length < 2) return;
  ctx.save();
  ctx.globalCompositeOperation = 'multiply';
  ctx.strokeStyle = color;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.lineWidth = penId === 'marker' ? 4.4 : penId === 'mech' ? 1.15 : penId === 'pencil' ? 2.1 : 1.9;
  ctx.globalAlpha = penId === 'pencil' ? 0.42 : penId === 'marker' ? 0.48 : 0.62;
  ctx.beginPath();
  ctx.moveTo(rawPts[0]!.x, rawPts[0]!.y);
  for (let i = 1; i < rawPts.length; i++) ctx.lineTo(rawPts[i]!.x, rawPts[i]!.y);
  ctx.stroke();
  ctx.restore();
}
