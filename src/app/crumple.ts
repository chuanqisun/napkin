import type { NapkinAudioController } from './audio';
import { smoothstep } from './rng';

export interface CrumpleDeps {
  canvas: HTMLCanvasElement;
  wrapEl: HTMLElement;
  overlay: HTMLCanvasElement;
  audio: NapkinAudioController;
  beforeCrumple(): void;
  hasContent(): boolean;
  renderForSnapshot(): void;
  clearAndGenerateFreshSheet(): void;
  liftCupForCrumple(): Promise<boolean | 'cancelled'>;
  returnCupAfterCrumple(): void;
}

function drawTexturedTriangle(
  ctx: CanvasRenderingContext2D,
  img: HTMLCanvasElement,
  u0: number,
  v0: number,
  u1: number,
  v1: number,
  u2: number,
  v2: number,
  x0: number,
  y0: number,
  x1: number,
  y1: number,
  x2: number,
  y2: number,
  shade: number,
): void {
  const denom = u0 * (v1 - v2) + u1 * (v2 - v0) + u2 * (v0 - v1);
  if (Math.abs(denom) < 1e-3) return;

  const cx = (x0 + x1 + x2) / 3;
  const cy = (y0 + y1 + y2) / 3;
  const ex = (x: number, y: number): [number, number] => {
    const dx = x - cx;
    const dy = y - cy;
    const l = Math.hypot(dx, dy) || 1;
    return [x + (dx / l) * 0.75, y + (dy / l) * 0.75];
  };
  const [px0, py0] = ex(x0, y0);
  const [px1, py1] = ex(x1, y1);
  const [px2, py2] = ex(x2, y2);

  ctx.save();
  ctx.beginPath();
  ctx.moveTo(px0, py0);
  ctx.lineTo(px1, py1);
  ctx.lineTo(px2, py2);
  ctx.closePath();
  ctx.clip();

  const a = (x0 * (v1 - v2) + x1 * (v2 - v0) + x2 * (v0 - v1)) / denom;
  const b = (y0 * (v1 - v2) + y1 * (v2 - v0) + y2 * (v0 - v1)) / denom;
  const c = (x0 * (u2 - u1) + x1 * (u0 - u2) + x2 * (u1 - u0)) / denom;
  const d = (y0 * (u2 - u1) + y1 * (u0 - u2) + y2 * (u1 - u0)) / denom;
  const e = (x0 * (u1 * v2 - u2 * v1) + x1 * (u2 * v0 - u0 * v2) + x2 * (u0 * v1 - u1 * v0)) / denom;
  const f = (y0 * (u1 * v2 - u2 * v1) + y1 * (u2 * v0 - u0 * v2) + y2 * (u0 * v1 - u1 * v0)) / denom;

  ctx.transform(a, b, c, d, e, f);
  ctx.drawImage(img, 0, 0);
  ctx.restore();

  if (Math.abs(shade) > 0.01) {
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(px0, py0);
    ctx.lineTo(px1, py1);
    ctx.lineTo(px2, py2);
    ctx.closePath();
    ctx.fillStyle =
      shade > 0
        ? `rgba(255, 249, 238, ${Math.min(0.45, shade * 0.45).toFixed(3)})`
        : `rgba(74, 52, 28, ${Math.min(0.56, -shade * 0.56).toFixed(3)})`;
    ctx.fill();
    ctx.restore();
  }
}

export function createCrumpleRunner(deps: CrumpleDeps): () => Promise<void> {
  let crumpleRaf: number | null = null;
  let crumpleSeq = 0;

  return async function clearAllWithCrumple(): Promise<void> {
    const token = ++crumpleSeq;
    deps.beforeCrumple();

    // 1. Lift drink completely off the napkin BEFORE crumpling, decoding audio in parallel
    const preparePromise = deps.audio.prepareCrumple();
    const liftedDrink = await deps.liftCupForCrumple();
    if (token !== crumpleSeq || liftedDrink === 'cancelled') return;
    if (liftedDrink) {
      await new Promise(r => setTimeout(r, 45));
      if (token !== crumpleSeq) return;
    }
    await preparePromise;
    if (token !== crumpleSeq) return;

    // 2. Snapshot current napkin (with ink & stains, without mug)
    const hadContent = deps.hasContent();
    deps.renderForSnapshot();
    const frameRect = deps.wrapEl.getBoundingClientRect();
    const snap = document.createElement('canvas');
    snap.width = Math.max(1, Math.round(frameRect.width));
    snap.height = Math.max(1, Math.round(frameRect.height));
    snap.getContext('2d')?.drawImage(deps.canvas, 0, 0, snap.width, snap.height);

    // 3. Swap to fresh blank napkin underneath
    deps.clearAndGenerateFreshSheet();

    // 4. Prepare 10x7 triangulated crumpled paper mesh
    if (crumpleRaf) {
      cancelAnimationFrame(crumpleRaf);
      crumpleRaf = null;
    }
    const overlay = deps.overlay;
    const dpr = window.devicePixelRatio || 1;
    overlay.width = Math.round(window.innerWidth * dpr);
    overlay.height = Math.round(window.innerHeight * dpr);
    overlay.style.display = 'block';
    const octx = overlay.getContext('2d');
    if (!octx) return;

    const cols = 10;
    const rows = 7;
    const ballR = Math.min(56, Math.min(frameRect.width, frameRect.height) * 0.135);
    const centerX = frameRect.left + frameRect.width * 0.5;
    const centerY = frameRect.top + frameRect.height * 0.5;

    const verts: Array<{
      u: number;
      v: number;
      sx: number;
      sy: number;
      bx: number;
      by: number;
      bz: number;
      delay: number;
      jitterPhase: number;
    }> = [];
    for (let j = 0; j <= rows; j++) {
      for (let i = 0; i <= cols; i++) {
        const u = (i / cols) * snap.width;
        const v = (j / rows) * snap.height;
        const sx = frameRect.left + u;
        const sy = frameRect.top + v;
        const ang = Math.atan2(sy - centerY, sx - centerX) + (Math.random() - 0.5) * 0.85;
        const normDist = Math.hypot(i / cols - 0.5, j / rows - 0.5) * 1.414;
        const foldR = ballR * (0.28 + 0.72 * Math.pow(normDist, 0.75) * (0.68 + Math.random() * 0.45));
        const bx = Math.cos(ang) * foldR + (Math.random() - 0.5) * ballR * 0.28;
        const by = Math.sin(ang) * foldR + (Math.random() - 0.5) * ballR * 0.28;
        const bz = (Math.random() - 0.45) * 2.1;
        const creaseDelay = Math.random() * 0.2;
        verts.push({
          u,
          v,
          sx,
          sy,
          bx,
          by,
          bz,
          delay: creaseDelay * (1 - normDist * 0.42),
          jitterPhase: Math.random() * Math.PI * 2,
        });
      }
    }

    const borderIdx: number[] = [];
    for (let i = 0; i <= cols; i++) borderIdx.push(i);
    for (let j = 1; j <= rows; j++) borderIdx.push(j * (cols + 1) + cols);
    for (let i = cols - 1; i >= 0; i--) borderIdx.push(rows * (cols + 1) + i);
    for (let j = rows - 1; j >= 1; j--) borderIdx.push(j * (cols + 1));

    const tris: [number, number, number][] = [];
    for (let j = 0; j < rows; j++) {
      for (let i = 0; i < cols; i++) {
        const a = j * (cols + 1) + i;
        const b = a + 1;
        const c = a + (cols + 1);
        const d = c + 1;
        if ((i + j) % 2 === 0) {
          tris.push([a, b, d]);
          tris.push([a, d, c]);
        } else {
          tris.push([a, b, c]);
          tris.push([b, d, c]);
        }
      }
    }

    const TOSS_START = 920;
    const TOTAL_DUR = 1420;

    deps.audio.playCrumple();
    if (hadContent) {
      setTimeout(() => {
        if (token === crumpleSeq) {
          void deps.audio.playClip('argh_stain', { channel: 'vocal', interrupt: true });
        }
      }, 140);
    }

    const tStart = performance.now();
    let cupReturnFired = false;

    const step = (now: number) => {
      const ms = Math.max(0, now - tStart);
      const t = Math.min(1, ms / TOTAL_DUR);

      if (!cupReturnFired && ms >= 1340) {
        cupReturnFired = true;
        if (token === crumpleSeq) deps.returnCupAfterCrumple();
      }

      octx.setTransform(dpr, 0, 0, dpr, 0, 0);
      octx.clearRect(0, 0, window.innerWidth, window.innerHeight);

      let cPhase = 0;
      if (ms < 445) {
        const p1 = ms / 445;
        const snapP = 1 - Math.pow(1 - p1, 2.35);
        cPhase = 0.72 * snapP;
      } else if (ms < 555) {
        cPhase = 0.72 + 0.025 * smoothstep(445, 555, ms);
      } else {
        cPhase = 0.745 + 0.255 * smoothstep(555, 935, ms);
      }

      const squeeze2 = smoothstep(555, 920, ms);
      const pulse = (peakMs: number, width: number) => Math.exp(-Math.pow((ms - peakMs) / width, 2));
      const crunchShake =
        pulse(82, 24) * 4.2 +
        pulse(228, 30) * 2.8 +
        pulse(310, 28) * 2.0 +
        pulse(728, 32) * 3.0 +
        pulse(892, 24) * 2.4;

      const tossPhase = Math.max(0, Math.min(1, (ms - TOSS_START) / (TOTAL_DUR - TOSS_START)));
      const windupY = ms > 820 && ms < 990 ? Math.sin(((ms - 820) / 170) * Math.PI) * 12 : 0;
      const tx = tossPhase * (window.innerWidth * 0.6 + 175);
      const ty = -Math.sin(tossPhase * Math.PI * 0.84) * 230 + tossPhase * tossPhase * 95 + windupY;
      const ballCx = centerX + tx;
      const ballCy = centerY + ty;

      const spin = squeeze2 * 0.42 + Math.pow(tossPhase, 1.55) * 9.2;
      const crushScale = 1.14 - 0.26 * squeeze2;
      const flightScale = 1.0 - 0.74 * Math.pow(tossPhase, 1.15);
      const scale = crushScale * flightScale;
      const alpha = tossPhase > 0.78 ? Math.max(0, 1 - (tossPhase - 0.78) / 0.22) : 1;

      const cosS = Math.cos(spin);
      const sinS = Math.sin(spin);

      const cur = verts.map(vt => {
        const lc = Math.max(0, Math.min(1, (cPhase - vt.delay) / Math.max(0.12, 1 - vt.delay)));
        const ease = lc < 0.5 ? 4 * lc * lc * lc : 1 - Math.pow(-2 * lc + 2, 3) / 2;
        const rx = (vt.bx * cosS - vt.by * sinS) * scale;
        const ry = (vt.bx * sinS + vt.by * cosS) * scale;
        const jx = Math.cos(vt.jitterPhase + ms * 0.04) * crunchShake * ease;
        const jy = Math.sin(vt.jitterPhase + ms * 0.04) * crunchShake * ease;
        const x = vt.sx * (1 - ease) + (ballCx + rx) * ease + jx;
        const y = vt.sy * (1 - ease) + (ballCy + ry) * ease + jy;
        return { u: vt.u, v: vt.v, x, y, z: vt.bz * ease * (0.85 + 0.35 * squeeze2) };
      });

      const sheetShadowAlpha = Math.sin(Math.min(1, cPhase) * Math.PI) * 0.2 * (1 - tossPhase);
      if (sheetShadowAlpha > 0.006) {
        octx.save();
        octx.fillStyle = `rgba(48, 34, 16, ${sheetShadowAlpha.toFixed(3)})`;
        octx.beginPath();
        borderIdx.forEach((vi, idx) => {
          const pt = cur[vi]!;
          if (idx === 0) octx.moveTo(pt.x + 8, pt.y + 12);
          else octx.lineTo(pt.x + 8, pt.y + 12);
        });
        octx.closePath();
        octx.fill();
        octx.restore();
      }

      const shadowAlpha = 0.2 * Math.pow(cPhase, 1.35) * (1 - tossPhase * 0.85);
      if (shadowAlpha > 0.005) {
        octx.save();
        octx.fillStyle = `rgba(50, 36, 18, ${shadowAlpha.toFixed(3)})`;
        octx.beginPath();
        const sRad = ballR * (0.95 + (1 - cPhase) * 1.35) * scale;
        octx.ellipse(ballCx + 10, ballCy + 18 + tossPhase * 90, sRad, sRad * 0.55, 0, 0, Math.PI * 2);
        octx.fill();
        octx.restore();
      }

      const sorted = tris
        .map(([i0, i1, i2]) => {
          const p0 = cur[i0]!;
          const p1 = cur[i1]!;
          const p2 = cur[i2]!;
          const avgZ = (p0.z + p1.z + p2.z) / 3;
          const ux = p1.x - p0.x;
          const uy = p1.y - p0.y;
          const uz = (p1.z - p0.z) * 30;
          const vx = p2.x - p0.x;
          const vy = p2.y - p0.y;
          const vz = (p2.z - p0.z) * 30;
          const nx = uy * vz - uz * vy;
          const ny = uz * vx - ux * vz;
          const nz = ux * vy - uy * vx;
          const nl = Math.hypot(nx, ny, nz) || 1;
          const dot = (nx * -0.5 + ny * -0.6 + Math.abs(nz) * 0.62) / nl;
          const shade = (dot - 0.5) * Math.min(1.15, cPhase * 1.55);
          return { p0, p1, p2, avgZ, shade };
        })
        .sort((a, b) => a.avgZ - b.avgZ);

      octx.save();
      octx.globalAlpha = alpha;
      for (const tr of sorted) {
        drawTexturedTriangle(
          octx,
          snap,
          tr.p0.u,
          tr.p0.v,
          tr.p1.u,
          tr.p1.v,
          tr.p2.u,
          tr.p2.v,
          tr.p0.x,
          tr.p0.y,
          tr.p1.x,
          tr.p1.y,
          tr.p2.x,
          tr.p2.y,
          tr.shade,
        );
      }
      octx.restore();

      if (t < 1) {
        crumpleRaf = requestAnimationFrame(step);
      } else {
        overlay.style.display = 'none';
        crumpleRaf = null;
        if (!cupReturnFired && token === crumpleSeq) {
          cupReturnFired = true;
          deps.returnCupAfterCrumple();
        }
      }
    };

    crumpleRaf = requestAnimationFrame(step);
  };
}
