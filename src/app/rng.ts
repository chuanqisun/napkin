export type Rng = () => number;

export function makeRng(seed: number): Rng {
  let s = (seed >>> 0) || 1;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
}

export function randn(rng: Rng = Math.random): number {
  let u = 0;
  let v = 0;
  while (u <= 1e-7) u = rng();
  while (v <= 1e-7) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.max(0, Math.min(1, (x - edge0) / Math.max(1e-6, edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

export function clampByte(v: number): number {
  return Math.max(0, Math.min(255, Math.round(v)));
}

export function rgbToHex(r: number, g: number, b: number): string {
  return '#' + [r, g, b].map(v => clampByte(v).toString(16).padStart(2, '0')).join('');
}

export function chaikin(pts: [number, number][], iters = 1): [number, number][] {
  let cur = pts;
  for (let k = 0; k < iters; k++) {
    if (cur.length < 3) break;
    const first = cur[0];
    const last = cur[cur.length - 1];
    if (!first || !last) break;
    const next: [number, number][] = [first];
    for (let i = 0; i < cur.length - 1; i++) {
      const a = cur[i];
      const b = cur[i + 1];
      if (!a || !b) continue;
      next.push([0.75 * a[0] + 0.25 * b[0], 0.75 * a[1] + 0.25 * b[1]]);
      next.push([0.25 * a[0] + 0.75 * b[0], 0.25 * a[1] + 0.75 * b[1]]);
    }
    next.push(last);
    cur = next;
  }
  return cur;
}
